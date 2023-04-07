import pickle
import numpy as np
import torch
import torch.nn as nn

# pickle_eng 파일에서 eng.vocab.stoi 정보를 불러옴
pickle_eng = open('pickles/eng.pickle', 'rb')
eng = pickle.load(pickle_eng)

# eng.vocab.stoi 딕셔너리에서 <pad>의 인덱스 번호를 불러와 변수에 저장
pad_idx = eng.vocab.stoi['<pad>']

# CUDA를 사용할 수 있다면 GPU를 사용하고, 사용할 수 없다면 CPU를 사용하여 device에 할당
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_subsequent_mask(target):
    #디코더의 마스크 생성
    """
    if target length is 5 and diagonal is 1, this function returns
        [[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]]
    :param target: [batch size, target length]
    :return:
        # Decoder Self-Attention에서 사용하는 마스크 생성 함수

    """
    # target의 크기(배치 크기, 타겟 문장 길이) 추출
    batch_size, target_length = target.size()
    
    # diagonal을 1로 설정하면 대각선과 그 위쪽 부분이 1, 그 아래는 0인 행렬 생성
    # 이때 bool()을 사용하여 행렬의 값을 True 또는 False로 변환
    # torch.triu returns the upper triangular part of a matrix based on user defined diagonal
    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device)
    # subsequent_mask = [target length, target length]

    # repeat subsequent_mask 'batch size' times to cover all data instances in the batch
    # 마스크를 배치 크기만큼 복사하여 사용할 수 있도록 함
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    # subsequent_mask = [batch size, target length, target length]

    return subsequent_mask


def create_source_mask(source):
    # Encoder Self-Attention에서 사용하는 마스크 생성 함수

    """
    create masking tensor for encoder's self attention
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :return: source mask
    """
    # 소스 문장의 길이 추출

    source_length = source.shape[1]

    # create boolean tensors which will be used to mask padding tokens of both source and target sentence
    # 소스 문장의 패딩 토큰을 체크하여 True or False로 구성된 마스크 생성
    source_mask = (source == pad_idx)
    # source_mask = [batch size, source length]

    # repeat sentence masking tensors 'sentence length' times
        # 마스크를 소스 문장의 길이만큼 복사하여 사용할 수 있도록 함
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)
    # source_mask = [batch size, source length, source length]

    return source_mask


def create_target_mask(source, target):
    """
    create masking tensor for decoder's self attention and decoder's attention on the output of encoder
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :param target: [batch size, target length]
    :return:
    """
    # 타겟 문장 길이(target_length)를 구합니다.
    target_length = target.shape[1]

    # 디코더의 self attention과 인코더 출력에 대한 디코더 attention에 대한 마스킹 텐서를 생성합니다.
    subsequent_mask = create_subsequent_mask(target)
    # subsequent_mask = [batch size, target length, target length]

    # 소스 문장에 있는 <pad> 토큰을 마스킹합니다.
    source_mask = (source == pad_idx)
    target_mask = (target == pad_idx)
    # target_mask    = [batch size, target length]

    # dec_enc_mask: 디코더에서 인코더 출력에 대한 attention에 사용할 마스킹 텐서를 생성합니다.
    # target_mask: 디코더의 self attention에 사용할 마스킹 텐서를 생성합니다.
    # 두 마스킹 텐서 모두 1의 값을 가지는 <pad> 토큰을 마스킹합니다.
    # 마스킹 텐서를 반복해 문장 길이만큼 만듭니다.
    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

    # subsequent_mask와 target_mask를 합치는데, 두 텐서 중 하나가 True이면 True로 설정합니다.
    target_mask = target_mask | subsequent_mask
    # target_mask = [batch size, target length, target length]
    return target_mask, dec_enc_mask


def create_position_vector(sentence):
    """
    create position vector which contains positional information
    0th position is used for pad index
    :param sentence: [batch size, sentence length]
    :return: [batch size, sentence length]
    """
    # 입력 문장에서 배치 크기와 문장 길이를 가져옵니다.
    batch_size, _ = sentence.size()
    
    # pos_vec: 문장에서 각 단어의 위치 정보를 담은 위치 텐서를 생성합니다.
    # pad_idx에 해당하는 위치는 0으로 설정합니다.
    pos_vec = np.array([(pos+1) if word != pad_idx else 0
                        for row in range(batch_size) for pos, word in enumerate(sentence[row])])
    pos_vec = pos_vec.reshape(batch_size, -1)
    pos_vec = torch.LongTensor(pos_vec).to(device)
    return pos_vec



def create_positional_encoding(max_len, hidden_dim):
    # positional encoding의 각 차원(dimension)은 주기적인 형태를 띄도록 설계됩니다.
    # 따라서 pos와 i의 값을 이용하여 sinusoid_table을 생성합니다.
    # 이 때, 각 차원마다 주기를 다르게 하기 위해 hidden_dim으로 나누어 줍니다.
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(max_len) for i in range(hidden_dim)])
    # sinusoid_table은 [max_len * hidden_dim] 크기의 numpy array입니다.

    # 각 pos와 i에 대해 구해진 값을 이용해 sinusoid_table을 구성합니다.
    # 각 dimension마다 짝수번째 값과 홀수번째 값을 구분하여 sin/cos를 적용합니다.
    sinusoid_table = sinusoid_table.reshape(max_len, -1)
    # 이 때, reshape 함수를 이용해 [max_len, hidden_dim] 크기로 변경합니다.

    # 각 dimension마다 짝수/홀수 번째 값에 대해 sin/cos를 적용한 후, sinusoid_table에 저장합니다.
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 짝수번째 차원
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 홀수번째 차원

    # numpy array를 pytorch tensor로 변경한 후, 0번째 position의 값을 0으로 변경합니다.
    # (0번째 position은 padding에 사용합니다.)
    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.

    return sinusoid_table

def init_weight(layer):
    # 가중치 행렬을 xavier_uniform 분포를 따르도록 초기화합니다.
    nn.init.xavier_uniform_(layer.weight)
    # 편향(bias)이 존재한다면, 0으로 초기화합니다.
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
