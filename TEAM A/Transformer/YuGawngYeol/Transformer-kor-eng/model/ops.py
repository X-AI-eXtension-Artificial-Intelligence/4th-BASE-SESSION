import pickle
import numpy as np
import torch
import torch.nn as nn

pickle_eng = open('pickles/eng.pickle', 'rb')
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_subsequent_mask(target):
    """
    if target length is 5 and diagonal is 1, this function returns
        [[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]]
    :param target: [batch size, target length]
    :return:
    """
    batch_size, target_length = target.size() #batch,target_length 지정

    # torch.triu returns the upper triangular part of a matrix based on user defined diagonal
    # Torch.triu는 사용자 정의 대각선을 기준으로 행렬의 위쪽 삼각형 부분을 반환합니다
    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device) # 삼각 행렬 생성
    # subsequent_mask = [target length, target length]

    # repeat subsequent_mask 'batch size' times to cover all data instances in the batch
    # subsequent_mask 'batch size'를 반복하여 배치의 모든 데이터 인스턴스를 포함합니다
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    # subsequent_mask = [batch size, target length, target length]

    return subsequent_mask


def create_source_mask(source):
    """
    create masking tensor for encoder's self attention
    인코더의 자체 주의를 위한 마스킹 텐서 생성
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :return: source mask
    """
    source_length = source.shape[1]

    # create boolean tensors which will be used to mask padding tokens of both source and target sentence
    source_mask = (source == pad_idx) # pad가 있으면 마스크 처리
    # source_mask = [batch size, source length]

    # repeat sentence masking tensors 'sentence length' times
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)
    # source_mask = [batch size, source length, source length]

    return source_mask


def create_target_mask(source, target):
    """
    create masking tensor for decoder's self attention and decoder's attention on the output of encoder
    인코더 출력에 대한 디코더의 자체 주의 및 디코더의 주의를 위한 마스킹 텐서 생성
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :param target: [batch size, target length]
    :return:
    """
    target_length = target.shape[1]

    subsequent_mask = create_subsequent_mask(target) # 함수 이동 -> 삼각 행렬 target 생성
    # subsequent_mask = [batch size, target length, target length]

    source_mask = (source == pad_idx) # source에서 pad
    target_mask = (target == pad_idx) # target에서 pad
    # target_mask    = [batch size, target length]

    # repeat sentence masking tensors 'sentence length' times
    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1) # source
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1) # target mask

    # combine <pad> token masking tensor and subsequent masking tensor for decoder's self attention
    target_mask = target_mask | subsequent_mask # 둘 중 하나라도 1이면 1
    # target_mask = [batch size, target length, target length]
    return target_mask, dec_enc_mask


def create_position_vector(sentence):
    """
    create position vector which contains positional information
    0th position is used for pad index
    위치 정보를 포함하는 위치 벡터 생성
    패드 인덱스에 0번째 위치가 사용됩니다
    :param sentence: [batch size, sentence length]
    :return: [batch size, sentence length]
    """
    # sentence = [batch size, sentence length]
    batch_size, _ = sentence.size()
    pos_vec = np.array([(pos+1) if word != pad_idx else 0
                        for row in range(batch_size) for pos, word in enumerate(sentence[row])])  # pad를 0으로 나머지 1씩 미룸
    pos_vec = pos_vec.reshape(batch_size, -1)
    pos_vec = torch.LongTensor(pos_vec).to(device)
    return pos_vec


def create_positional_encoding(max_len, hidden_dim): # positional encoding 생성
    # PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim))
    # PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim))
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(max_len) for i in range(hidden_dim)]) # 안에 식 생성
    # sinusoid_table = [max len * hidden dim]

    sinusoid_table = sinusoid_table.reshape(max_len, -1)
    # sinusoid_table = [max len, hidden dim]

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # calculate pe for even dimension 짝수 차원에 대한 pe 계산
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # calculate pe for odd dimension 홀수 차원에 대한 pe 계산

    # convert numpy based sinusoid table to torch.tensor and repeat it 'batch size' times
    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.

    return sinusoid_table


def init_weight(layer): # 가중치 초기화 
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
