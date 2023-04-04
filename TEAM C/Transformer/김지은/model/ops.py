import pickle
import numpy as np
import torch
import torch.nn as nn

pickle_eng = open('pickles/eng.pickle', 'rb')
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
def create_subsequent_mask(target): #target 시퀀스-> 마스크 생성 (각 위치에서 이전 위치들에 대한 정보만 사용하도록 하는 역할)
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
    batch_size, target_length = target.size() # 배치 크기, 시퀀스 길이 input으로 받음

    # torch.triu returns the upper triangular part of a matrix based on user defined diagonal
    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device) # 그림과 같이 값 채워줌 (아래:0으로, 위는1로)
    # subsequent_mask = [target length, target length]

    # repeat subsequent_mask 'batch size' times to cover all data instances in the batch
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1) #subsequent_mask 텐서에 차원추가, 배치 크기만큼 반복
    # subsequent_mask = [batch size, target length, target length]

    return subsequent_mask


def create_source_mask(source): #인코더의 self attention을 위한 마스크 생성 함수
    """
    create masking tensor for encoder's self attention
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :return: source mask
    """
    source_length = source.shape[1]

    # create boolean tensors which will be used to mask padding tokens of both source and target sentence
    source_mask = (source == pad_idx) #패딩토큰인 부분은 true, 아니면 false
    # source_mask = [batch size, source length]

    # repeat sentence masking tensors 'sentence length' times
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1) #차원 추가 및 반복
    # source_mask = [batch size, source length, source length]

    return source_mask


def create_target_mask(source, target): # 디코더의 self attention, 인코더의 출력에 대한 디코더의 attention을 위한 마스크 생성함수
    # attention masking
    """
    create masking tensor for decoder's self attention and decoder's attention on the output of encoder
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :param target: [batch size, target length]
    :return:
    """
    target_length = target.shape[1]

    subsequent_mask = create_subsequent_mask(target)
    # subsequent_mask = [batch size, target length, target length]

    source_mask = (source == pad_idx)
    target_mask = (target == pad_idx)
    # target_mask    = [batch size, target length]

    # repeat sentence masking tensors 'sentence length' times
    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

    # combine <pad> token masking tensor and subsequent masking tensor for decoder's self attention
    target_mask = target_mask | subsequent_mask
    # target_mask = [batch size, target length, target length]
    return target_mask, dec_enc_mask


def create_position_vector(sentence): #position vector 생성 함수
    """
    create position vector which contains positional information
    0th position is used for pad index
    :param sentence: [batch size, sentence length]
    :return: [batch size, sentence length]
    """
    # sentence = [batch size, sentence length]
    batch_size, _ = sentence.size()
    pos_vec = np.array([(pos+1) if word != pad_idx else 0 # 패딩 토큰인 경우 위치 정보는 0으로 설정
                        for row in range(batch_size) for pos, word in enumerate(sentence[row])])
    pos_vec = pos_vec.reshape(batch_size, -1)
    pos_vec = torch.LongTensor(pos_vec).to(device)
    return pos_vec


def create_positional_encoding(max_len, hidden_dim): #positional encoding 함수
    # PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim))
    # PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim))
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(max_len) for i in range(hidden_dim)])
    # sinusoid_table = [max len * hidden dim]

    sinusoid_table = sinusoid_table.reshape(max_len, -1)
    # sinusoid_table = [max len, hidden dim]

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # calculate pe for even dimension
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # calculate pe for odd dimension

    # convert numpy based sinusoid table to torch.tensor and repeat it 'batch size' times
    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.

    return sinusoid_table


def init_weight(layer): #가중치 초기화 함수
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
