import pickle
import numpy as np
import torch
import torch.nn as nn

pickle_eng = open('pickles/eng.pickle', 'rb')
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Decoder에서 사용되는 subsequent mask를 생성하는 함수 ->  현재 위치 이후의 단어들을 masking 처리
def create_subsequent_mask(target):
    ## 대각선 아래쪽 값이 모두 0
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
    batch_size, target_length = target.size()

    # torch.triu returns the upper triangular part of a matrix based on user defined diagonal
    ## subsequent mask를 bool 형태로 반환
    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device)
    # subsequent_mask = [target length, target length]

    # repeat subsequent_mask 'batch size' times to cover all data instances in the batch
    ## batch size만큼 반복하여 mask를 만듦
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    # subsequent_mask = [batch size, target length, target length]

    return subsequent_mask

## Encoder에서 self-attention을 수행할 때 사용하는 mask를 생성하는 함수
def create_source_mask(source):
    """
    create masking tensor for encoder's self attention
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :return: source mask
    """
    source_length = source.shape[1]

    # create boolean tensors which will be used to mask padding tokens of both source and target sentence
    ## 패딩 토큰 부분을 True로, 나머지를 False로 -> boolean tensor
    source_mask = (source == pad_idx)
    # source_mask = [batch size, source length]

    # repeat sentence masking tensors 'sentence length' times
    ## tensor의 길이만큼 복사하여 형태 맞춰줌
    ## 각 문장의 패딩 token을 masking하는데 사용됨
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)
    # source_mask = [batch size, source length, source length]

    return source_mask

## decoder의 self-atention / encoder의 출력에 대한 decoder의 attention에 사용되는 mask를 생성하는 함수
def create_target_mask(source, target):
    """
    create masking tensor for decoder's self attention and decoder's attention on the output of encoder
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :param target: [batch size, target length]
    :return:
    """
    target_length = target.shape[1] ## target tensor의 길이
    
    ## subsequent_mask 생성
    subsequent_mask = create_subsequent_mask(target)
    # subsequent_mask = [batch size, target length, target length]
    
    ## source_mask와 target_mask를 각각 패딩 token으로 구성
    source_mask = (source == pad_idx)
    target_mask = (target == pad_idx)
    # target_mask    = [batch size, target length]

    # repeat sentence masking tensors 'sentence length' times
    ## 3차원 tensor로 확장 후 target_length번째 차원으로 source_mask를 복사
    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
    ## 
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

    # combine <pad> token masking tensor and subsequent masking tensor for decoder's self attention
    ## target_mask와 subsequent_mask를 합쳐 decoder의 self-attention에서 패딩 token과 subsequent token을 모두 마스킹
    target_mask = target_mask | subsequent_mask
    # target_mask = [batch size, target length, target length]
    return target_mask, dec_enc_mask


## 입력 문장의 단어 위치 정보를 담은 position vector를 생성하는 함수
def create_position_vector(sentence):
    """
    create position vector which contains positional information
    0th position is used for pad index
    :param sentence: [batch size, sentence length]
    :return: [batch size, sentence length]
    """
    # sentence = [batch size, sentence length]
    batch_size, _ = sentence.size() ## batch size 추출
    ## 0으로 초기화한 후 각 단어의 위치와 값을 순회하면서 word가 pad_idx가 아닐 경우 pos+1 값을 넣어줌
    pos_vec = np.array([(pos+1) if word != pad_idx else 0
                        for row in range(batch_size) for pos, word in enumerate(sentence[row])])
    ## reshape 한 후 long 타입의 tensor로 변환
    pos_vec = pos_vec.reshape(batch_size, -1)
    pos_vec = torch.LongTensor(pos_vec).to(device)
    return pos_vec

## positional encoding에 사용될 테이블을 생성하는 역할
def create_positional_encoding(max_len, hidden_dim):
    # PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim))
    # PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim))
    ## 모든 위치와 모든 hidden dimension에서 반복하여 계산
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(max_len) for i in range(hidden_dim)])
    # sinusoid_table = [max len * hidden dim]
    
    ## 생성된 테이블을 max_len과 hidden_dim의 크기로 reshape
    sinusoid_table = sinusoid_table.reshape(max_len, -1)
    # sinusoid_table = [max len, hidden dim]
    
    ## 짝수 차원과 홀수 차원에 대해 각각 sin과 cos 계산
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # calculate pe for even dimension
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # calculate pe for odd dimension

    # convert numpy based sinusoid table to torch.tensor and repeat it 'batch size' times
    ## numpy 기반 테이블을 torch tensor로 변환
    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.

    return sinusoid_table

## layer의 가중치와 편향 초기화
def init_weight(layer):
    nn.init.xavier_uniform_(layer.weight)
    ## bias가 None이 아니라면 0으로 초기화
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
