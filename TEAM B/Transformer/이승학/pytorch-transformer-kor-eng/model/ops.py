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
    batch_size, target_length = target.size()                                                           # target값이 들어오면 batch_size, target_length를 받음

    # torch.triu returns the upper triangular part of a matrix based on user defined diagonal
    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device)# matrix의 upper triangular part를 반환
    # subsequent_mask = [target length, target length]

    # repeat subsequent_mask 'batch size' times to cover all data instances in the batch
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)                             # 반환된 mask matrix를 (batch_size, 1, 1)으로 모형을 맞춰줌
    # subsequent_mask = [batch size, target length, target length]

    return subsequent_mask


def create_source_mask(source):
    """
    create masking tensor for encoder's self attention
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :return: source mask
    """
    source_length = source.shape[1]                                               # source 문장의 길이를 저장

    # create boolean tensors which will be used to mask padding tokens of both source and target sentence
    source_mask = (source == pad_idx)                                             # source_mask에서 적용될 pad_idx 생성
    # source_mask = [batch size, source length]

    # repeat sentence masking tensors 'sentence length' times
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)            #{pad, source_length, pad}로 모형을 맞춰줌
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
    target_length = target.shape[1]                                                                 # target 문장의 길이를 저장

    subsequent_mask = create_subsequent_mask(target)                                                # create_subsequent_mask함수 활용 mask matrix를 만듬
    # subsequent_mask = [batch size, target length, target length]

    source_mask = (source == pad_idx)                                                               # source_mask와 target_mask에서 적용될 pad_idx 생성>?
    target_mask = (target == pad_idx)
    # target_mask    = [batch size, target length]

    # repeat sentence masking tensors 'sentence length' times
    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)                             # dec_enc_mask와 target_mask를 {pad, source_length, pad}로 모형을 맞춰줌
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

    # combine <pad> token masking tensor and subsequent masking tensor for decoder's self attention
    target_mask = target_mask | subsequent_mask                                                     # decoder의 self attention을 위한 target_mask를 padding masking과 subsequent_mask을 합침
    # target_mask = [batch size, target length, target length]
    return target_mask, dec_enc_mask


def create_position_vector(sentence):
    """
    create position vector which contains positional information
    0th position is used for pad index
    :param sentence: [batch size, sentence length]
    :return: [batch size, sentence length]
    """
    # sentence = [batch size, sentence length]
    batch_size, _ = sentence.size()                                                              # input 들어오면 batch_size 받기
    pos_vec = np.array([(pos+1) if word != pad_idx else 0
                        for row in range(batch_size) for pos, word in enumerate(sentence[row])]) # batch 단위에서 embedding된 각 문장마다 position_vector를 생성해줌.
    pos_vec = pos_vec.reshape(batch_size, -1)                                                    # pad_idx가 아니면 pos+1, 아니면 0으로 array를 생성
    pos_vec = torch.LongTensor(pos_vec).to(device)                                               # longTensor로 변환 후 GPU로 올려줌
    return pos_vec


def create_positional_encoding(max_len, hidden_dim):
    # PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim))
    # PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim))
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(max_len) for i in range(hidden_dim)]) # Positional embedding을 위한 table 생성
    # sinusoid_table = [max len * hidden dim]

    sinusoid_table = sinusoid_table.reshape(max_len, -1)
    # sinusoid_table = [max len, hidden dim]

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # calculate pe for even dimension
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # calculate pe for odd dimension

    # convert numpy based sinusoid table to torch.tensor and repeat it 'batch size' times
    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.

    return sinusoid_table


def init_weight(layer):
    nn.init.xavier_uniform_(layer.weight)   # Xavier uniform init 으로 weight를 uniform distribution
    if layer.bias is not None:              # bias 추가
        nn.init.constant_(layer.bias, 0)
