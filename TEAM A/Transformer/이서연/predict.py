import pickle
import argparse

import torch
from soynlp.tokenizer import LTokenizer

from utils import Params, clean_text, display_attention
from model.transformer import Transformer


def predict(config):
    input = clean_text(config.input) ## 입력 문장 전처리리
    params = Params('config/params.json') ## 하이퍼파라미터

    # load tokenizer and torchtext Fields
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb') ## pickle 형태로 저장된 tokenizer 불러옴옴
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores) ## tokenizer 생성성

    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)
    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)
    eos_idx = kor.vocab.stoi['<eos>'] #!! ## <eos> token의 index

    # select model and load trained model
    model = Transformer(params) ## Transformer 모델 생성성
    model.load_state_dict(torch.load(params.save_model)) ## 저장된 모델 가중치 불러오기기
    model.to(params.device)
    model.eval() ## 추론 모드로 설정정

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(input) ## 입력 문장 토큰화화
    indexed = [eng.vocab.stoi[token] for token in tokenized] ## token을 index로 변환환

    ## source, target 텐서 생성
    source = torch.LongTensor(indexed).unsqueeze(0).to(params.device)  # [1, source_len]: unsqueeze to add batch size
    target = torch.zeros(1, params.max_len).type_as(source.data)       # [1, max_len]

    encoder_output = model.encoder(source) ## encoder에 입력 source를 넣어 출력값 생성성
    next_symbol = kor.vocab.stoi['<sos>'] #!! ## <sos> token의 index 가져오기기

    for i in range(0, params.max_len):
        target[0][i] = next_symbol ## i번째 index에 next symbol 대입
        decoder_output, _ = model.decoder(target, source, encoder_output)  # [1, target length, output dim]
        prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1] ## decoding 결과 중 확률이 가장 높은 단어의 index 추출
        next_word = prob.data[i] ## 추출한 index를 next word에 대입입
        next_symbol = next_word.item() ## next word를 int형으로 변환하여 next symbol에 대입입

    eos_idx = int(torch.where(target[0] == eos_idx)[0][0]) ## target에서 <eos>의 위치를 찾음음
    target = target[0][:eos_idx].unsqueeze(0) ## <eos> 이전까지의 target만 사용하여 차원 맞춰줌

    # translation_tensor = [target length] filed with word indices
    target, attention_map = model(source, target) ## 입력 시퀀스에 대한 번역 출력력
    target = target.squeeze(0).max(dim=-1)[1]

    translated_token = [kor.vocab.itos[token] for token in target] #!! ## target tensor 내 단어 인덱스를 영어 단어로 변환환
    translation = translated_token[:translated_token.index('<eos>')] ## 번역 결과에서 <eos> token 이전까지의 단어들만 선택택
    translation = ' '.join(translation) ## 번역된 단어 결합합

    ## 번역된 결과와 관련된 attention map 시각화화
    print(f'eng> {config.input}') #!!
    print(f'kor> {translation.capitalize()}') #!!
    display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kor-Eng Translation prediction')
    parser.add_argument('--input', type=str, default='Good morning') ## 입력 인자 추가, 기본값 설정정
    option = parser.parse_args()

    predict(option)
