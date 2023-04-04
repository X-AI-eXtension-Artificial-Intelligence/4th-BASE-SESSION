import pickle
import argparse

import torch
from soynlp.tokenizer import LTokenizer

from utils import Params, clean_text, display_attention
from model.transformer import Transformer


def predict(config):
    input = clean_text(config.input)
    params = Params('config/params.json')

    # load tokenizer and torchtext Fields
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)
    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)
    eos_idx = eng.vocab.stoi['<eos>']  # eng.vocab.stoi에 <eos>를 eos_idx로 지정

    # select model and load trained model
    model = Transformer(params)                          # model을 받아옴
    model.load_state_dict(torch.load(params.save_model)) # 받아온 model에 pt파일을 통해 weight를 바꿔줌
    model.to(params.device)                              # gpu에 올려주고
    model.eval()                                         # eval로 설정

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(input)                      # input을 tokenizer를 통해 token화
    indexed = [kor.vocab.stoi[token] for token in tokenized]   # index 변환

    source = torch.LongTensor(indexed).unsqueeze(0).to(params.device)  # [1, source_len]: unsqueeze to add batch size
    target = torch.zeros(1, params.max_len).type_as(source.data)       # [1, max_len]

    encoder_output = model.encoder(source)                # model.encoder에 source를 넣어서 output을 뽑음
    next_symbol = eng.vocab.stoi['<sos>']                 # 다음 나올 token은 <sos>

    for i in range(0, params.max_len):
        target[0][i] = next_symbol                        # <sos>로 target 첫번째 지정
        decoder_output, _ = model.decoder(target, source, encoder_output)  # [1, target length, output dim]   #decoder output 생성
        prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1]     # decoder output의 단어별로 max값이 나온 값들 prob으로 지정
        next_word = prob.data[i]                                           # 다음 단어 = prob
        next_symbol = next_word.item()                                     # next_symbol 도 예측된 단어

    eos_idx = int(torch.where(target[0] == eos_idx)[0][0])                 # eos_idx 지정
    target = target[0][:eos_idx].unsqueeze(0)                              # target output을 eos_idx까지의 idx들을 반환

    # translation_tensor = [target length] filed with word indices
    target, attention_map = model(source, target)                          # 위 과정이랑 똑같은 과정 같은디;
    target = target.squeeze(0).max(dim=-1)[1]

    translated_token = [eng.vocab.itos[token] for token in target]         # token idxs를 words로 변환 
    translation = translated_token[:translated_token.index('<eos>')]       # <eos> idx가 나오기 전까지 단어 반환
    translation = ' '.join(translation)                                    # 단어 합치기

    print(f'kor> {config.input}')
    print(f'eng> {translation.capitalize()}')
    display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kor-Eng Translation prediction')
    parser.add_argument('--input', type=str, default='내일 여자친구를 만나러 가요')
    option = parser.parse_args()

    predict(option)
