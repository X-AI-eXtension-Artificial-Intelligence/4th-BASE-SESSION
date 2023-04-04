import pickle
import argparse

import torch
from soynlp.tokenizer import LTokenizer

from utils import Params, clean_text, display_attention
from model.transformer import Transformer


def predict(config):
    input = clean_text(config.input)
    params = Params('config/params.json')

    # load tokenizer and torchtext Fields (데이터 불러오기)
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores) #LTokenizer -> 나는 에서 나는 명사고, 는은 조사  -> 이런식으로 더 세세하게 나눠줌

    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)
    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)
    eos_idx=kor.vocab.stoi['<eos>']
    #kor로 바꿔줌

    # select model and load trained model (모델 불러오기)
    model = Transformer(params)
    model.load_state_dict(torch.load(params.save_model))
    model.to(params.device)
    model.eval()

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(input) #입력 텍스트를 토큰화
    indexed = [eng.vocab.stoi[token] for token in tokenized] 
    #바꿈

    source = torch.LongTensor(indexed).unsqueeze(0).to(params.device)  # [1, source_len]: unsqueeze to add batch size
    target = torch.zeros(1, params.max_len).type_as(source.data)       # [1, max_len]
    #처음엔 아무것도 없음(target 값에) -> 순차적으로 output을 뽑아내는
    
    
    encoder_output = model.encoder(source)# encoder 통과
    next_symbol = kor.vocab.stoi['<sos>'] #디코더 부분 kor로 바꿔줌

    
    #transformer 모델을 통해 번역 결과를 예측하는 부분
    for i in range(0, params.max_len):
        target[0][i] = next_symbol #decoder의 예측결과(next_symbol)을 target에 저장
        decoder_output, _ = model.decoder(target, source, encoder_output)  # [1, target length, output dim]
        # target, source, encoder_output을 입력으로 받아 디코더 실행
        # 첫번째 target은 sos
        #64단어, [1,64,19545]
        prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1] # 디코더 출력값의 마지막 차원(단어집합크기)에서 가장 큰 값과 해당 위치 구함
        next_word = prob.data[i] #현재에서 예측된 다음 단어를 구함
        next_symbol = next_word.item() # next word 값을 스칼라 값으로 변환한 값을 저장


    eos_idx = int(torch.where(target[0] == params.eos_idx)[0][0])
    #target에서 eos 토큰이 처음 나오는 위치를 찾아 eos_idx에 저장시킴

    target = target[0][:eos_idx].unsqueeze(0) # target에서 eos 토큰 이전까지의 부분 시퀀스만 선택해서 target을 재설정 -> 이 부분 시퀀스가 번역 결과가 됨.
    #sos를 넣은 sequence
    

    # translation_tensor = [target length] filed with word indices
    target, attention_map = model(source, target) #source: 입력된 한국어 문장
    #eos는 없고 sos의 부분의 타겟
    
    #source를 model에 넣어 target 예측 => attention map도 계산됨. 같이 저장
    target = target.squeeze(0).max(dim=-1)[1] # max를 통해 각 위치에서 가장 큰 값을 가지는 단어의 index(위치)를 구함
    #
    #decoder을 거쳐 나온 target 값

    translated_token = [kor.vocab.itos[token] for token in target] # target을 translation에 저장
    #각 해당하는 숫자를 (단어)임베딩로 변환해서 영어로 바꿔줌
    ## 바꿔줌 kor로
    
    translation = translated_token[:translated_token.index('<eos>')] # #eos 이후의 단어들은 제거 (앞에 embedding을 거쳐서 나온 애들 값을 출력)
    
    translation = ' '.join(translation) # 단어들을 하나의 문장으로 합침

    print(f'eng> {config.input}')
    print(f'kor> {translation.capitalize()}')
    display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1]) #시각화


#predict 실행
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eng-Kor Translation prediction')
    parser.add_argument('--input', type=str, default="Hi I'm your friend")
    option = parser.parse_args()

    predict(option)
