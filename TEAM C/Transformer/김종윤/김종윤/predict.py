import pickle
import argparse
import spacy

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
    
    #tokenizer = LTokenizer(scores=cohesion_scores) # 수정, 영어로 들어올거기에 다른 tokenizer 사용해보자
    tokenizer = spacy.load('en')
    
    pickle_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(pickle_kor)
    pickle_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(pickle_eng)
    eos_idx = kor.vocab.stoi['<eos>'] # 수정, eos는 한글 eos의 index를 가져야함
    
    
    # select model and load trained model
    model = Transformer(params)
    model.load_state_dict(torch.load(params.save_model))
    model.to(params.device)
    model.eval()

    # convert input into tensor and forward it through selected model
    
    #tokenized = tokenizer.tokenize(input) # input이 영어기에 여기도 수정해야되겠지?
    tokenized = list(tokenizer(input))
    
    indexed = [eng.vocab.stoi[token] for token in tokenized] # 수정, source로 eng가 들어가야하기 때문에 eng index 필요
    source = torch.LongTensor(indexed).unsqueeze(0).to(params.device)  # [1, source_len]: unsqueeze to add batch size
    target = torch.zeros(1, params.max_len).type_as(source.data)       # [1, max_len]
    
    encoder_output = model.encoder(source) 
    next_symbol = kor.vocab.stoi['<sos>'] # decoder 첫번째에 들어갈 sos 인덱스, 수정, sos 는 kor로 바꿔야됨
    
    for i in range(0, params.max_len):
        target[0][i] = next_symbol
        decoder_output, _ = model.decoder(target, source, encoder_output)
        #print("decoder output:",decoder_output,'\n', decoder_output.shape)
        # [1, target length, output dim]
        # ex : [1, 64, 19545], target의 최대문장길이를 64단어로 설정, 
        prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1] # 64개의 위치에서 가장 나올 확률이 높은 단어(여기서는 영어)의 인덱스 값, 인덱스는 19544까지 존재. 가장 높은 확률이 높은 인덱스를 골랐으므로 prob의 shape는 64
        #print(f"probability shape : {prob} \n {prob.shape}")
        next_word = prob.data[i]
        #print(f"{i}번째 단어 인덱스? {next_word}")
        next_symbol = next_word.item()
        #print(f"{i}번째 단어? {next_symbol}")
    
    eos_idx = int(torch.where(target[0] == eos_idx)[0][0])
    target = target[0][:eos_idx].unsqueeze(0) # decoder 첫번째에 들어갈 target값들(eos 제외)
    #print(f"이건 인덱스인가? : {target}")
    #print(target.shape)
    # translation_tensor = [target length] filed with word indices
    target, attention_map = model(source, target) 
    #print(f"이건 뭐야? : {target.shape}")
    target = target.squeeze(0).max(dim=-1)[1] # 모든 디코더를 거쳐 나온 target 값들, eos로 끝남
    #print(f"이건 확률값인가? : {target}")

    translated_token = [kor.vocab.itos[token] for token in target] #최종 타겟의 index 값들을 단어 value로 바꿔줌, 수정, 영어->한글
    #print("이건 번역?",translation_token 
    translation = translated_token[:translated_token.index('<eos>')]
    translation = ' '.join(translation)
    

    print(f'eng> {config.input}')
    print(f'kor> {translation.capitalize()}')
    display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eng-Kor Translation prediction')
    parser.add_argument('--input', type=str, default='I want to go home')
    option = parser.parse_args()

    predict(option)
