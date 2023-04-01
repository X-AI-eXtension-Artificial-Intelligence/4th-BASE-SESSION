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
    eos_idx = kor.vocab.stoi['<eos>'] #!! ## <eos> token의 index

    # select model and load trained model
    model = Transformer(params) 
    model.load_state_dict(torch.load(params.save_model)) 
    model.to(params.device)
    model.eval() 

    # convert input into tensor and forward it through selected model
    tokenized = tokenizer.tokenize(input) 
    indexed = [eng.vocab.stoi[token] for token in tokenized] 

    source = torch.LongTensor(indexed).unsqueeze(0).to(params.device)  # [1, source_len]: unsqueeze to add batch size
    target = torch.zeros(1, params.max_len).type_as(source.data)       # [1, max_len]

    encoder_output = model.encoder(source) 
    next_symbol = kor.vocab.stoi['<sos>'] 

    for i in range(0, params.max_len):
        target[0][i] = next_symbol 
        decoder_output, _ = model.decoder(target, source, encoder_output)  
        prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1] 
        next_word = prob.data[i] 
        next_symbol = next_word.item() 

    eos_idx = int(torch.where(target[0] == eos_idx)[0][0]) 
    target = target[0][:eos_idx].unsqueeze(0) 

    # translation_tensor = [target length] filed with word indices
    target, attention_map = model(source, target) 
    target = target.squeeze(0).max(dim=-1)[1]

    translated_token = [kor.vocab.itos[token] for token in target] 
    translation = translated_token[:translated_token.index('<eos>')] 
    translation = ' '.join(translation)

    print(f'eng> {config.input}') #!!
    print(f'kor> {translation.capitalize()}') #!!
    display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kor-Eng Translation prediction')
    parser.add_argument('--input', type=str, default='Good morning') 
    option = parser.parse_args()

    predict(option)
