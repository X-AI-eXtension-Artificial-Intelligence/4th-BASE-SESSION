import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils import convert_to_dataset

from torchtext import data as ttd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

## 한국어 문장 토큰화
def build_tokenizer(): 
    """
    Train soynlp tokenizer which will be used to tokenize Korean input sentence
    """
    print(f'Now building soy-nlp tokenizer . . .')

    data_dir = Path().cwd() / 'data' ## 데이터 읽어옴
    train_file = os.path.join(data_dir, 'corpus.csv')

    df = pd.read_csv(train_file, encoding='utf-8')

    # if encounters non-text row, we should skip it
    ## korean 컬럼의 내용만 가져와, 리스트에 넣음
    kor_lines = [row.korean
                 for _, row in df.iterrows() if type(row.korean) == str]

    ## 주어진 문장에서 단어 추출 알고리즘 학습
    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(kor_lines)

    word_scores = word_extractor.extract() ## 추출된 단어들과 그 단어들의 특징 정보가 저장되어 있음
    cohesion_scores = {word: score.cohesion_forward
                       for word, score in word_scores.items()} ## 단어간 유사도 계산

    ## 학습된 토큰화 모델 저장장
    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)

## 한국어와 영어 데이터를 전처리하기 위한 단어 사전을 만듦
def build_vocab(config):
    """
    Build vocab used to convert input sentence into word indices using soynlp and spacy tokenizer
    Args:
        config: configuration containing various options
    """
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    # include lengths of the source sentences to use pack pad sequence
    ## source와 target에 대해 필요한 설정을 해줌
    eng = ttd.Field(tokenize='spacy', #!!
                    lower=True,
                    batch_first=True) 

    ## 문장과 끝을 나타내는 token 설정
    kor = ttd.Field(tokenize=tokenizer.tokenize, #!!
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_file, encoding='utf-8')
    ## DataFrame을 source와 target 필드로 나누어서 Example 형식으로 변환하여 Dataset 객체를 반환하는 함수
    train_data = convert_to_dataset(train_data, kor, eng)

    print(f'Build vocabulary using torchtext . . .')

    ## 데이터셋을 이용해 단어 사전을 만듦
    kor.build_vocab(train_data, max_size=config.kor_vocab)
    eng.build_vocab(train_data, max_size=config.eng_vocab)

    ## 각 단어 사전의 크기기
    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}')
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}')

    ## 가장 빈번하게 등장하는 단어어
    print(f'Most commonly used Korean words are as follows:')
    print(kor.vocab.freqs.most_common(20))

    print(f'Most commonly used English words are as follows:')
    print(eng.vocab.freqs.most_common(20))

    ## 만들어진 단어 사전을 pickle 파일로 저장장
    with open('pickles/kor.pickle', 'wb') as kor_file:
        pickle.dump(kor, kor_file)

    with open('pickles/eng.pickle', 'wb') as eng_file:
        pickle.dump(eng, eng_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--kor_vocab', type=int, default=55000)
    parser.add_argument('--eng_vocab', type=int, default=30000)

    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)
