import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils2 import convert_to_dataset

from torchtext import data as ttd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

# Kor -> Eng 용 Pickle 만들기

# 한국어 tokenizer 
def build_tokenizer():
    """
    Train soynlp tokenizer which will be used to tokenize Korean input sentence
    """
    print(f'Now building soy-nlp tokenizer . . .')

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'corpus.csv')

    df = pd.read_csv(train_file, encoding='utf-8')

    # if encounters non-text row, we should skip it
    kor_lines = [row.korean for _, row in df.iterrows() if type(row.korean) == str]

    # https://github.com/lovit/soynlp#word-extraction
    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(kor_lines)

    word_scores = word_extractor.extract()
    # cohesion score : https://lovit.github.io/nlp/2018/04/09/cohesion_ltokenizer/
    cohesion_scores = {word: score.cohesion_forward
                       for word, score in word_scores.items()}
    
    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)


# target인 korean에 sos, eos 추가하고 위의 tokenizer 방법 적용 
def build_vocab(config):
    """
    Build vocab used to convert input sentence into word indices using soynlp and spacy tokenizer
    Args:
        config: configuration containing various options
    """
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores) # L part의 단어 점수 입력

    # include lengths of the source sentences to use pack pad sequence
    eng = ttd.Field(tokenize='spacy',
                    lower=True,
                    batch_first=True)

    # Field를 통해 앞으로 어떤 전처리를 할 것인지를 정의
    kor = ttd.Field(tokenize= tokenizer.tokenize, 
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)
    
    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_file, encoding='utf-8')
    train_data = convert_to_dataset(train_data, kor, eng)

    print(f'Build vocabulary using torchtext . . .')

    # https://wikidocs.net/65348
    # 단어 집합을 생성, max_size : 단어 집합의 최대 크기
    # 어휘 빌드 
    kor.build_vocab(train_data, max_size=config.kor_vocab)
    eng.build_vocab(train_data, max_size=config.eng_vocab)

    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}')
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}')

    print(f'Most commonly used Korean words are as follows:')
    print(kor.vocab.freqs.most_common(20))

    print(f'Most commonly used English words are as follows:')
    print(eng.vocab.freqs.most_common(20))

    with open('pickles/kor.pickle', 'wb') as kor_file:
        pickle.dump(kor, kor_file)

    with open('pickles/eng.pickle', 'wb') as eng_file:
        pickle.dump(eng, eng_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    # 받아들일 인수 추가
    # max_size
    parser.add_argument('--kor_vocab', type=int, default=55000)
    parser.add_argument('--eng_vocab', type=int, default=30000)

    # 인수를 분석
    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)
