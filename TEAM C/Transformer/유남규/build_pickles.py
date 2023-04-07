import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils import convert_to_dataset

from torchtext import data as ttd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer


def build_tokenizer():
    """
    Train soynlp tokenizer which will be used to tokenize Korean input sentence
    """
    # soynlp를 사용하여 한국어 토크나이저를 학습하는 함수
    print(f'Now building soy-nlp tokenizer . . .')
    print(Path().cwd())
    # 현재 작업 중인 디렉토리 경로
    data_dir = Path().cwd() / 'data'
    # 학습 데이터가 저장된 경로
    train_file = os.path.join(data_dir, 'corpus.csv')

    # 학습 데이터 불러오기
    df = pd.read_csv(train_file, encoding='utf-8')

    # 학습 데이터에서 텍스트 데이터만 추출
    # 만약 텍스트 데이터가 아닌 데이터를 만나면 해당 데이터를 건너뛴다.
    kor_lines = [row.korean
                 for _, row in df.iterrows() if type(row.korean) == str]

    # soynlp의 WordExtractor 클래스를 사용하여 단어 점수를 계산한다.
    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(kor_lines)

    # 단어별 점수 계산 결과를 저장
    word_scores = word_extractor.extract()
    cohesion_scores = {word: score.cohesion_forward
                       for word, score in word_scores.items()}

    # pickle을 사용하여 cohesion_scores 변수를 저장
    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)


def build_vocab(config):
    """
    Build vocab used to convert input sentence into word indices using soynlp and spacy tokenizer
    Args:
        config: configuration containing various options
    """
    # tokenizer.pickle 파일에서 cohesion_scores 변수를 불러온다.
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)

    # 한국어 및 영어 vocab을 저장할 객체 생성
    # 한국어 vocab은 cohesion_scores를 이용하여 토크나이즈하며, pack pad sequence를 위해 각 문장의 길이를 저장한다.
    # 영어 vocab은 spacy 토크나이저를 이용하여 토크나이즈하며, 시작 및 끝 토큰을 추가하여 저장한다.
    kor = ttd.Field(tokenize=tokenizer.tokenize,
                    lower=True,
                    batch_first=True)

    eng = ttd.Field(tokenize='spacy',
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

    data_dir = Path().cwd() / 'data'  # 현재 작업 디렉토리에 data 폴더를 지정
    train_file = os.path.join(data_dir, 'train.csv')  # train.csv 파일 경로 지정
    train_data = pd.read_csv(train_file, encoding='utf-8')  # train.csv 파일 읽어오기
    train_data = convert_to_dataset(train_data, kor, eng)  # torchtext로 변환 가능한 dataset으로 변환

    print(f'Build vocabulary using torchtext . . .')  # vocab 생성 시작을 알리는 출력문

    kor.build_vocab(train_data, max_size=config.kor_vocab)  # 한글 vocab 생성
    eng.build_vocab(train_data, max_size=config.eng_vocab)  # 영어 vocab 생성

    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}')  # 한글 vocab의 token 개수 출력
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}')  # 영어 vocab의 token 개수 출력

    print(f'Most commonly used Korean words are as follows:')  # 가장 빈도가 높은 한글 단어 20개 출력
    print(kor.vocab.freqs.most_common(20))

    print(f'Most commonly used English words are as follows:')  # 가장 빈도가 높은 영어 단어 20개 출력
    print(eng.vocab.freqs.most_common(20))

    with open('pickles/kor.pickle', 'wb') as kor_file:  # 한글 vocab을 pickle 파일로 저장
        pickle.dump(kor, kor_file)

    with open('pickles/eng.pickle', 'wb') as eng_file:  # 영어 vocab을 pickle 파일로 저장
        pickle.dump(eng, eng_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--kor_vocab', type=int, default=55000)
    parser.add_argument('--eng_vocab', type=int, default=30000)

    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)
