import csv
import argparse
from trainer import Trainer
from utils import load_dataset, make_iter, Params


def main(config):
    #학습에 쓰이는 파라미터 파일 load
    params = Params('config/params.json')

    if config.mode == 'train': # train 모드일 경우
        #data load
        train_data, valid_data = load_dataset(config.mode)
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                            train_data=train_data, valid_data=valid_data)

        # 모델 학습을 위한 Trainer 클래스 생성
        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        # 학습 진행
        trainer.train()

    else: # test 모드일 경우
        # data load
        test_data = load_dataset(config.mode)
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        # 모델 test를 위한 Trainer 클래스 생성
        trainer = Trainer(params, config.mode, test_iter=test_iter)
        # 추론 진행
        trainer.inference()


if __name__ == '__main__':
    # 명령행을 파이썬 데이터형으로 파싱하기 위한 ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    # 인자 정의
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # 인자 파싱
    args = parser.parse_args()
    # main함수 실행
    main(args)
