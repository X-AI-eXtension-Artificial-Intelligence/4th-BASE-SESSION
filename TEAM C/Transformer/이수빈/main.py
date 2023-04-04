import csv
import argparse
from trainer import Trainer
from utils import load_dataset, make_iter, Params


def main(config):
    params = Params('config/params.json') # params.json에 저장된 파라미터 불러오기

    if config.mode == 'train':
        train_data, valid_data = load_dataset(config.mode) # train.csv, valid.csv 불러오기
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data)
        # make_iter : pandas dataset -> torchtext dataset

        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()
        # Trainer : Transformer로 학습
        # Train, Test mode 설정 후, Optimzer 설정

    else:
        test_data = load_dataset(config.mode) # test.csv 불러오기
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.inference()
        # 학습 후 저장된 모델 사용하여 inference 출력


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation') # 인자 도움말 전에 표시할 텍스트
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test']) # '--mode' 인수 추가
    args = parser.parse_args()
    main(args)
