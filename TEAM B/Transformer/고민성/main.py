import csv
import argparse
from trainer import Trainer
from utils import load_dataset, make_iter, Params


def main(config):# config 파일을 받아옴
    params = Params('config/params.json')

    if config.mode == 'train':# train 일 경우
        train_data, valid_data = load_dataset(config.mode)# load_dataset 진행, csv파일 불러오기
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data)
        # 모델 정의, loss정의, optimizer정의
        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()

    else:
        test_data = load_dataset(config.mode)
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')# argparser 생성
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test']) # parser 인자 지정
    args = parser.parse_args()
    main(args)