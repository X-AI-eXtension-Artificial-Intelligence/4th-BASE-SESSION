import csv
import argparse
from trainer import Trainer
from utils import load_dataset, make_iter, Params


def main(config):
    params = Params('config/params.json') # 파라미터 파일 로드

    if config.mode == 'train':  
        train_data, valid_data = load_dataset(config.mode) # mode에 따른 load dataset -> util로 이동
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data) # -> util로 이동

        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter) 
        trainer.train()

    else:
        test_data = load_dataset(config.mode)
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation') # parser 생성
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test']) # argument 추가
    args = parser.parse_args() # parser 변수 지정
    main(args) # main 함수 실행 
