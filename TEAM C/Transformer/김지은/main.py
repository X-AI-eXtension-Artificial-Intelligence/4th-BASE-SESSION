import csv
import argparse
from trainer import Trainer
from utils import load_dataset, make_iter, Params


def main(config):
    params = Params('config/params.json')

    if config.mode == 'train': #mode가 train이면
        train_data, valid_data = load_dataset(config.mode)
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data)
        #train과 valid 데이터를 불러와서 학습해라

        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()

    else: #inference이면 test 데이터 불러와서 학습해라
        test_data = load_dataset(config.mode)
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.inference()


if __name__ == '__main__': #파이썬 스크립트?
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    main(args)
