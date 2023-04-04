import csv
import argparse
from trainer import Trainer
from utils import load_dataset, make_iter, Params


def main(config):
    params = Params('config/params.json') # utils.py의 Params class로 이동

    if config.mode == 'train':                                                           # config에서 mode가 train일 경우
        train_data, valid_data = load_dataset(config.mode)                               # utils.py load_dataset과 make_iter
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,               # batch단위로 data를 load
                                           train_data=train_data, valid_data=valid_data)

        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter) #trainer.py의 Trainer class
        trainer.train()                                                                      #trainer.py의 Trainer.train()

    else:
        test_data = load_dataset(config.mode)                                            # data loader
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)       # utils.py의 make_iter

        trainer = Trainer(params, config.mode, test_iter=test_iter)                      # Trainer
        trainer.inference()                                                              # trainer.py의 Trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')#parser로 인자 지정
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    main(args)
