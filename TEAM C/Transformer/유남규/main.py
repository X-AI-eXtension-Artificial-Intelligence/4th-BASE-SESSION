import csv
import argparse
from trainer import Trainer
from utils import load_dataset, make_iter, Params


def main(config):
    # 설정 파일 파라미터 가져오기
    params = Params('config/params.json')

    if config.mode == 'train':
        # 데이터셋 로드
        train_data, valid_data = load_dataset(config.mode)
        # 훈련용 데이터와 검증용 데이터의 이터레이터 생성
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data)
        # 트레이너 생성 및 훈련 수행
        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()

    else:
        # 테스트 데이터셋 로드
        test_data = load_dataset(config.mode)
        # 테스트 데이터의 이터레이터 생성
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)
        # 트레이너 생성 및 추론 수행
        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.inference()


if __name__ == '__main__':
    # 프로그램 실행 시 인자값 파싱
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    # main 함수 호출
    main(args)
