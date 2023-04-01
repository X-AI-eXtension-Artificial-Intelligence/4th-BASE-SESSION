import csv
import argparse
from trainer2 import Trainer2
from utils2 import load_dataset, make_iter, Params
import wandb

# wandb 새로운 project 실행
wandb.init(project='transformer')

# wandb 적용시 오류
# ImportError: cannot import name 'TypeAlias' from 'typing_extensions'
# pip install typing-extensions --upgrade로 해결!

# 학습 시 오류 (차원이 어딘가 안맞아서! -> util2의 Param을 바꿔줘야됨)
# RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
# https://otugi.tistory.com/377

# Eng -> Kor
def main(config):
    params = Params('config/params.json')

    # wandb config 설정
    wandb.config = params

    if config.mode == 'train':
        train_data, valid_data = load_dataset(config.mode)
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data)

        trainer = Trainer2(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train()

    else:
        test_data = load_dataset(config.mode)
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        trainer = Trainer2(params, config.mode, test_iter=test_iter)
        trainer.inference()


if __name__ == '__main__':
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    # 입력받을 인자 값 설정
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # args에 위의 인자값 내용 저장장
    args = parser.parse_args()
    main(args)
