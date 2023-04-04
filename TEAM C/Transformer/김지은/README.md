# Kor-Eng to Eng-Kor #
## parameter ##




    "model": "transformer",
    "save_model": "model.pt",

    "mode": "train",
    "optim": "Adam",

    "random_seed": 32,
    "clip": 1,

    "batch_size": 256,
    "num_epoch": 200,
    "warm_steps": 4000,

    "hidden_dim": 512,
    "feed_forward_dim": 2048,
    "n_layer": 6,
    "n_head": 8,
    "max_len": 64,
    "dropout": 0.2
    
    

## 코드 실행 순서 ##
python build_pickle2.py : pickle file 생성

python main2.py : Training

python main2.py --mode test : Test

python predict2.py --input "{sentence}" : Inferece





## Evaluation ##
포맷 이슈로 실험을 많이 해보지 못했습니다.


4.602


