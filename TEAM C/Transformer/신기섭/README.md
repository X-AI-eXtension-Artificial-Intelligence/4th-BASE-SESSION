## Transformer ENG-KOR

### parameters
    "model": "transformer",
    "save_model": "model.pt",

    "mode": "train",
    "optim": "Adam",

    "random_seed": 32,
    "clip": 1,

    "batch_size": 256,
    "num_epoch": 350,
    "warm_steps": 5000,

    "hidden_dim": 512,
    "feed_forward_dim": 2048,
    "n_layer": 6,
    "n_head": 8,
    "max_len": 64,
    "dropout": 0.2

### 변경사항

- tokenizer 변경 -> 기존 soynlp에서 konlpy에 있는 okt로 변경
- dropout 비율을 0.1에서 0.2로 변경


### score
- train = 1.8
- validation = 4.0
- test = 4.116

입력 - I want to got home. <br/>
출력 - 우리 는 일찍 출발 하기 위해 집 으로 갔어

