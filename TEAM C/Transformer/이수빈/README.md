# X:AI_Transformer_Eng-Kor

*컴퓨터 사양 문제로 성능보다는 오류없이 inference 진행해보는 것에 초점 맞춰 진행했습니다.*

### 사용 Params.
- batch_size": 512,
- num_epoch": 10,
- warm_steps": 400,
- hidden_dim": 128,
- feed_forward_dim": 512,
- n_layer": 3,
- n_head": 4,
- max_len": 64,
- dropout": 0.2
*추가적으로 build_pickles.py에서 단어 사이즈 1/10로 줄여 진행함.*

### test loss
약 4.3