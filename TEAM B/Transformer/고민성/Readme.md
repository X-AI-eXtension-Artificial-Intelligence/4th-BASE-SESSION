### 100epoch 학습 결과 로스

loss : 5.125

### 변경 사항

transformer 논문에 나오는 가장 좋은 성능을 나타내는 파라미터로 설정해서 학습을 진행해봤습니다.


batch_size : 200


n_heads : 16


hidden_dim: 1024,


feed_forward_dim: 4096,


dropout: 0.3

### INFERENCE

input : Our team is the best

output : "그 사람 은 <unk>"
