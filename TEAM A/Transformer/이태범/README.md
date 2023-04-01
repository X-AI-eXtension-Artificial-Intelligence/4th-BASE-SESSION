## 참고 코드  
https://github.com/Huffon/pytorch-transformer-kor-eng

{code}.py : 기존 Kor-Eng 변환 코드  
{code}2.py : 수정한 Eng-Kor 변환 코드

## Eng-Kor 변경 사항

epoch : 200  
random_seed : 2020  
batch_size : 164  
optim : AdamW  
dropout: 0.2  
warm_step : 3500

## Evaluation  
Test Loss : 4.070

## Inference Example  
Input : "I love you"  

output : "당신 은 진심으로 사랑해요"
