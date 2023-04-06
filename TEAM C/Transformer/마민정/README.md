## Kor-Eng to Eng-Kor

    epoch : 200  
    random_seed : 2020  
    batch_size : 164  
    optim : Adam  
    dropout: 0.2  
    warm_step : 3500  

<br/>

## Code Execution
```
# 한국어 tokenization 후 pickle file 생성
python build_pickle.py     
```
```
# Training 결과 확인
python main.py      
```
```
# Test Loss 확인
python main.py --mode test    
```
```
# Inference 결과 확인
python predict.py --input "{sentence}" 
```

<br/>

## Evaluation  
- Test Loss : 4.981  

<br/>

## Inference Example  
- Input : "I love you"  
- output : "마지막으로 는 그들의 삶을 <unk>"  

<br/>

![result](https://user-images.githubusercontent.com/75362328/229744136-523ca181-dff9-438e-989c-34a8d5496bb1.png)

<br/>

## Reference
- https://github.com/Huffon/pytorch-transformer-kor-eng


