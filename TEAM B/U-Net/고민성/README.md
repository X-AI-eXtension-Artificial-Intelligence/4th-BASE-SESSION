# U-Net: Semantic segmentation with PyTorch
https://www.kaggle.com/competitions/4th-xai-base-session-segmentation-competiton-1st/leaderboard


- 변경점


데이터를 확인해보니, 이미지의 크기가 제각각이었고 이에따라 리사이즈를 진행합니다. 또한, 이미지는 세로의 길이가 더 컸고 이미지의 크기를 최대한 보전할 수 있는 가로 360, 세로 640으로 resize하여 진행하였습니다. 

epoch을 다양하게 하며 실험을 진행하였습니다.
epoch을 다양하게 하며 실험한 결과, test dataset의 크기가 작다보니 epoch을 크게 할 수록 과적합이 발생하여 오히려 점수가 떨어지는 현상이 나타났습니다.


- test 성능 


0.58096
