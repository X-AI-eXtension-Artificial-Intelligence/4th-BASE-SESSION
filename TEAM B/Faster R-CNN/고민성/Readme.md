### Faster R-CNN 과제 코드 진행.

최종 score 0.55 DACON 100위 정도

### 변경 사항

기본 base code에서 하이퍼 파라미터 튜닝을 위주로 진행했습니다.
mmdetection이나 다른 sota code를 시도했지만, 제대로 적용되지는 않았던 것 같습니다.

하이퍼 파라미터 튜닝에서는 data augmentation과 이미지 사이즈를 최대로 키우면서 진행해봤습니다.
data augmentation에서는 이미지를 살펴본 결과 train과 test사진의 차이는 햇빛의 밝기, 노이즈값 이런 부분의 차이가 있어 이것들을 중점으로 변형하여 진행했습니다.
