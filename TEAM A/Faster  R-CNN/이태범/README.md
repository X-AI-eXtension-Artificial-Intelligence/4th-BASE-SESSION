## Dacon 합성데이터 기반 객체 탐지 AI 경진대회
### YOLOv7 

**테스트 Inference 결과 저장**  
python3 detect.py --weights yolov7-e6e.pt --source ./train/syn_00000.png 

**테스트 성능 결과 보여줌**  
python3 test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val

**학습 코드 예시**  
python3 train.py --workers 0 --device 0 --batch-size 2 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml

**Inference 코드 예시** 
python3 detect.py --weights ./runs/train/yolov7-custom4/weights/best.pt --conf 0.25 --img-size 1280 --save-txt --save-conf --source ./custom/images/test/
