#pip install typing-extensions --upgrade
#pip install albumentations
import warnings
warnings.filterwarnings(action='ignore')

import random
import pandas as pd
import numpy as np
import os
import glob
import cv2
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
#from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'NUM_CLASS':34,
    'IMG_SIZE':(1280, 720),
    'EPOCHS':100,
    'LR':3e-4,
    'BATCH_SIZE':4,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


def draw_boxes_on_image(image_path, annotation_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # txt 파일에서 Class ID와 Bounding Box 정보 읽기
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        values = list(map(float, line.strip().split(' ')))
        class_id = int(values[0])
        x_min, y_min = int(round(values[1])), int(round(values[2]))
        x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

        # 이미지에 바운딩 박스 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, str(class_id), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 이미지와 바운딩 박스 출력
    #plt.figure(figsize=(25, 25))
    #plt.imshow(image)
    #plt.show()
    
# # 파일 경로 설정
# image_file = './train/syn_00001.png'
# annotation_file = './train/syn_00001.txt'

# # 함수 실행
# draw_boxes_on_image(image_file, annotation_file)


def collate_fn(batch):
    images, targets_boxes, targets_labels = tuple(zip(*batch))
    images = torch.stack(images, 0)
    targets = []
    
    for i in range(len(targets_boxes)):
        target = {
            "boxes": targets_boxes[i],
            "labels": targets_labels[i]
        }
        targets.append(target)

    return images, targets


class CustomDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = root
        self.train = train
        self.transforms = transforms
        
        if train:
            self.imgs = sorted(glob.glob('/home/work/X-AI/car/training/image_2'+'/*.png'))
            self.boxes = sorted(glob.glob('/home/work/X-AI/car/training/label_2'+'/*.txt'))
        else:
            self.imgs = sorted(glob.glob(root+'/*.png'))
    def parse_boxes(self, box_path):
        with open(box_path, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []

        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        if self.train:
            box_path = self.boxes[idx]
            boxes, labels = self.parse_boxes(box_path)
            labels += 1 # Background = 0

            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]
                
            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

        else:
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed["image"]
            file_name = img_path.split('/')[-1]
            return file_name, img, width, height

    def __len__(self):
        return len(self.imgs)


def get_train_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'][0], CFG['IMG_SIZE'][1]),
        A.GaussNoise(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.4),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_test_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'][0], CFG['IMG_SIZE'][1]),
        ToTensorV2(),
    ])


def build_model(num_classes=CFG['NUM_CLASS']+1):
    #model  = fasterrcnn_resnet50_fpn_v2(pretrained=True)fasterrcnn_resnet50_fpn
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    print("model 받아왔습니당")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

#FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
def train(model, train_loader, optimizer, scheduler, device):
    print(1)
    model.to(device)
    print(2)
    best_loss = 9999999
    best_model = None
    print(3)
    for epoch in range(1, CFG['EPOCHS']+1):
        print(epoch, " : 번째 학습 진행 중 ")
        f = open("/home/work/X-AI/result.txt", 'a')
        model.train()
        train_loss = []
        for images, targets in tqdm(iter(train_loader), mininterval=10):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            train_loss.append(losses.item())

        if scheduler is not None:
            scheduler.step()
        
        tr_loss = np.mean(train_loss)
        f.write(f'Epoch [{epoch}] Train loss : [{tr_loss:.5f}]\n')
        print(f'Epoch [{epoch}] Train loss : [{tr_loss:.5f}]\n')
        
        if best_loss > tr_loss:
            best_loss = tr_loss
            best_model = model
            torch.save(model, "best_model.pth")
        f.close()
    return best_model

def box_denormalize(x1, y1, x2, y2, width, height):
    x1 = (x1 / CFG['IMG_SIZE'][0]) * width
    y1 = (y1 / CFG['IMG_SIZE'][1]) * height
    x2 = (x2 / CFG['IMG_SIZE'][0]) * width
    y2 = (y2 / CFG['IMG_SIZE'][1]) * height
    return x1.item(), y1.item(), x2.item(), y2.item()

def inference(model, test_loader, device):
    model.eval()
    model.to(device)
    
    results = pd.read_csv('/home/work/X-AI/sample_submission.csv')

    for img_files, images, img_width, img_height in tqdm(iter(test_loader)):
        images = [img.to(device) for img in images]
        print(img_files)
        with torch.no_grad():
            outputs = model(images)

        for idx, output in enumerate(outputs):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = box_denormalize(x1, y1, x2, y2, img_width[idx], img_height[idx])
                results = results.append({
                    "file_name": img_files[idx],
                    "class_id": label-1,
                    "confidence": score,
                    "point1_x": x1, "point1_y": y1,
                    "point2_x": x2, "point2_y": y1,
                    "point3_x": x2, "point3_y": y2,
                    "point4_x": x1, "point4_y": y2
                }, ignore_index=True)


    # 결과를 CSV 파일로 저장
    results.to_csv('baseline_submit.csv', index=False)
    print('Done.')

if __name__=="__main__":

    train_dataset = CustomDataset('/home/work/X-AI/train', train=True, transforms=get_train_transforms())
    test_dataset = CustomDataset('/home/work/X-AI/test', train=False, transforms=get_test_transforms())

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
    print("dataset 준비 완료.")
    model = build_model()
    print("model 준비 완료")
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    print(" 학습 시작 ")
    infer_model = train(model, train_loader, optimizer, scheduler, device)
    save_path = "/home/work/X-AI/best_model.pth"
    print(" inference 시작 ")
    inference(torch.load(save_path), test_loader, device)