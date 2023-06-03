import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img, ## 분할할 이미지
                device,
                scale_factor=1, ## 이미지 크기 축소 비율
                out_threshold=0.5):
    net.eval() ## 평가 모드로 설정
    ## 이미지 전처리
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad(): ## 가중치 갱신 x
        output = net(img).cpu()
        ## 출력값을 원래 이미지 크기로 보간
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1) ## 클래스 인덱스 얻음
        else:
            mask = torch.sigmoid(output) > out_threshold ## 이진분류

    return mask[0].long().squeeze().numpy() ## mask를 numpy 배열로 반환


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

## 입력된 인자를 바탕으로, 출력 파일 이름 생성
def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))

## 모델이 예측한 마스크를 이미지 형식으로 변환
def mask_to_image(mask: np.ndarray, mask_values):
    ## 마스크와 같은 크기의 배열로
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    ## 가장 큰 값을 가지는 클래스 인덱스
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    
    ## 각 클래스의 값을 out 배열에 할당
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out) ## 변경된 배열을 이미지로 변환하여 반환


if __name__ == '__main__':
    args = get_args() ## 명령행 인자를 받아와 설정 초기화
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args) ## 출력 이미지 파일명을 리스트로 생성

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    ## 모델 불러오고 각 클래스 레이블 값 나타냄
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    ## 모든 이미지 파일에 대해 U-Net 모델 사용하여 마스크 예측
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values) ## 출력 이미지로 변환
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz: ## 결과 시각화
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
