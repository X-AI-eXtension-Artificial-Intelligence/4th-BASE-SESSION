# https://hipolarbear.tistory.com/19

import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
import PIL.ImageOps   
from natsort import natsorted # 문자열이 들어가더라도 숫자순서대로 정렬

# 만들 이미지 개수 입력 (train 이미지 데이터 갯수)
num_augmented_images = 15

file_path = '../train/img/'
mask_file_path = '../train/masks/'

file_names = natsorted(os.listdir(file_path))

total_origin_image_num = len(file_names)

augment_cnt = 1

# 반복문을 이용해서 
for _ in range(10):
    for i in range(num_augmented_images):
        
        file_name = file_names[i]
        print('작업하는 이미지 :', file_name)
        
        origin_image_path = '../train/original_image/' + file_name
        origin_mask_path = '../train/original_mask/' + file_name

        image = Image.open(origin_image_path)
        mask = Image.open(origin_mask_path)
        
        #오리지널 이미지 저장
        image.save(file_path + str(augment_cnt) + '.png')
        mask.save(mask_file_path + str(augment_cnt) + '.png')

        augment_cnt += 1

        #이미지 좌우 반전
        inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        inverted_image.save(file_path + str(augment_cnt) + '.png')

        inverted_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        inverted_mask.save(mask_file_path + str(augment_cnt) + '.png')

        augment_cnt += 1

        '''
        #이미지 기울이기
        rotated_image = image.rotate(random.randrange(-20, 20))
        rotated_image.save(file_path + str(augment_cnt) + '.png')

        rotated_mask = mask.rotate(random.randrange(-20, 20))
        rotated_mask.save(mask_file_path + str(augment_cnt) + '.png')

        augment_cnt += 1
        '''
        
        #노이즈 추가하기 (노이즈는 mask 그대로)
        img = cv2.imread(origin_image_path)

        row,col,ch= img.shape   
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)

        noisy_array = img + gauss
        noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
        
        noisy_image.save(file_path + str(augment_cnt) + '.png')
        mask.save(mask_file_path + str(augment_cnt) + '.png')
            
        augment_cnt += 1