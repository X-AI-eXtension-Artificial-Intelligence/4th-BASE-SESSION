import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 이미지 파일들이 저장된 디렉토리 경로 
directory = "/home/work/XAI/test/img_rs" # 해당 부분을 변경하세요

# 디렉토리 내에 있는 모든 이미지 파일 경로 리스트
img_files = [os.path.join(directory, f) for f in os.listdir(directory)]

# 이미지 파일들을 3차원 numpy array로 불러오기
images = []
for f in tqdm(img_files):
    img = np.load(f)
    images.append(img)

# 3차원 numpy array를 데이터프레임으로 변환
df_list = []
for i, img in enumerate(images):
    height, width, channels = img.shape
    img_df = pd.DataFrame(img.reshape(height*width, channels))
    img_df['Image'] = i+1  # 이미지 번호
    df_list.append(img_df)

df = pd.concat(df_list, axis=0)
df.columns = ['Red', 'Green', 'Blue', 'Image']
df.reset_index(inplace=True)
df.drop(['Image','index'],axis=1,inplace=True)
df.reset_index(inplace=True)
df.columns = ['Image','Red', 'Green', 'Blue']
now = now = datetime.now()
current_time = now.strftime("%H_%M_%S")
df.to_csv(f'/home/work/XAI/Pytorch-UNet/subeen_submissions/submission_label_{current_time}.csv', index=False)
