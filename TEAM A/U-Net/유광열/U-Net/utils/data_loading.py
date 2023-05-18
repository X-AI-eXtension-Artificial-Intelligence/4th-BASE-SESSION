import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename): # 이미지 불러오기
    ext = splitext(filename)[1] # 이미지 이름 불러오기 
    if ext == '.npy': # npy 형태면 np로 불러오기
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']: # pt는 torch로 불러오기
        return Image.fromarray(torch.load(filename).numpy())
    else: # 나머지는 PIL
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0] # mask_file 이름 생성
    mask = np.asarray(load_image(mask_file)) # 이미지 불러오기
    if mask.ndim == 2: # 2차원일떄 unqiue
        return np.unique(mask)
    elif mask.ndim == 3: # 3차원 reshape 하고 unique
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset): # 데이터 셋 생성
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')] # 이미지 이름 리스트 생성
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there') # 없으면 오류 발생

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        # multiprocessing -> unique_mask_values함수 multi값 얻음
        with Pool() as p: 
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist())) # unique_mask 합침
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask): # 이미지 전처리
        w, h = pil_img.size # 이미지 크기 구하기
        newW, newH = int(scale * w), int(scale * h) # 이미지 scaling
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel' # 0보다 안크면 오류 발생
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # 이미지 resize mask값이면 주변값으로 보간 아니면 4x4를 사용해서 보간
        img = np.asarray(pil_img) # array로 변한

        if is_mask: #mask값이면
            mask = np.zeros((newH, newW), dtype=np.int64) #크기만큼 빈값 생성
            for i, v in enumerate(mask_values): #unqie mask 값만큼 채워주기
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else: #이미지 픽셀 스케일링
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0]) # 이미지 불러오기
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False) #이미지 전처리
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(), #이미지 mask 변환 돌려줌
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset): # Bascic DataSet 상속
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')