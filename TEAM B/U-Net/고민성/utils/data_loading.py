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


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask) # 이러면 고유값들을 반환한다. 0,1 밖에 없지 않나,.?
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir) #경로 설정
        self.mask_dir = Path(mask_dir) # 경로 설정
        assert 0 < scale <= 1, 'Scale must be between 0 and 1' # 이미지의 사이즈 조정
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        # 경로 내 이미지가 있으며 '.'으로 시작하는 파일이 아닐 경우 파일의 0번째를 인덱스 리스트로 가져옴
        # splittext의 경우 파일명과 확장자를 구분시켜주는 함수

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        # 파일 탐지가 되지 않을 경우 경고 발생

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        with Pool() as p: #병렬 처리인듯함.  병렬 처리하여서 마스크 처리된 이미지를 불러오는 것.
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids) # partial 은 함수의 인자를 미리 넣어주는 것. unique_mask_values 함수에 대해서 진행하는 것 
            ))
        # 마스킹된 이미지 가져오기
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist())) # 마스킹 값에 유니크값.
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, train_right=1):
        w, h = pil_img.size # 이미지 사이즈를 받아온 후
        #print(w, h)
        newW, newH = int(scale * w), int(scale * h) # 이미지 크기 조정을 합니다.
        newW, newH = 360, 640
        #print(newH, newW, "확인바람.")
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img) # 불러온 이미지를 numpy 배열로 변환
        # 새로운 이미지 조정
        # 아 예를들어 마스킹 값이 1,3 이라면 0,1로 값을 조정해준다. 
        if is_mask: # 마스크가 존재한다면
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2: # 이미지가 2차원일 경우
                    mask[img == v] = i # 해당 부분을 마스킹 값으로 채워넣음 나머지는 0
                else:
                    mask[(img == v).all(-1)] = i # 이미지가 3차원일 경우 해당 부분의 모든 채널에 값을 채워넣는다.

            return mask

        else: # 마스킹이 없다면
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0 # 연산을 위해 255로 나눔.

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False) # 이미지는 그대로 가져와!
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True) # 마스킹 이미지는 특정 부분 마스킹 처리 필요

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')
