import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
entire_loss=9999; val_score=9999
#dir_img = Path('/home/work/XAI/train/img/')
#dir_mask = Path('/home/work/XAI/train/masks/')
dir_img = Path('/home/work/XAI/Pytorch-UNet/data_aug/img/')
dir_mask = Path('/home/work/XAI/Pytorch-UNet/data_aug/masks/')
dir_checkpoint = Path('./checkpoints/')

# 학습하는 모델 함수 설정
def train_model(
        model,
        device,
        epochs: int = 100, #1
        batch_size: int = 5, #3
        learning_rate: float = 1e-5,
        val_percent: float = 0.2,
        save_checkpoint: bool = True,
        img_scale: float = 2, #0.5
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    global entire_loss, val_score
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale) # 데이터 셋 설정 위에꺼는  mask_suffix='_mask'으로 설정
        # 왜냐하면 마스킹 데이터의 이름은 원본 이미지의 이름 +"_mask"
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent) # validiation dataset 수량
    n_val = 0
    n_train = len(dataset) - n_val # train dataset 수량
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging) # 시각화 wandb 실행
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )
    # 정보 남기기 진행
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # 스케쥴러 설정
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # 메모리 절약을 위해 데이터 타입 변경시 기울기가 0이 되는 경우가 생김 따라서 이를 방지하기 위한 GradScaler
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 로스 설정
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        print(epoch)
        model.train() # 모델 학습 진행
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            #print("오류발생")
            for batch in train_loader:
                #print("durlsrk")
                images, true_masks = batch['image'], batch['mask']

                # 이미지의 shape[1] rgb 혹은 흑백 값이 모델의 인풋 채널과 같아야 한다는 뜻 같음.
                assert images.shape[1] == model.n_channels, f'Network has been defined with {model.n_channels} input channels, ' 
                    # f'but loaded images have {images.shape[1]} channels. Please check that '
                    # 'the images are loaded correctly.'
                
                # 데이터에 gpu할당
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # mps gpu가 아닐 시 cuda 로 진행 mps라면 cpu로 진행
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # 만약 흑백 구분이라면,
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    # 다양한 세그먼트 구분이라면
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                # 파라미터 업데이트 진행.
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # 로스 산정
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # 여기 아래는 Wandb 쪽 관련인건가..? 출력 기록 남기는 듯함.
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (4 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)
                        print('Validation Dice score: {}'.format(val_score))
                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        # 학습한 모델 저장
        #print('Validation Dice score: {}'.format(val_score))
        if entire_loss > val_score:
            entire_loss = val_score
            print(entire_loss, "가장 낮은 Loss")
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

    # parser의 인자 대입해주기
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1000, help='Number of epochs') #100
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size') #4
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default='', help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Downscaling factor of the images') # 0.5
    parser.add_argument('--validation', '-v', dest='val', type=float, default=50,
                        help='Percent of the data that is used as validation (0-100)') #20
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=15, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args() # args 설정

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}') # 실행 정보를 남기기 위한 logging

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear) # 모델 정의
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device) # pretrained model 있다면 사용
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device) # 모델에 gpu할당 
    # try: 학습 함수 실행
    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100, # 현재 10%으로 설정되어 있음
        amp=args.amp
    )
    # except torch.cuda.OutOfMemoryError:
    #     logging.error('Detected OutOfMemoryError! '
    #                   'Enabling checkpointing to reduce memory usage, but this slows down training. '
    #                   'Consider enabling AMP (--amp) for fast and memory efficient training')
    #     torch.cuda.empty_cache()
    #     model.use_checkpointing()
    #     train_model(
    #         model=model,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device,
    #         img_scale=args.scale,
    #         val_percent=args.val / 100,
    #         amp=args.amp
    #     )
