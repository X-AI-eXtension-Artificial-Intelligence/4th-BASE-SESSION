python train.py --amp

python predict.py -m "/home/work/XAI/Pytorch-UNet/checkpoints/checkpoint_epoch1000.pth" -i "/home/work/XAI/test/img/16.png" -o "/home/work/XAI/test/img_rs/16"
python predict.py -m "/home/work/XAI/Pytorch-UNet/checkpoints/checkpoint_epoch1000.pth" -i "/home/work/XAI/test/img/17.png" -o "/home/work/XAI/test/img_rs/17"
python predict.py -m "/home/work/XAI/Pytorch-UNet/checkpoints/checkpoint_epoch1000.pth" -i "/home/work/XAI/test/img/18.png" -o "/home/work/XAI/test/img_rs/18"
python predict.py -m "/home/work/XAI/Pytorch-UNet/checkpoints/checkpoint_epoch1000.pth" -i "/home/work/XAI/test/img/19.png" -o "/home/work/XAI/test/img_rs/19"
python predict.py -m "/home/work/XAI/Pytorch-UNet/checkpoints/checkpoint_epoch1000.pth" -i "/home/work/XAI/test/img/20.png" -o "/home/work/XAI/test/img_rs/20"

python Generation_sub.py