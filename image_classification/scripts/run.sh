# o2c
CUDA_VISIBLE_DEVICES=0 python 0_mixup_bsp.py your_datapath -d medical_images -s O -t C -a resnet50 \
 --bottleneck-dim 256 --epochs 30 -i 500 --log logs/ \
 --lr 0.01 --classifier-mode 4  \
 --current-class 1 --test-batch-size 1024 \
 --pretrain --pretrain-model-path your_pretrain_model_path \
 --decouple \
 --source-mixup-mode 1 --source-mixup-alpha 4 \
 --hi-threshold 0.9 --lo-threshold 0.1  --lam-alpha 0.5 --trade-off-st 1  \
 --trade-off-sd 0.0002 \
 --sd --f-hi-threshold 0.7 --f-lo-threshold 0.3
# o2n
CUDA_VISIBLE_DEVICES=0 python 0_mixup_bsp.py your_datapath -d medical_images -s O -t N -a resnet50 \
 --bottleneck-dim 256 --epochs 30 -i 500 --log logs/ \
 --lr 0.01 --classifier-mode 4  \
 --current-class 1 --test-batch-size 1024 \
 --pretrain --pretrain-model-path your_pretrain_model_path \
 --decouple \
 --source-mixup-mode 1 --source-mixup-alpha 4 \
 --hi-threshold 0.9 --lo-threshold 0.1  --lam-alpha 0.5 --trade-off-st 1  \
 --trade-off-sd 0.0002 \
 --sd --f-hi-threshold 0.7 --f-lo-threshold 0.3
# o2m
CUDA_VISIBLE_DEVICES=0 python 0_mixup_bsp.py your_datapath -d medical_images -s O -t M -a resnet50 \
 --bottleneck-dim 256 --epochs 30 -i 500 --log logs/ \
 --lr 0.01 --classifier-mode 4  \
 --current-class 1 --test-batch-size 1024 \
 --pretrain --pretrain-model-path your_pretrain_model_path \
 --decouple \
 --source-mixup-mode 1 --source-mixup-alpha 4 \
 --hi-threshold 0.9 --lo-threshold 0.1  --lam-alpha 0.5 --trade-off-st 1  \
 --trade-off-sd 0.0002 \
 --sd --f-hi-threshold 0.7 --f-lo-threshold 0.3
