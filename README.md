# ACCV 2022 Fine-grained Image Analysis Challenge

## 3rd Place Solution


### [Competition🔗](https://www.cvmart.net/race/10412/base)



### HARDWARE & SOFTWARE

Ubuntu 18.04.3 LTS

CPU: AMD EPYC 7543 32-Core Processor

GPU: 8 * NVIDIA A5000, Memory: 24G

Python: 3.8

Pytorch: 1.9.0+cu111

### Environment

Requirements
```bash
git clone https://github.com/XL-H/ACCV2022.git
cd ACCV2022
pip install -r requirements.txt
```
Apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


### Data Preparation

1. Run **Data_preprocessing.ipynb**
    1. Remove broken images
    2. Make csv file
    3. Resampling
    4. StratifiedKfold

### Model Preparation

1. Pre-trained models from ImageNet1K/ImageNet21K:
    - [beitv2-224](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth)
    - [beit-384](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_384_pt22k_ft22kto1k.pth)
    - [beit-512](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth)
    - [deit3-384](https://dl.fbaipublicfiles.com/deit/deit_3_large_384_21k.pth)
    - [swin-384](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth)
    - [swinv2-384](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth)


### Training & Inference

1. Configurations for training can be found in ACCV/config_timm.py

2. Training:
```bash
!CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 \
/root/ACCV2022/train.py \
--csv-dir autodl-tmp/ACCV_384_balance_fold.csv \
--config-name 'timm' \
--image-size 384 \
--batch-size 7 \
--num-workers 10 \
--init-lr 6e-5 \
--n-epochs 10 \
--cpkt_epoch 1 \
--n_batch_log 300 \
--warm_up_epochs 1 \
--fold 1
```

3. Training and Inference **Tools-Train-Inference.ipynb** :


### Contact

Feel free to contact, email: 3579628328@qq.com
