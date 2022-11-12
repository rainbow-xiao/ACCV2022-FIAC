import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.IMG_SIZE = 384
_C.SEED = 999
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'TIMM'
_C.MODEL.num_classes = 5000
_C.MODEL.finetune = None
_C.MODEL.output_dir = '/root/autodl-tmp/beitv2_224'
# _C.MODEL.output_dir = '/root/autodl-tmp/beit_224'
# _C.MODEL.output_dir = '/root/autodl-tmp/beit_384'
# _C.MODEL.output_dir = '/root/autodl-tmp/beit_512'
# _C.MODEL.output_dir = '/root/autodl-tmp/conv_224'
# _C.MODEL.output_dir = '/root/autodl-tmp/conv_384'
# _C.MODEL.output_dir = '/root/autodl-tmp/deit_224'
# _C.MODEL.output_dir = '/root/autodl-tmp/deit_384'
# _C.MODEL.output_dir = '/root/autodl-tmp/effnet_475'
# _C.MODEL.output_dir = '/root/autodl-tmp/effnet_800'
# _C.MODEL.output_dir = '/root/autodl-tmp/swin_224'
# _C.MODEL.output_dir = '/root/autodl-tmp/swin_384'
# _C.MODEL.output_dir = '/root/autodl-tmp/swinv2_224'
# _C.MODEL.output_dir = '/root/autodl-tmp/swinv2_384'
#head
_C.MODEL.head = CN()
_C.MODEL.head.name = 'Sub_Arc_face'
_C.MODEL.head.K = 2

#pool
_C.MODEL.pool = CN()
_C.MODEL.pool.GeM_p_trainable = False

# backbone
_C.MODEL.backbone = CN()
_C.MODEL.backbone.from_timm = True
# _C.MODEL.backbone.name = 'beitv2_large_patch16_224_in22k'
# _C.MODEL.backbone.out_dim = 1024
# _C.MODEL.backbone.pretrained = 'pretrained_models/beitv2_large_patch16_224_pt1k_ft21k.pth'

# _C.MODEL.backbone.name = 'beit_large_patch16_224_in22k'
# _C.MODEL.backbone.out_dim = 1024
# _C.MODEL.backbone.pretrained = 'pretrained_models/beit_large_patch16_224_pt22k_ft22k.pth'

_C.MODEL.backbone.name = 'beit_large_patch16_384'
_C.MODEL.backbone.out_dim = 1024
_C.MODEL.backbone.pretrained = '../pretrained_models/beit_large_patch16_384_pt22k_ft22kto1k.pth'

# _C.MODEL.backbone.name = 'beit_large_patch16_512'
# _C.MODEL.backbone.out_dim = 1024
# _C.MODEL.backbone.pretrained = 'pretrained_models/beit_large_patch16_512_pt22k_ft22kto1k.pth'

# _C.MODEL.backbone.name = 'convnext_large_in22k'
# _C.MODEL.backbone.out_dim = 1536
# _C.MODEL.backbone.pretrained = 'pretrained_models/convnext_large_22k_224.pth'

# _C.MODEL.backbone.name = 'convnext_large_384_in22ft1k'
# _C.MODEL.backbone.out_dim = 1536
# _C.MODEL.backbone.pretrained = 'pretrained_models/convnext_large_22k_1k_384.pth'

# _C.MODEL.backbone.name = 'deit3_large_patch16_224_in21ft1k'
# _C.MODEL.backbone.out_dim = 1024
# _C.MODEL.backbone.pretrained = 'pretrained_models/deit_3_large_224_21k.pth'

# _C.MODEL.backbone.name = 'deit3_large_patch16_384_in21ft1k'
# _C.MODEL.backbone.out_dim = 1024
# _C.MODEL.backbone.pretrained = 'pretrained_models/deit_3_large_384_21k.pth'

# _C.MODEL.backbone.name = 'tf_efficientnet_l2_ns_475'
# _C.MODEL.backbone.out_dim = 5504
# _C.MODEL.backbone.pretrained = 'pretrained_models/tf_efficientnet_l2_ns_475-bebbd00a.pth'

# _C.MODEL.backbone.name = 'tf_efficientnet_l2_ns'
# _C.MODEL.backbone.out_dim = 5504
# _C.MODEL.backbone.pretrained = 'pretrained_models/tf_efficientnet_l2_ns-df73bb44.pth'

# _C.MODEL.backbone.name = 'swin_large_patch4_window7_224_in22k'
# _C.MODEL.backbone.out_dim = 1536
# _C.MODEL.backbone.pretrained = 'pretrained_models/swin_large_patch4_window7_224_22k.pth'

# _C.MODEL.backbone.name = 'swin_large_patch4_window12_384_in22k'
# _C.MODEL.backbone.out_dim = 1536
# _C.MODEL.backbone.pretrained = 'pretrained_models/swin_large_patch4_window12_384_22k.pth'

# _C.MODEL.backbone.name = 'swinv2_large_window12to16_192to256_22kft1k'
# _C.MODEL.backbone.out_dim = 1536
# _C.MODEL.backbone.pretrained = 'pretrained_models/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth'

# _C.MODEL.backbone.name = 'swinv2_large_window12to24_192to384_22kft1k'
# _C.MODEL.backbone.out_dim = 1536
# _C.MODEL.backbone.pretrained = 'pretrained_models/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth'

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'AdamW'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 1e-4
_C.Optimizer.backbone_lr_scale_factor = 4e-2

#Loss
_C.Loss = CN()
_C.Loss.name = 'ArcFaceLossAdaptiveMargin'
_C.Loss.crit = 'smoth_ce'
_C.Loss.ls = 0.8
_C.Loss.s = 30.0
_C.Loss.m = 0.1
_C.Loss.stride_m = 0.1
_C.Loss.max_m = 0.8

def get_config():
    config = _C.clone()
    return config
