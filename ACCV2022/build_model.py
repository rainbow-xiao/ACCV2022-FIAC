import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import apex
from utils import *
import timm

def build_vit(config, logger):
    model_type = config.MODEL.backbone.name
    if model_type=='clip_vit':
        model = VisionTransformer(
                  image_size=config.MODEL.backbone.VIT.image_size,
                  patch_size=config.MODEL.backbone.VIT.patch_size,
                  width=config.MODEL.backbone.VIT.width,
                  layers=config.MODEL.backbone.VIT.layers,
                  heads=config.MODEL.backbone.VIT.heads,
                  mlp_ratio=config.MODEL.backbone.VIT.mlp_ratio,
                  output_dim=config.MODEL.backbone.VIT.output_dim
              )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    if config.MODEL.backbone.pretrained != None:
        model.load_state_dict(torch.load(config.MODEL.backbone.pretrained, map_location='cpu'), strict=True)
        logger.info(f"=> Load pretrained vit_backbone '{config.MODEL.backbone.pretrained}' successfully")
    return model
    
def build_head(config):
    name = config.MODEL.head.name
    if name=='Arc_face':
        head = ArcMarginProduct(config.MODEL.backbone.out_dim, config.MODEL.num_classes)
    elif name=='Sub_Arc_face':
        head = ArcMarginProduct_subcenter(config.MODEL.backbone.out_dim, config.MODEL.num_classes, k=config.MODEL.head.K)
    else:
        raise NotImplementedError(f"Unkown head: {name}")
    return head

class Backbone(nn.Module):
    def __init__(self, config=None, logger=None):
        super(Backbone, self).__init__()
        self.name = config.MODEL.backbone.name
        self.from_timm = config.MODEL.backbone.from_timm
        self.img_size = config.IMG_SIZE
        if self.from_timm==True:
            self.net = timm.create_model(self.name,
                                         pretrained=False)
            if 'beitv2' in self.name:
                self.net.load_state_dict(torch.load(config.MODEL.backbone.pretrained, map_location='cpu')['module'], strict=True) #beitv2
            elif 'beit' in self.name:
                self.net.load_state_dict(torch.load(config.MODEL.backbone.pretrained, map_location='cpu')['model'], strict=True) #beit
            elif 'deit3' or 'Deit' in self.name:
                dicts = torch.load(config.MODEL.backbone.pretrained, map_location='cpu')['model']
                new_state_dict = {}
                for k,v in dicts.items():
                    if 'gamma_1' in k:
                        key = k.replace('gamma_1','ls1.gamma')
                    elif 'gamma_2' in k:
                        key = k.replace('gamma_2','ls2.gamma')
                    else:
                        key = k
                    new_state_dict[key]=v
                self.net.load_state_dict(new_state_dict, strict=True) #deit3
            elif 'efficient' in self.name:
                self.net.load_state_dict(torch.load(config.MODEL.backbone.pretrained, map_location='cpu'), strict=True)          #efficientnet
            elif 'swin' in self.name:
                self.net.load_state_dict(torch.load(config.MODEL.backbone.pretrained, map_location='cpu')['model'], strict=True) #swinv1/v2
#             logger.info(f"=> Load pretrained backbone '{config.MODEL.backbone.pretrained}' successfully")

            self.net.head = nn.Identity()
            
        elif 'vit' in self.name:
            self.net = build_vit(config, logger)
        else:
            raise NotImplementedError(f"Unkown model_name: {name}")

    def forward(self, x):
        x = self.net.forward_features(x)
#         print(x.shape)
        if self.name == 'Deit':
            x = x[:, 0, :]
        elif 'beit' in self.name or 'deit' in self.name:
            x = x[:, 1:]
            x = x.permute(0,2,1).contiguous()
            x = x.view(-1, 1024, self.img_size//16, self.img_size//16)
        elif 'convnext' in self.name:
            x = x.view(-1, 1536, self.img_size//32, self.img_size//32)
        elif 'efficient' in self.name:
#             x = x.view(-1, 5504, self.img_size//32, self.img_size//32)
            x = x.view(-1, 5504, self.img_size//32, self.img_size//32)
        elif 'swin' in self.name:
            x = x.permute(0,2,1).contiguous()
            x = x.view(-1, 1536, self.img_size//32, self.img_size//32)
        return x

class XL_TIMM_Net(nn.Module):
    def __init__(self, config, logger):
        super(XL_TIMM_Net, self).__init__()        
        self.backbone = Backbone(config, logger)
#         self.global_pool = GeM_Pooling(p_trainable=config.MODEL.pool.GeM_p_trainable)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = build_head(config)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x).flatten(1)
        logits = self.head(x)
        return logits

class XL_DEIT_Net(nn.Module):
    def __init__(self, config, logger):
        super(XL_DEIT_Net, self).__init__()        
        self.backbone = Backbone(config, logger)
        self.head = build_head(config)
    
    def forward(self, x):
        x = self.backbone(x)
        logits = self.head(x)
        return logits
    
class XL_CLIP_Net(nn.Module):
    def __init__(self, config, logger):
        super(XL_CLIP_Net, self).__init__()        
        self.backbone = Backbone(config, logger)
        self.Neck = Neck(config.MODEL.backbone.output_dim, config.MODEL.Embedding_dim, style=config.MODEL.neck.style)
        self.head = build_head(config)
        if config.MODEL.backbone.frozen:
            for param in self.backbone.parameters():
                param.requires_grad=False
        
    def forward_embedding(self, x):
        x = self.backbone(x)
        embedding = self.Neck(x)
        return embedding
    
    def forward(self, x):
        x = self.backbone(x)
        embedding = self.Neck(x)
        logits = self.head(embedding)
        return logits


class XL_DOLG_Net(nn.Module):
    def __init__(self, config, logger):
        super(XL_DOLG_Net, self).__init__()        
        self.backbone = Backbone(config, logger)
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, config.MODEL.hidden_dim, size=int(config.IMG_SIZE/8))
        self.global_pool = GeM_Pooling(p_trainable=config.MODEL.pool.GeM_p_trainable)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.neck_glob = Neck(1024, config.MODEL.hidden_dim, style=config.MODEL.neck.style)
        self.fc_cat = nn.Linear(int(2*config.MODEL.hidden_dim), config.MODEL.Embedding_dim)
        self.head = build_head(config)
        if config.MODEL.backbone.frozen:
            for param in self.backbone.parameters():
                param.requires_grad=False
    
    def forward_embedding(self, x):
        features = self.backbone(x)
        global_feat = self.neck_glob(self.global_pool(features[1]).squeeze())
        local_feat = self.local_branch(features[0])
        local_feat = self.orthogonal_fusion(local_feat, global_feat)
        local_feat = self.gap(local_feat).squeeze()
        feats = torch.cat([global_feat, local_feat], dim=1)
        feats = self.fc_cat(feats)
        return feats
    
    def forward(self, x):
        features = self.backbone(x)
        global_feat = self.neck_glob(self.global_pool(features[1]).squeeze())
        local_feat = self.local_branch(features[0])
        local_feat = self.orthogonal_fusion(local_feat, global_feat)
        local_feat = self.gap(local_feat).squeeze()
        feats = torch.cat([global_feat, local_feat], dim=1)
        feats = self.fc_cat(feats)
        logits = self.head(feats)
        return logits
