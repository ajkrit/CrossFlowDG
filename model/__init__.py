import os
from functools import partial

from transformers import CLIPModel
from .CrossFlowDG import *

import torch
import torch.nn as nn
from collections import OrderedDict

from .VMamba import VSSM, Permute
from .VAE import *
from .flow_model import *

try:
    from .heat import HeatM
except:
    HeatM = None

clip_emb_dims = {
    "openai/clip-vit-base-patch32": 512,
    "openai/clip-vit-base-patch16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "RN50x16": 768,
    "RN50x64": 1024,
}


# still on developing...
def build_vssm_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["VMamba"]:
        model = VMamba.Backbone_VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=False, # config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # ===================
            posembed=False, # config.MODEL.VSSM.POSEMBED,
            imgsize=config.DATA.IMG_SIZE,
        )

        if is_pretrain:
            # load pretrain weights
            ckpt_path = config.MODEL.PRETRAINED
            if os.path.exists(ckpt_path):
                print(f"Loading pre-trained weights from {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(state_dict["model"], strict=False) # change [...] depending on the type of state dict
                del state_dict
                torch.cuda.empty_cache()
            else:
                print(f"Pre-trained weights not found at {ckpt_path}")

        # for param in model.parameters():
        #     param.requires_grad = False

        return model

    return None


def build_txt_vae(config):
    input_dim = clip_emb_dims[config.CLIP.NAME]
    latent_dim = config.FLOW.LATENT_DIM
    vae = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[512, 256],
        activation_fn=nn.GELU,
        use_batchnorm=True,
        dropout_rate=0.0)
    return vae

def build_img_vae(config):
    input_dim = 1440 # sum of channel dimensions across feature maps (e.g., for VMamba 768+384+192+96)
    latent_dim = config.FLOW.LATENT_DIM
    vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[1024, 512, 256],
            activation_fn=nn.GELU,
            use_batchnorm=True,
            dropout_rate=0.0)
    return vae



# still on developing...
def build_heat_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["heat"]:
        model = HeatM(
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        )
        return model


def build_flow_model(config):
    model = ResNet(latent_dim=config.FLOW.LATENT_DIM)
    return model
    

def build_classifier(config):
    latent_dim = config.FLOW.LATENT_DIM
    model = Classifier(
        embedding_dim=latent_dim, 
        num_classes=config.MODEL.NUM_CLASSES, 
        hidden_dims=[64, 64], 
        dropout_prob=0.2
    )
    return model


# used for analyze
def build_mmpretrain_models(cfg="swin_tiny", ckpt=True, only_backbone=False, with_norm=True, **kwargs):
    import os
    from functools import partial
    from mmengine.runner import CheckpointLoader
    from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
    from mmengine.config import Config
    config_root = os.path.join(os.path.dirname(__file__), "../../analyze/mmpretrain_configs/configs/") 
    
    CFGS = dict(
        swin_tiny=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-tiny_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth",
        ),
        convnext_tiny=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-tiny_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth",
        ),
        deit_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./deit/deit-small_4xb256_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth",
        ),
        resnet50=dict(
            model=Config.fromfile(os.path.join(config_root, "./resnet/resnet50_8xb32_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth",
        ),
        # ================================
        swin_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-small_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth",
        ),
        convnext_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-small_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth",
        ),
        deit_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./deit/deit-base_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth",
        ),
        resnet101=dict(
            model=Config.fromfile(os.path.join(config_root, "./resnet/resnet101_8xb32_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth",
        ),
        # ================================
        swin_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-base_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth",
        ),
        convnext_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-base_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.pth",
        ),
        replknet_base=dict(
            # comment this "from mmpretrain.models import build_classifier" in __base__/models/replknet...
            model=Config.fromfile(os.path.join(config_root, "./replknet/replknet-31B_32xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k_20221118-fd08e268.pth",
        ),
    )

    if cfg not in CFGS:
        return None

    model: ImageClassifier = build_classifier(CFGS[cfg]['model'])
    if ckpt:
        model.load_state_dict(CheckpointLoader.load_checkpoint(CFGS[cfg]['ckpt'])['state_dict'])

    if only_backbone:
        if isinstance(model.backbone, ConvNeXt):
            model.backbone.gap_before_final_norm = False
        if isinstance(model.backbone, VisionTransformer):
            model.backbone.out_type = 'featmap'

        def forward_backbone(self: ImageClassifier, x):
            x = self.backbone(x)[-1]
            return x
        if not with_norm:
            setattr(model, f"norm{model.backbone.out_indices[-1]}", lambda x: x)
        model.forward = partial(forward_backbone, model)

    return model


# used for analyze
def build_vssm_models_(cfg="vssm_tiny", ckpt=True, only_backbone=False, with_norm=True,
    CFGS = dict(
        vssm_tiny=dict(
            model=dict(
                depths=[2, 2, 9, 2], 
                dims=96, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.1, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth"),
        ),
        vssm_small=dict(
            model=dict(
                depths=[2, 2, 27, 2], 
                dims=96, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.3, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth"),
        ),
        vssm_base=dict(
            model=dict(
                depths=[2, 2, 27, 2], 
                dims=128, 
                d_state=16, 
                dt_rank="auto", 
                ssm_ratio=2.0, 
                attn_drop_rate=0., 
                drop_rate=0., 
                drop_path_rate=0.6, 
                mlp_ratio=0.0,
                downsample_version="v1",
            ),  
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmbase/ckpt_epoch_260.pth"),
        ),
    ),
    ckpt_key="model",
    **kwargs):
    if cfg not in CFGS:
        return None
    
    model_params = CFGS[cfg]["model"]
    model_ckpt = CFGS[cfg]["ckpt"]

    model = VSSM(**model_params)
    if only_backbone:
        if with_norm:
            def forward(self: VSSM, x: torch.Tensor):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                x = self.classifier.norm(x)
                x = x.permute(0, 3, 1, 2).contiguous()
                return x
            model.forward = partial(forward, model)
            del model.classifier.norm
            del model.classifier.head
            del model.classifier.avgpool
        else:
            def forward(self: VSSM, x: torch.Tensor):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                x = x.permute(0, 3, 1, 2).contiguous()
                return x
            model.forward = partial(forward, model)
            del model.classifier

    if ckpt:
        ckpt = model_ckpt
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = model.load_state_dict(_ckpt[ckpt_key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    return model


# used for analyze
def build_heat_models_(cfg="heat_tiny", ckpt=True, only_backbone=False, with_norm=True,
    CFGS = dict(
        heat_mini=dict(
            model=dict(
                depths=[2, 2, 2, 1], 
                dims=96, 
                drop_path_rate=0.05, 
                mlp_ratio=0.0,
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/heat/heat_mini/ckpt_epoch_296.pth"),
            tag="model_ema",
        ),
        heat_tiny=dict(
            model=dict(
                depths=[2, 2, 6, 2], 
                dims=96, 
                drop_path_rate=0.2, 
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/heat/heat_tiny/ckpt_epoch_288.pth"),
            tag="model",
        ),
        heat_small=dict(
            model=dict(
                depths=[2, 2, 18, 2], 
                dims=96, 
                drop_path_rate=0.3, 
            ), 
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_276.pth"),
            tag="model_ema",
            comment="not finish...",
        ),
        heat_base=dict(
            model=dict(
                depths=[2, 2, 18, 2], 
                dims=128, 
                drop_path_rate=0.5, 
            ),  
            ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpts/classification/heat/heat_base/ckpt_epoch_288.pth"),
            tag="model",
        ),
    ),
    **kwargs):
    if cfg not in CFGS:
        return None
    
    model_params = CFGS[cfg]["model"]
    model_ckpt = CFGS[cfg]["ckpt"]
    ckpt_key = CFGS[cfg]["tag"]

    model = HeatM(**model_params)
    if only_backbone:
        if with_norm:
            def forward(self: HeatM, x: torch.Tensor):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                x = self.classifier.norm(x)
                # x = x.permute(0, 3, 1, 2).contiguous()
                return x
            model.forward = partial(forward, model)
            del model.classifier.norm
            del model.classifier.head
            del model.classifier.avgpool
        else:
            def forward(self: VSSM, x: torch.Tensor):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                # x = x.permute(0, 3, 1, 2).contiguous()
                return x
            model.forward = partial(forward, model)
            del model.classifier

    if ckpt:
        ckpt = model_ckpt
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = model.load_state_dict(_ckpt[ckpt_key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    return model



def build_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPModel.from_pretrained(config.CLIP.NAME, use_safetensors=True, device_map=device)

    # Freeze CLIP
    for param in clip_model.parameters():
        param.requires_grad = False

    encoder = None
    img_vae = None
    classifier = None

    if config.MODEL.TYPE in ["VMamba"]:
        encoder = build_vssm_model(config, config.MODEL.PRETRAINED is not None)
        img_vae = build_img_vae(config)
    
    classifier = build_classifier(config)

    txt_vae = build_txt_vae(config)

    flow_model = build_flow_model(config)

    for param in flow_model.parameters():
        param.requires_grad = True
    for param in img_vae.parameters():
        param.requires_grad = True
    for param in classifier.parameters():
        param.requires_grad = True
    for param in txt_vae.parameters():
        param.requires_grad = True
    for param in encoder.parameters():
        param.requires_grad = True

    model = CrossFlowDG(
        clip_model=clip_model,
        txt_vae=txt_vae,
        encoder=encoder,
        img_vae=img_vae,
        flow_model=flow_model,
        classifier=classifier,
        device=device
    )

    return model


def build_model_eval(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = None
    img_vae = None
    classifier = None

    if config.MODEL.TYPE in ["VMamba"]:
        encoder = build_vssm_model(config, config.MODEL.PRETRAINED is not None)
        img_vae = build_img_vae(config)
    
    classifier = build_classifier(config)

    txt_vae = build_txt_vae(config)

    flow_model = build_flow_model(config)

    model = CrossFlowDG(
        txt_vae=txt_vae,
        encoder=encoder,
        img_vae=img_vae,
        flow_model=flow_model,
        classifier=classifier,
        device=device
    )

    for param in model.parameters():
        param.requires_grad = False

    return model
