import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False, # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
           self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)


class DPTNormalModel(DPT):
    """DPT-Hybrid for surface normal estimation. Outputs L2-normalized [3, H, W] per pixel."""

    def __init__(self, path=None, **kwargs):
        features = kwargs.get("features", 256)

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self._load_weights(path)

    def _load_weights(self, path):
        """Load weights from Omnidata checkpoint (supports HF repo, Lightning ckpt, raw state_dict)."""
        if path.startswith("hf:") or path.startswith("hf://"):
            from huggingface_hub import hf_hub_download, list_repo_files
            repo_id = path.split(":", 1)[-1].replace("//", "").strip()
            candidates = [
                "omnidata_dpt_normal_v2.ckpt",
                "omnidata_normal_dpt_hybrid.pth",
                "pytorch_model.bin",
            ]
            try:
                files = list_repo_files(repo_id)
                for c in candidates:
                    if c in files:
                        path = hf_hub_download(repo_id=repo_id, filename=c)
                        break
                else:
                    path = hf_hub_download(repo_id=repo_id, filename=candidates[0])
            except Exception:
                path = hf_hub_download(repo_id=repo_id, filename=candidates[0])

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
        # Handle Lightning prefix "model."
        if state and any(k.startswith("model.") for k in state.keys()):
            state = {k.replace("model.", ""): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[DPTNormal] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[DPTNormal] Unexpected keys: {len(unexpected)}")

    def forward(self, x):
        out = super().forward(x)  # [B, 3, H, W]
        # L2 normalize per pixel for cosine similarity
        norm = out.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return out / norm

