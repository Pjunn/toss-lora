import torch
from cldm.toss import TOSS
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch import nn
import wandb
import lpips
import pytz
from datetime import datetime
# CRITICAL FIX: Completely disable gradient checkpointing to fix LoRA gradient flow
# The custom CheckpointFunction doesn't properly handle PEFT's dynamically added parameters
import ldm.modules.diffusionmodules.util as ldm_util
def _no_checkpoint(func, inputs, params, flag):
    """Always bypass checkpointing - just run the function directly"""
    return func(*inputs)
ldm_util.checkpoint = _no_checkpoint

run = None

def _cosine_similarity_loss(pred_normals, gt_normals, mask, eps=1e-8):
    """Masked cosine similarity loss. pred, gt: [B,3,H,W] L2-normalized. mask: [B,1,H,W]."""
    cos_sim = (pred_normals * gt_normals).sum(dim=1, keepdim=True).clamp(-1, 1)
    loss = 1 - cos_sim  # [B,1,H,W]
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + eps)
    return loss.mean()

class TossLoraModule(TOSS):
    def __init__(self, lora_config_params, *args, normal_estimator_path="hf:clay3d/omnidata", geometry_loss_weight=0.1, **kwargs):
        kwargs.pop("lora_config_params", None)  # consumed by us
        self.normal_estimator_path = kwargs.pop("normal_estimator_path", normal_estimator_path)
        self.geometry_loss_weight = kwargs.pop("geometry_loss_weight", geometry_loss_weight)
        super().__init__(*args, **kwargs)
        self._normal_estimator = None

        global run
        kst = pytz.timezone("Asia/Seoul")
        now_kst = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            entity="jsphilip54",
            project="toss-lora",
            name=f"toss-lora_{now_kst}",
            config={
                "lora_r": lora_config_params.get("r", 16),
                "lora_alpha": lora_config_params.get("lora_alpha", 16),
                "lora_dropout": lora_config_params.get("lora_dropout", 0.0),
                "target_modules": lora_config_params.get("target_modules", []),
                "architecture": "TOSS + LoRA",
                "base_model": "Stable Diffusion UNet"
            },
        )

        # Disable checkpoints
        unet = self.model.diffusion_model
        def disable_all_ckpt(m):
            for attr in ["use_checkpoint", "checkpoint", "use_checkpointing"]:
                if hasattr(m, attr):
                    setattr(m, attr, False)
        unet.apply(disable_all_ckpt)

        # 1. Freeze the base model
        self.requires_grad_(False)

        # 2. Configure LoRA
        peft_config = LoraConfig(**lora_config_params)
        self.model.diffusion_model = get_peft_model(self.model.diffusion_model, peft_config)

        # Disable checkpointing again after PEFT wrapping
        self.model.diffusion_model.apply(disable_all_ckpt)

        if not hasattr(self.model.diffusion_model, 'peft_config'):
            print("[WARNING] No peft_config found - PEFT may not be properly initialized!")

        # 3. Unfreeze PoseNet params
        for n, p in self.model.diffusion_model.named_parameters():
            if "pose_net" in n:
                p.requires_grad = True

        # 4. Explicitly enable gradients for LoRA parameters
        for n, p in self.model.diffusion_model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
                if "lora_B" in n:
                    with torch.no_grad():
                        nn.init.kaiming_uniform_(p, a=5**0.5)
                        p.mul_(0.01)

            if "base_model.model.out." in n:
                p.requires_grad = True

        self.model.diffusion_model.print_trainable_parameters()

        # Initialize perceptual loss (LPIPS)
        self.lpips_loss = lpips.LPIPS(net='vgg').eval()
        self.lpips_loss.requires_grad_(False)

        # Loss weights
        self.perceptual_weight = 1.0
        self.mse_weight = 0.1

    @property
    def normal_estimator(self):
        """Lazy-load frozen DPT-Hybrid normal estimator."""
        if self._normal_estimator is None:
            from ldm.modules.midas.api import DPTNormalInference
            self._normal_estimator = DPTNormalInference(self.normal_estimator_path).to(self.device)
        return self._normal_estimator

    def on_save_checkpoint(self, checkpoint):
        pass

    def on_train_start(self):
        """Called by PyTorch Lightning when training starts - log trainer config"""
        if run is not None and self.trainer is not None:
            run.config.update({
                "max_epochs": self.trainer.max_epochs,
                "max_steps": self.trainer.max_steps,
                "batch_size": self.trainer.datamodule.batch_size if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule else None,
                "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
                "gradient_clip_val": self.trainer.gradient_clip_val,
                "learning_rate": self.learning_rate,
                "image_size": self.image_size,
                "timesteps": self.num_timesteps,
                "loss_type": "perceptual + mse",
                "perceptual_weight": self.perceptual_weight,
                "mse_weight": self.mse_weight,
                "lpips_backbone": "vgg",
            }, allow_val_change=True)

    def training_step(self, batch, batch_idx):
        self.model.diffusion_model.train()
        if hasattr(self.model.diffusion_model, 'enable_adapters'):
            self.model.diffusion_model.enable_adapters()

        x, cond = self.get_input(batch, self.first_stage_key)

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t, cond)

        # Predict x0 from noise prediction
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        pred_x0 = (x_noisy - sqrt_one_minus_alphas_cumprod * model_output) / sqrt_alphas_cumprod

        # Decode to image space for perceptual loss
        pred_img = self.decode_first_stage(pred_x0)
        gt_img = self.decode_first_stage(x)

        # Perceptual loss (LPIPS)
        self.lpips_loss = self.lpips_loss.to(pred_img.device)
        perceptual_loss = self.lpips_loss(pred_img, gt_img).mean()

        # MSE loss on noise prediction
        mse_loss = F.mse_loss(model_output, noise, reduction="mean")

        loss = self.perceptual_weight * perceptual_loss + self.mse_weight * mse_loss

        loss_log = {
            "loss": loss,
            "perceptual_loss": perceptual_loss,
            "mse_loss": mse_loss,
        }

        # Geometry loss
        geom_loss = torch.tensor(0.0, device=self.device)
        if self.geometry_loss_weight > 0 and "normal" in batch and "normal_mask" in batch:
            pred_imgs = torch.clamp((pred_img + 1) / 2, 0, 1)
            if pred_imgs.ndim == 4 and pred_imgs.shape[-1] == 3:
                pred_imgs = pred_imgs.permute(0, 3, 1, 2)
            pred_normals = self.normal_estimator(pred_imgs)
            gt_normals = batch["normal"].to(self.device)
            normal_mask = batch["normal_mask"].to(self.device)
            geom_loss = _cosine_similarity_loss(pred_normals, gt_normals, normal_mask)
            loss = loss + self.geometry_loss_weight * geom_loss
            loss_log["geometry_loss"] = geom_loss

        run.log(loss_log)

        # WandB image logging: source | GT | prediction (same pose)
        if batch_idx % 50 == 0:
            with torch.no_grad():
                import math
                from ldm.models.diffusion.ddim import DDIMSampler

                # Source image
                source_img = batch[self.control_key][:1].to(self.device)
                if source_img.ndim == 4 and source_img.shape[-1] == 3:
                    source_img = source_img.permute(0, 3, 1, 2)
                source_img_display = torch.clamp(source_img, 0, 1)

                # GT target image (same as used in loss)
                gt_img_raw = batch[self.first_stage_key][:1].to(self.device)
                if gt_img_raw.ndim == 4 and gt_img_raw.shape[-1] == 3:
                    gt_img_raw = gt_img_raw.permute(0, 3, 1, 2)
                gt_display = torch.clamp((gt_img_raw + 1) / 2, 0, 1)

                # Generate prediction for the same pose as GT
                source_latent = self.encode_first_stage(source_img * 2 - 1).mode().detach()
                c_text = self.get_learned_conditioning([""])
                cond_vis = {
                    'c_crossattn': [c_text],
                    'c_concat': [source_img],
                    'in_concat': [source_latent],
                    'delta_pose': batch["delta_pose"][:1],
                }
                sampler = DDIMSampler(self)
                shape = [4, source_img.shape[2] // 8, source_img.shape[3] // 8]
                samples, _ = sampler.sample(
                    S=20, batch_size=1, shape=shape,
                    conditioning=cond_vis, verbose=False,
                    unconditional_guidance_scale=1.0, eta=0.0
                )
                pred_display = torch.clamp((self.decode_first_stage(samples) + 1) / 2, 0, 1)

                yaw_deg = math.degrees(batch["delta_pose"][0, 1].item())
                run.log({
                    "vis/source": wandb.Image(source_img_display[0], caption="Source"),
                    "vis/gt":     wandb.Image(gt_display[0],          caption=f"GT (yaw={yaw_deg:.1f}°)"),
                    "vis/pred":   wandb.Image(pred_display[0],         caption=f"Pred (yaw={yaw_deg:.1f}°)"),
                })

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        lora_params = []
        pose_net_params = []
        other_params = []

        for n, p in self.model.diffusion_model.named_parameters():
            if p.requires_grad:
                if "lora" in n.lower():
                    lora_params.append(p)
                elif "pose_net" in n:
                    pose_net_params.append(p)
                else:
                    other_params.append(p)

        param_groups = [{"params": lora_params, "lr": self.learning_rate, "name": "lora"}]

        if len(pose_net_params) > 0:
            param_groups.append({"params": pose_net_params, "lr": self.learning_rate * 0.1, "name": "pose_net"})

        if len(other_params) > 0:
            param_groups.append({"params": other_params, "lr": self.learning_rate, "name": "other"})

        return torch.optim.AdamW(param_groups)
