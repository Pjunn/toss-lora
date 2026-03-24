import torch
from cldm.toss import TOSS
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from torch import nn
import wandb
import lpips

# CRITICAL FIX: Completely disable gradient checkpointing to fix LoRA gradient flow
# The custom CheckpointFunction doesn't properly handle PEFT's dynamically added parameters
import ldm.modules.diffusionmodules.util as ldm_util
_original_checkpoint = ldm_util.checkpoint
def _no_checkpoint(func, inputs, params, flag):
    """Always bypass checkpointing - just run the function directly"""
    return func(*inputs)
ldm_util.checkpoint = _no_checkpoint
print("[PATCH] Disabled gradient checkpointing in ldm_util.checkpoint")

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
        run = wandb.init(
            entity="wndyd0131-sungkyunkwan-university",
            project="toss-lora",
            config={
                # LoRA config
                "lora_r": lora_config_params.get("r", 16),
                "lora_alpha": lora_config_params.get("lora_alpha", 16),
                "lora_dropout": lora_config_params.get("lora_dropout", 0.0),
                "target_modules": lora_config_params.get("target_modules", []),
                # Model config
                "architecture": "TOSS + LoRA",
                "base_model": "Stable Diffusion UNet"
            },
        )

        # Change pose_net in_feature channel from 51 to 16
        # self.model.diffusion_model.pose_net = nn.Sequential(
        #     nn.Linear(16, 320), # 51 -> 16
        #     nn.SiLU(),
        #     nn.Linear(320, 320)
        # )

        # Disable checkpoints
        unet = self.model.diffusion_model
        def disable_all_ckpt(m):
            for attr in ["use_checkpoint", "checkpoint", "use_checkpointing"]:
                if hasattr(m, attr):
                    setattr(m, attr, False)
        unet.apply(disable_all_ckpt)

        # 1. Freeze the base model
        self.requires_grad_(False)

        # 2. Configure LoRA (Critical)
        peft_config = LoraConfig(**lora_config_params)
        self.model.diffusion_model = get_peft_model(self.model.diffusion_model, peft_config)

        # CRITICAL: Disable checkpointing AGAIN after PEFT wrapping
        # PEFT changes module hierarchy, must ensure checkpointing is disabled
        self.model.diffusion_model.apply(disable_all_ckpt)
        
        # Verify checkpointing is disabled
        ckpt_enabled_count = 0
        for name, module in self.model.diffusion_model.named_modules():
            if hasattr(module, 'checkpoint') and module.checkpoint:
                ckpt_enabled_count += 1
                print(f"[WARNING] Checkpointing still enabled on: {name}")
        print(f"[INIT] Modules with checkpointing enabled: {ckpt_enabled_count}")

        # 3. Unfreeze PoseNet params
        pose_net_count = 0
        for n, p in self.model.diffusion_model.named_parameters():
            if "pose_net" in n:
                p.requires_grad = True
                pose_net_count += 1
                print(f"[INIT] Unfreezing pose_net param: {n}, shape={p.shape}")
        print(f"[INIT] Unfroze {pose_net_count} pose_net parameters")
        
        # 4. Explicitly enable gradients for LoRA parameters (in case PEFT didn't)
        lora_count = 0
        lora_params_info = []
        for n, p in self.model.diffusion_model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
                lora_count += 1
                lora_params_info.append((n, p.shape, p.requires_grad))

                # Initialize lora_B with small but meaningful values
                # This is essential for gradient flow - with lora_B=0, lora_A gets no gradients
                if "lora_B" in n:
                    print(f"[INIT] lora_B before init: {n}, mean={p.abs().mean().item():.6e}")
                    with torch.no_grad():
                        # Use kaiming uniform like lora_A for balanced gradients
                        nn.init.kaiming_uniform_(p, a=5**0.5)  # Same as lora_A default
                        p.mul_(0.01)  # Scale down to not disrupt pretrained model too much
                    print(f"[INIT] lora_B after init: {n}, mean={p.abs().mean().item():.6e}")
                        
            # Also enable output layers if needed
            if "base_model.model.out." in n:
                p.requires_grad = True

        print(f"[INIT] Enabled requires_grad for {lora_count} LoRA parameters")
        for name, shape, req_grad in lora_params_info:
            print(f"  -> {name}: shape={shape}, requires_grad={req_grad}")

        self.model.diffusion_model.print_trainable_parameters()
        
        # Verify PEFT is properly active
        if hasattr(self.model.diffusion_model, 'peft_config'):
            print(f"[INIT] PEFT config active: {self.model.diffusion_model.peft_config}")
        else:
            print("[WARNING] No peft_config found - PEFT may not be properly initialized!")
        
        # Debug: Check LoRA layer scaling and adapter status
        self._debug_lora_setup()
        
        # Debug: Check pose_net weights
        self._debug_pose_net_weights()


        # Initialize perceptual loss (LPIPS)
        self.lpips_loss = lpips.LPIPS(net='vgg').eval()
        self.lpips_loss.requires_grad_(False)  # Freeze LPIPS network
        print("[INIT] Initialized LPIPS perceptual loss (VGG backbone)")
        
        # Loss weights for hybrid loss
        self.perceptual_weight = 1.0  # Weight for perceptual loss
        self.mse_weight = 0.1  # Small MSE component for stability
        self.mask_min_weight = 0.2  # Soft mask: background contributes 20%, face contributes 100%

        # Debug: Check LoRA layer scaling and adapter status
        self._debug_lora_setup()
        self._debug_pose_net_weights()

    def _debug_pose_net_weights(self):
        """Debug helper to inspect pose_net weights"""
        print("\n[DEBUG] Pose Net Weight Analysis:")
        
        # Access pose_net (may be wrapped by PEFT)
        unet = self.model.diffusion_model
        pose_net = None
        
        # Try to find pose_net in the model
        if hasattr(unet, 'base_model'):
            # PEFT wrapped model
            if hasattr(unet.base_model, 'model') and hasattr(unet.base_model.model, 'pose_net'):
                pose_net = unet.base_model.model.pose_net
        elif hasattr(unet, 'pose_net'):
            pose_net = unet.pose_net
        
        if pose_net is None:
            print("  [WARNING] pose_net not found!")
            return
        
        print(f"  pose_net structure: {pose_net}")
        print(f"  pose_enc type: {getattr(unet.base_model.model if hasattr(unet, 'base_model') else unet, 'pose_enc', 'unknown')}")
        
        # Iterate through pose_net layers
        for name, param in pose_net.named_parameters():
            print(f"\n  Layer: pose_net.{name}")
            print(f"    shape: {param.shape}")
            print(f"    requires_grad: {param.requires_grad}")
            print(f"    mean: {param.data.mean().item():.6f}")
            print(f"    std: {param.data.std().item():.6f}")
            print(f"    min: {param.data.min().item():.6f}")
            print(f"    max: {param.data.max().item():.6f}")
            print(f"    abs_mean: {param.data.abs().mean().item():.6f}")
            
            # Check if weights look initialized (not all zeros)
            if param.data.abs().sum() == 0:
                print(f"    [WARNING] All zeros - may not be loaded properly!")
            
        # Print input/output dimensions
        if hasattr(pose_net, '0') and hasattr(pose_net[0], 'in_features'):
            print(f"\n  Input features (pose_net[0].in_features): {pose_net[0].in_features}")
        if hasattr(pose_net, '2') and hasattr(pose_net[2], 'out_features'):
            print(f"  Output features (pose_net[2].out_features): {pose_net[2].out_features}")

    @property
    def normal_estimator(self):
        """Lazy-load frozen DPT-Hybrid normal estimator."""
        if self._normal_estimator is None:
            from ldm.modules.midas.api import DPTNormalInference
            self._normal_estimator = DPTNormalInference(self.normal_estimator_path).to(self.device)
            print(f"[INIT] Loaded DPT-Hybrid normal estimator from {self.normal_estimator_path}")
        return self._normal_estimator

    def _register_lora_debug_hook(self):
        """Register hooks to debug LoRA forward pass"""
        from peft.tuners.lora import Linear as LoraLinear
        
        def make_hook(name):
            def hook(module, input, output):
                # Check if LoRA is actually contributing
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    x = input[0]
                    for adapter_name in module.lora_A.keys():
                        lora_A = module.lora_A[adapter_name]
                        lora_B = module.lora_B[adapter_name]
                        scaling = module.scaling.get(adapter_name, 1.0)
                        
                        # Compute LoRA contribution manually
                        lora_out = lora_B(lora_A(x)) * scaling
                        
                        print(f"\n[HOOK] {name}:")
                        print(f"  input mean: {x.abs().mean().item():.6f}")
                        print(f"  lora_A(x) mean: {lora_A(x).abs().mean().item():.6f}")
                        print(f"  lora_B(lora_A(x)) mean: {lora_B(lora_A(x)).abs().mean().item():.6f}")
                        print(f"  scaling: {scaling}")
                        print(f"  lora_contribution mean: {lora_out.abs().mean().item():.6f}")
                        print(f"  output mean: {output.abs().mean().item():.6f}")
                        break
            return hook
        
        # Find and hook one LoRA layer
        for name, module in self.model.diffusion_model.named_modules():
            if isinstance(module, LoraLinear) and 'to_q' in name:
                module.register_forward_hook(make_hook(name))
                print(f"[DEBUG] Registered forward hook on {name}")
                
                # Also register backward hook on the lora_B weight
                for adapter_name in module.lora_B.keys():
                    lora_B = module.lora_B[adapter_name]
                    def backward_hook(grad):
                        print(f"\n[BACKWARD HOOK] lora_B gradient received!")
                        print(f"  grad shape: {grad.shape if grad is not None else None}")
                        print(f"  grad mean: {grad.abs().mean().item() if grad is not None else None}")
                        return grad
                    lora_B.weight.register_hook(backward_hook)
                    print(f"[DEBUG] Registered backward hook on lora_B.weight")
                    break
                break

    def _debug_lora_setup(self):
        """Debug helper to verify LoRA is properly configured"""
        from peft.tuners.lora import LoraLayer
        
        print("\n[DEBUG] LoRA Layer Analysis:")
        for name, module in self.model.diffusion_model.named_modules():
            if isinstance(module, LoraLayer):
                # Check scaling
                scaling = getattr(module, 'scaling', {})
                # Check if adapter is disabled
                disable_adapters = getattr(module, 'disable_adapters', False)
                # Check merged status  
                merged = getattr(module, 'merged', False)
                
                print(f"  {name}:")
                print(f"    scaling: {scaling}")
                print(f"    disable_adapters: {disable_adapters}")
                print(f"    merged: {merged}")
                
                # Check lora_A and lora_B
                if hasattr(module, 'lora_A'):
                    for adapter_name, lora_a in module.lora_A.items():
                        print(f"    lora_A[{adapter_name}]: shape={lora_a.weight.shape}, requires_grad={lora_a.weight.requires_grad}")
                if hasattr(module, 'lora_B'):
                    for adapter_name, lora_b in module.lora_B.items():
                        print(f"    lora_B[{adapter_name}]: shape={lora_b.weight.shape}, requires_grad={lora_b.weight.requires_grad}")
                break  # Only check one to avoid spam
        
        # Check active adapter
        if hasattr(self.model.diffusion_model, 'active_adapter'):
            print(f"\n[DEBUG] Active adapter: {self.model.diffusion_model.active_adapter}")
        if hasattr(self.model.diffusion_model, 'active_adapters'):
            print(f"[DEBUG] Active adapters: {self.model.diffusion_model.active_adapters}")

    def _debug_pose_net_weights(self):
        """Debug helper to inspect pose_net weights"""
        print("\n[DEBUG] Pose Net Weight Analysis:")
        
        # Access pose_net (may be wrapped by PEFT)
        unet = self.model.diffusion_model
        pose_net = None
        
        # Try to find pose_net in the model
        if hasattr(unet, 'base_model'):
            # PEFT wrapped model
            if hasattr(unet.base_model, 'model') and hasattr(unet.base_model.model, 'pose_net'):
                pose_net = unet.base_model.model.pose_net
        elif hasattr(unet, 'pose_net'):
            pose_net = unet.pose_net
        
        if pose_net is None:
            print("  [WARNING] pose_net not found!")
            return
        
        print(f"  pose_net structure: {pose_net}")
        print(f"  pose_enc type: {getattr(unet.base_model.model if hasattr(unet, 'base_model') else unet, 'pose_enc', 'unknown')}")
        
        # Iterate through pose_net layers
        for name, param in pose_net.named_parameters():
            print(f"\n  Layer: pose_net.{name}")
            print(f"    shape: {param.shape}")
            print(f"    requires_grad: {param.requires_grad}")
            print(f"    mean: {param.data.mean().item():.6f}")
            print(f"    std: {param.data.std().item():.6f}")
            print(f"    min: {param.data.min().item():.6f}")
            print(f"    max: {param.data.max().item():.6f}")
            print(f"    abs_mean: {param.data.abs().mean().item():.6f}")
            
            # Check if weights look initialized (not all zeros)
            if param.data.abs().sum() == 0:
                print(f"    [WARNING] All zeros - may not be loaded properly!")
            
        # Print input/output dimensions
        if hasattr(pose_net, '0') and hasattr(pose_net[0], 'in_features'):
            print(f"\n  Input features (pose_net[0].in_features): {pose_net[0].in_features}")
        if hasattr(pose_net, '2') and hasattr(pose_net[2], 'out_features'):
            print(f"  Output features (pose_net[2].out_features): {pose_net[2].out_features}")

    def on_save_checkpoint(self, checkpoint):
        # We override this to prevent the parent class from 
        # trying to call self.embedding_manager.save()
        
        # If you want to keep the checkpoint small (LoRA only),
        # you can clear the base model weights from the checkpoint:
        # checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if "lora" in k or "pose_net" in k}
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
                # Training config from kwargs if available
                "image_size": self.image_size,
                "timesteps": self.num_timesteps,
                # Loss config
                "loss_type": "perceptual + mse",
                "perceptual_weight": self.perceptual_weight,
                "mse_weight": self.mse_weight,
                "mask_min_weight": self.mask_min_weight,
                "lpips_backbone": "vgg",
            }, allow_val_change=True)

    def training_step(self, batch, batch_idx):

        # Ensure model is in training mode
        self.model.diffusion_model.train()
        
        # CRITICAL: Explicitly enable LoRA adapters for PEFT
        if hasattr(self.model.diffusion_model, 'enable_adapters'):
            self.model.diffusion_model.enable_adapters()
        
        # Debug: verify LoRA is active on first step
        if batch_idx == 0 and self.global_step == 0:
            print(f"[TRAIN] Training step 0, verifying LoRA setup...")
            lora_active = 0
            for n, m in self.model.diffusion_model.named_modules():
                if 'lora' in n.lower():
                    lora_active += 1
            print(f"[TRAIN] Found {lora_active} LoRA modules in forward path")
            
            # CRITICAL: Verify LoRA weights are non-zero
            for n, p in self.model.diffusion_model.named_parameters():
                if "lora" in n.lower():
                    print(f"[TRAIN] {n}: mean={p.abs().mean().item():.6e}, requires_grad={p.requires_grad}")
            
            # Check adapter state
            if hasattr(self.model.diffusion_model, 'active_adapters'):
                print(f"[TRAIN] Active adapters: {self.model.diffusion_model.active_adapters}")
            if hasattr(self.model.diffusion_model, 'disable_adapters'):
                print(f"[TRAIN] disable_adapters attr: {getattr(self.model.diffusion_model, 'disable_adapters', 'N/A')}")

            # Register a hook to check LoRA layer output
            self._register_lora_debug_hook()

        # Ensure model is in training mode
        self.model.diffusion_model.train()
        
        # CRITICAL: Explicitly enable LoRA adapters for PEFT
        if hasattr(self.model.diffusion_model, 'enable_adapters'):
            self.model.diffusion_model.enable_adapters()
        
        # Debug: verify LoRA is active on first step
        if batch_idx == 0 and self.global_step == 0:
            print(f"[TRAIN] Training step 0, verifying LoRA setup...")
            lora_active = 0
            for n, m in self.model.diffusion_model.named_modules():
                if 'lora' in n.lower():
                    lora_active += 1
            print(f"[TRAIN] Found {lora_active} LoRA modules in forward path")
            
            # Register a hook to check LoRA layer output
            self._register_lora_debug_hook()

        x, cond = self.get_input(batch, self.first_stage_key)

        print("DEBUG_DELTA_POSE:", cond['delta_pose'][0])

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        '''Forward'''
        model_output = self.apply_model(x_noisy, t, cond)

        target = noise

        '''MSE Loss'''
        # loss = (model_output - target) ** 2
        # print("LOSS shape 1:", loss.shape)
        # loss = F.mse_loss(model_output, target, reduction="mean")
        # print("LOSS shape 2:", loss.shape)
        # # if torch.rand(1) < 0.1:
        # #     cond['in_concat'][0] = torch.zeros_like(cond['in_concat'][0])
        # # loss, loss_dict = self.forward(x, cond)

        '''Perceptual Loss Computation'''
        # Predict x0 from the noise prediction using the diffusion formula:
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # => x_0 = (x_t - sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        # Predict x0 from model's noise prediction
        pred_x0 = (x_noisy - sqrt_one_minus_alphas_cumprod * model_output) / sqrt_alphas_cumprod
        # Ground truth x0
        gt_x0 = x  # The clean latent we started with
        
        # Decode to image space for perceptual loss
        # Use torch.no_grad for decoder to save memory (only need gradients through encoder path)
        pred_img = self.decode_first_stage(pred_x0)  # [-1, 1] range
        gt_img = self.decode_first_stage(gt_x0)  # [-1, 1] range
        
        # Compute perceptual loss (LPIPS expects [-1, 1] range)
        # Move LPIPS to same device as images
        self.lpips_loss = self.lpips_loss.to(pred_img.device)
        perceptual_loss = self.lpips_loss(pred_img, gt_img).mean()

        # Optional: small MSE component on noise for training stability
        mse_loss = F.mse_loss(model_output, target, reduction="mean")

        '''Combine Loss'''
        loss = self.perceptual_weight * perceptual_loss + self.mse_weight * mse_loss
        print(f"LOSS: perceptual={perceptual_loss.item():.4f}, mse={mse_loss.item():.4f}, total={loss.item():.4f}")


        mask = batch.get("mask") # Original mask [B, 1, 256, 256]        

        '''Latent mask (MSE)'''
        # if mask is not None:
        #     # latent_mask = F.interpolate(mask, size=loss.shape[-2:], mode="area")
        #     soft_mask = mask * (1.0 - self.mask_min_weight) + self.mask_min_weight  # Range: [min_weight, 1.0]
        #     latent_mask = F.interpolate(soft_mask, size=model_output.shape[-2:], mode="area")
            
        #     # loss = loss * latent_mask
        #     # loss = loss.sum() / (latent_mask.sum() + 1e-8)
        #     loss = (F.mse_loss(model_output, noise, reduction="none") * latent_mask).sum() / (latent_mask.sum() + 1e-8)
        # else:
        #     loss = loss.mean()

        '''Latent mask (Perceptual)'''
        if mask is not None:
            # ===== Soft Mask =====
            # Convert binary mask to soft mask so non-masked regions still contribute
            # mask=1 (face) -> weight=1.0, mask=0 (background) -> weight=min_weight
            soft_mask = mask * (1.0 - self.mask_min_weight) + self.mask_min_weight  # Range: [min_weight, 1.0]
            
            # For perceptual loss with mask, we need to apply mask in image space
            img_mask = F.interpolate(soft_mask, size=pred_img.shape[-2:], mode="bilinear", align_corners=False)
            
            # Masked perceptual loss: compute per-pixel LPIPS isn't straightforward,
            # so we use a masked MSE on decoded images as an approximation
            masked_img_loss = (F.mse_loss(pred_img, gt_img, reduction="none") * img_mask).mean()
            
            # Also apply mask to latent MSE
            latent_mask = F.interpolate(soft_mask, size=model_output.shape[-2:], mode="area")
            masked_mse_loss = (F.mse_loss(model_output, noise, reduction="none") * latent_mask).mean()
            
            # Combine masked losses
            loss = self.perceptual_weight * masked_img_loss + self.mse_weight * masked_mse_loss
            print(f"MASKED LOSS: img={masked_img_loss.item():.4f}, mse={masked_mse_loss.item():.4f}")

        ''' Geometry loss (frozen DPT-Hybrid proxy) '''
        if self.geometry_loss_weight > 0 and "normal" in batch and "normal_mask" in batch:
            x0_pred = x - model_output
            pred_imgs = self.decode_first_stage(x0_pred)
            pred_imgs = torch.clamp((pred_imgs + 1) / 2, 0, 1)
            if pred_imgs.ndim == 4 and pred_imgs.shape[-1] == 3:
                pred_imgs = pred_imgs.permute(0, 3, 1, 2)
            pred_normals = self.normal_estimator(pred_imgs)
            gt_normals = batch["normal"].to(self.device)
            normal_mask = batch["normal_mask"].to(self.device)
            geom_loss = _cosine_similarity_loss(pred_normals, gt_normals, normal_mask)
            loss = loss + self.geometry_loss_weight * geom_loss

        run.log({
            "loss": loss,
            "perceptual_loss": perceptual_loss,
            "mse_loss": mse_loss,
            "geometry_loss": geom_loss
        })
        print(f"LOSS logged: total={loss.item():.4f}")

        '''WanDB logging'''
        if batch_idx % 500 == 0:
            # Generate 4 multiview predictions from a single source image
            with torch.no_grad():
                import math
                
                # Get one source image from batch
                source_img = batch[self.control_key][:1].to(self.device)  # [1, C, H, W]
                if source_img.ndim == 4 and source_img.shape[-1] == 3:
                    source_img = source_img.permute(0, 3, 1, 2)
                source_img_display = torch.clamp(source_img, 0, 1)
                
                # Encode source image to latent
                source_latent = self.encode_first_stage(source_img * 2 - 1).mode().detach()
                
                # Get text conditioning (empty)
                c_text = self.get_learned_conditioning([""])
                
                # Define 4 different yaw angles for multiview (in radians)
                # e.g., -15°, -5°, +5°, +15°
                yaw_angles_deg = [-15, -5, 5, 15]
                
                wandb_images = []
                
                # Add source image first
                wandb_images.append(wandb.Image(
                    source_img_display[0],
                    caption=f"Step {self.global_step} | SOURCE"
                ))
                
                # Generate prediction for each pose
                for yaw_deg in yaw_angles_deg:
                    yaw_rad = math.radians(yaw_deg)
                    # Create pose: [pitch, yaw, distance]
                    delta_pose_mv = torch.tensor([[0.0, yaw_rad, 0.0]], device=self.device)
                    
                    # Create conditioning dict for this pose
                    cond_mv = {
                        'c_crossattn': [c_text],
                        'c_concat': [source_img],
                        'in_concat': [source_latent],
                        'delta_pose': delta_pose_mv
                    }
                    
                    # Sample using DDIM for faster inference
                    from ldm.models.diffusion.ddim import DDIMSampler
                    sampler = DDIMSampler(self)
                    
                    shape = [4, source_img.shape[2] // 8, source_img.shape[3] // 8]
                    
                    # Use fewer steps for visualization (faster)
                    samples, _ = sampler.sample(
                        S=20,  # Quick sampling
                        batch_size=1,
                        shape=shape,
                        conditioning=cond_mv,
                        verbose=False,
                        unconditional_guidance_scale=1.0,
                        eta=0.0
                    )
                    
                    # Decode to image
                    pred_img = self.decode_first_stage(samples)
                    pred_img = torch.clamp((pred_img + 1) / 2, 0, 1)
                    
                    wandb_images.append(wandb.Image(
                        pred_img[0],
                        caption=f"Step {self.global_step} | Yaw: {yaw_deg}°"
                    ))
                
                # Log all multiview predictions
                run.log({"multiview_predictions": wandb_images})
                print(f"[VIS] Logged multiview predictions at step {self.global_step}")

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def on_after_backward(self):
        # Print every 50 steps for better visibility
        if self.global_step % 50 == 0:
            print(f"\n[GRAD CHECK] Step {self.global_step}")
            
            lora_grads = []
            pose_net_grads = []
            
            for n, p in self.model.diffusion_model.named_parameters():
                if not p.requires_grad:
                    continue
                    
                if "lora" in n.lower():
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        grad_mean = p.grad.abs().mean().item()
                        grad_max = p.grad.abs().max().item()
                        lora_grads.append((n, grad_norm, grad_mean, grad_max))
                    else:
                        lora_grads.append((n, None, None, None))
                        
                elif "pose_net" in n:
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        pose_net_grads.append((n, grad_norm))
                    else:
                        pose_net_grads.append((n, None))
            
            # Print LoRA gradients
            print(f"  LoRA params ({len(lora_grads)}):")
            has_nonzero_grad = False
            for name, norm, mean, max_val in lora_grads:
                if norm is not None:
                    if norm > 1e-10:
                        has_nonzero_grad = True
                    print(f"    {name.split('.')[-3]}: norm={norm:.2e}, mean={mean:.2e}, max={max_val:.2e}")
                else:
                    print(f"    {name.split('.')[-3]}: grad=None!")
            
            if not has_nonzero_grad:
                print("  [WARNING] All LoRA gradients are zero or None!")
            
            # Print pose_net gradients  
            print(f"  pose_net params ({len(pose_net_grads)}):")
            for name, norm in pose_net_grads:
                if norm is not None:
                    print(f"    {name}: norm={norm:.2e}")
                else:
                    print(f"    {name}: grad=None!")


    def configure_optimizers(self):
        # Explicitly collect LoRA and pose_net params separately
        lora_params = []
        pose_net_params = []
        other_params = []
        
        # Use named_parameters to ensure we get the actual parameter objects
        for n, p in self.model.diffusion_model.named_parameters():
            if p.requires_grad:
                if "lora" in n.lower():
                    lora_params.append(p)
                    print(f"[OPT] LoRA param: {n}, shape={p.shape}")
                elif "pose_net" in n:
                    pose_net_params.append(p)
                    print(f"[OPT] pose_net param: {n}, shape={p.shape}")
                else:
                    other_params.append(p)
                    print(f"[OPT] Other param: {n}, shape={p.shape}")
        
        print(f"\n[OPT] Summary:")
        print(f"  LoRA params: {len(lora_params)}")
        print(f"  pose_net params: {len(pose_net_params)}")
        print(f"  Other trainable params: {len(other_params)}")
        print(f"  Total params in optimizer: {len(lora_params) + len(pose_net_params) + len(other_params)}")
        
        if len(lora_params) == 0:
            print("[WARNING] No LoRA params found! Check if PEFT is properly configured.")
            # Fallback: try to get params differently
            for n, p in self.named_parameters():
                if "lora" in n.lower() and p.requires_grad:
                    lora_params.append(p)
                    print(f"[OPT-FALLBACK] Found LoRA param: {n}")
        
        # Use separate param groups with different learning rates
        # pose_net uses lower LR (0.1x) for gentle fine-tuning
        param_groups = [
            {"params": lora_params, "lr": self.learning_rate, "name": "lora"},
        ]
        
        if len(pose_net_params) > 0:
            pose_net_lr = self.learning_rate * 0.1  # 10x lower than LoRA
            param_groups.append({"params": pose_net_params, "lr": pose_net_lr, "name": "pose_net"})
            print(f"[OPT] pose_net learning rate: {pose_net_lr} (0.1x of LoRA lr: {self.learning_rate})")
        
        if len(other_params) > 0:
            param_groups.append({"params": other_params, "lr": self.learning_rate, "name": "other"})
        
        optimizer = torch.optim.AdamW(param_groups)
        
        return optimizer