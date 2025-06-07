#!/usr/bin/env python3
"""
Simplified SDXL Training Script
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    StableDiffusionXLPipeline,
    DDPMScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import wandb

console = Console()

class SDXLDataset(Dataset):
    """Simple dataset for SDXL training."""
    
    def __init__(self, data_dir, tokenizer_one, tokenizer_two, size=1024, center_crop=True):
        self.data_dir = Path(data_dir)
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.size = size
        self.center_crop = center_crop
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(self.data_dir.glob(f'**/{ext}'))
        
        console.print(f"[green]Found {len(self.image_paths)} images[/green]")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Simple resize and crop
        if self.center_crop:
            # Center crop to square
            width, height = image.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            image = image.crop((left, top, left + size, top + size))
        
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        image = np.array(image) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Load caption (assume .txt file with same name)
        caption_path = image_path.with_suffix('.txt')
        if caption_path.exists():
            caption = caption_path.read_text().strip()
        else:
            caption = ""
        
        # Tokenize caption
        tokens_one = self.tokenizer_one(
            caption,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).input_ids[0]
        
        tokens_two = self.tokenizer_two(
            caption,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "pixel_values": image,
            "input_ids_one": tokens_one,
            "input_ids_two": tokens_two,
            "caption": caption
        }

def get_models(model_name, device, dtype):
    """Load SDXL models."""
    console.print("[yellow]Loading models...[/yellow]")
    
    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    
    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        model_name, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", torch_dtype=dtype
    ).to(device)
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", torch_dtype=dtype
    ).to(device)
    
    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    return {
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler
    }

def encode_prompt(text_encoders, tokenized_prompts):
    """Encode prompt for SDXL."""
    prompt_embeds_list = []
    
    for tokenized_prompt, text_encoder in zip(tokenized_prompts, text_encoders):
        prompt_embeds = text_encoder(
            tokenized_prompt,
            output_hidden_states=True,
        )
        
        # We are only interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)
    
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds

def train_step(models, batch, config, accelerator):
    """Single training step."""
    vae = models["vae"]
    unet = models["unet"]
    noise_scheduler = models["noise_scheduler"]
    text_encoders = [models["text_encoder_one"], models["text_encoder_two"]]
    
    # Move batch to device
    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
    
    # Encode images to latents
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    # Sample noise
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    
    # Sample timesteps
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
    ).long()
    
    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Encode prompts
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders,
        [batch["input_ids_one"], batch["input_ids_two"]]
    )
    
    # Prepare added time embeddings for SDXL
    add_time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]] * bsz,
        dtype=prompt_embeds.dtype,
        device=accelerator.device
    )
    
    # Predict the noise residual
    model_pred = unet(
        noisy_latents,
        timesteps,
        prompt_embeds,
        added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
        return_dict=False,
    )[0]
    
    # Get target
    if config.get("advanced", {}).get("v_prediction", False):
        # v-prediction parameterization
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise
    
    # Calculate loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    
    return loss

def main():
    parser = argparse.ArgumentParser(description="SDXL Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with="wandb" if config["logging"]["use_wandb"] else None,
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup logging
    if accelerator.is_main_process:
        console.print("[bold green]SDXL Training Script[/bold green]")
        console.print(f"Config: {args.config}")
        
        if config["logging"]["use_wandb"]:
            wandb.init(
                project=config["logging"]["wandb_project"],
                config=config,
                name=f"sdxl-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
    
    # Setup dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.float32)
    
    # Load models
    models = get_models(config["model"]["name"], accelerator.device, dtype)
    
    # Setup dataset and dataloader
    dataset = SDXLDataset(
        config["dataset"]["train_dir"],
        models["tokenizer_one"],
        models["tokenizer_two"],
        size=config["dataset"]["image_size"],
        center_crop=config["dataset"]["center_crop"]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        models["unet"].parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Setup learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(dataloader) * config["training"]["num_epochs"],
        eta_min=1e-8
    )
    
    # Prepare for training
    models["unet"], optimizer, dataloader, lr_scheduler = accelerator.prepare(
        models["unet"], optimizer, dataloader, lr_scheduler
    )
    
    # Enable gradient checkpointing
    if config["training"]["gradient_checkpointing"]:
        models["unet"].enable_gradient_checkpointing()
    
    # Training loop
    global_step = 0
    for epoch in range(config["training"]["num_epochs"]):
        models["unet"].train()
        
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch in progress:
            with accelerator.accumulate(models["unet"]):
                loss = train_step(models, batch, config, accelerator)
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if global_step % config["logging"]["log_every_n_steps"] == 0:
                accelerator.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
                progress.set_postfix(loss=f"{loss.item():.4f}")
            
            # Checkpointing
            if global_step % config["training"]["save_every_n_steps"] == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint_path = Path(config["training"]["checkpoint_dir"]) / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save model
                    accelerator.unwrap_model(models["unet"]).save_pretrained(
                        checkpoint_path / "unet",
                        safe_serialization=True
                    )
                    
                    # Save optimizer and scheduler states
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }, checkpoint_path / "training_state.pt")
                    
                    console.print(f"[green]Saved checkpoint to {checkpoint_path}[/green]")
            
            global_step += 1
    
    # Save final model
    if accelerator.is_main_process:
        final_path = Path(config["training"]["checkpoint_dir"]) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        
        accelerator.unwrap_model(models["unet"]).save_pretrained(
            final_path / "unet",
            safe_serialization=True
        )
        
        console.print(f"[bold green]Training complete! Final model saved to {final_path}[/bold green]")

if __name__ == "__main__":
    main()