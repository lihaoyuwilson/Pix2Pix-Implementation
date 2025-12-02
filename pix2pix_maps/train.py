#!/usr/bin/env python3
"""
Pix2Pix training on the 'maps' dataset.

- U-Net generator + 70x70 PatchGAN discriminator
- Loss: conditional GAN (BCEWithLogits) + λ * L1 (paper Eq.1)
- Train/val splits only
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import if_main_process
from speechbrain.utils.logger import get_logger
from torchvision.utils import save_image

logger = get_logger(__name__)

# ----------------------------- Dataset -----------------------------
class MapsAlignedDataset(Dataset):
    """Loads horizontally-concatenated AB images (left=A, right=B)."""

    def __init__(self, root, split, direction="AtoB", load_size=286, crop_size=256, random_flip=True):
        super().__init__()
        self.root = Path(root) / split
        self.paths = sorted([p for p in self.root.glob("*.jpg")])
        if len(self.paths) == 0:
            logger.warning(f"No .jpg found in {self.root}")
            
        self.direction = direction
        self.load_size = load_size
        self.crop_size = crop_size
        self.random_flip = random_flip and (split == "train")

    def __len__(self):
        return len(self.paths)

    def _split_ab(self, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        w, h = img.size
        # assert w % 2 == 0, f"Image width {w} not even for {self.paths[0].name}"
        w2 = w // 2
        return img.crop((0, 0, w2, h)), img.crop((w2, 0, w, h))

    def _to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1)
        return ten * 2.0 - 1.0  # pix2pix: inputs in [-1,1], generator ends with tanh

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        A, B = self._split_ab(img)
        src, tgt = (A, B) if self.direction == "AtoB" else (B, A)

        # pix2pix: "resize to 286 then random crop to 256" = jitter augmentation
        if self.load_size != self.crop_size:
            src = src.resize((self.load_size, self.load_size), Image.BICUBIC)
            tgt = tgt.resize((self.load_size, self.load_size), Image.BICUBIC)

        # pix2pix: random crop (train) / center crop (eval)
        if self.crop_size < self.load_size:
            if self.random_flip:
                x = random.randint(0, self.load_size - self.crop_size)
                y = random.randint(0, self.load_size - self.crop_size)
            else:
                x = (self.load_size - self.crop_size) // 2
                y = (self.load_size - self.crop_size) // 2
            box = (x, y, x + self.crop_size, y + self.crop_size)
            src = src.crop(box)
            tgt = tgt.crop(box)

        # pix2pix: random mirroring (horizontal flip)
        if self.random_flip and random.random() < 0.5:
            src = src.transpose(Image.FLIP_LEFT_RIGHT)
            tgt = tgt.transpose(Image.FLIP_LEFT_RIGHT)

        src = self._to_tensor(src)
        tgt = self._to_tensor(tgt)
        return {"src": src, "tgt": tgt, "id": path.stem}


# ----------------------------- Models -----------------------------
def conv_block(in_c, out_c, norm=True, down=True):
    # pix2pix: conv -> BatchNorm -> (Leaky)ReLU module stack
    ks, st, pad = (4, 2, 1) if down else (4, 2, 1)
    layers = [nn.Conv2d(in_c, out_c, ks, st, pad, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    """U-Net generator from pix2pix (8 downs/ups with skip connections)."""
    def __init__(self, in_c=3, out_c=3, nf=64):
        super().__init__()
        # pix2pix: encoder (downsampling)
        self.e1 = nn.Conv2d(in_c, nf, 4, 2, 1)              # first block: no norm
        self.e2 = conv_block(nf, nf*2)
        self.e3 = conv_block(nf*2, nf*4)
        self.e4 = conv_block(nf*4, nf*8)
        self.e5 = conv_block(nf*8, nf*8)
        self.e6 = conv_block(nf*8, nf*8)
        self.e7 = conv_block(nf*8, nf*8)
        self.e8 = nn.Conv2d(nf*8, nf*8, 4, 2, 1)            # bottleneck no norm

        # pix2pix: decoder (upsampling) with dropout on first 3 (regularization)
        def up(in_c, out_c, drop=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU(True)]
            if drop:
                layers.append(nn.Dropout(0.5))              # paper uses dropout in decoder
            return nn.Sequential(*layers)

        self.d1 = up(nf*8, nf*8, drop=True)
        self.d2 = up(nf*16, nf*8, drop=True)
        self.d3 = up(nf*16, nf*8, drop=True)
        self.d4 = up(nf*16, nf*8)
        self.d5 = up(nf*16, nf*4)
        self.d6 = up(nf*8, nf*2)
        self.d7 = up(nf*4, nf)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, out_c, 4, 2, 1),
            nn.Tanh()                                       # pix2pix: tanh output to match [-1,1] scaling
        )

        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # pix2pix: U-Net skip connections via concatenation
        e1 = self.lrelu(self.e1(x))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8); d1 = torch.cat([d1, e7], dim=1)
        d2 = self.d2(d1); d2 = torch.cat([d2, e6], dim=1)
        d3 = self.d3(d2); d3 = torch.cat([d3, e5], dim=1)
        d4 = self.d4(d3); d4 = torch.cat([d4, e4], dim=1)
        d5 = self.d5(d4); d5 = torch.cat([d5, e3], dim=1)
        d6 = self.d6(d5); d6 = torch.cat([d6, e2], dim=1)
        d7 = self.d7(d6); d7 = torch.cat([d7, e1], dim=1)
        return self.d8(d7)

class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN.
    Modified to support Unconditional GAN (Standard GAN).
    """
    def __init__(self, in_c=3, nf=64, n_layers=3, conditional=True):
        super().__init__()
        self.conditional = conditional
        
        # Pix2Pix (Conditional): Input is Source(3) + Target(3) = 6 channels
        # Standard GAN (Unconditional): Input is Target(3) = 3 channels
        input_channels = in_c * 2 if self.conditional else in_c
        
        layers = [nn.Conv2d(input_channels, nf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = nf
        for _ in range(1, n_layers):
            layers += [nn.Conv2d(ch, ch*2, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(ch*2),
                       nn.LeakyReLU(0.2, True)]
            ch *= 2
        layers += [nn.Conv2d(ch, ch*2, 4, 1, 1, bias=False),
                   nn.BatchNorm2d(ch*2),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ch*2, 1, 4, 1, 1)] 
        self.net = nn.Sequential(*layers)

    def forward(self, src, tgt):
        # src: Input image (Satellite)
        # tgt: Real Map or Fake Map
        if self.conditional:
            # Condition on source image
            return self.net(torch.cat([src, tgt], dim=1))
        else:
            # Ignore source, just judge the map
            return self.net(tgt)


# ----------------------------- Brain -----------------------------
class Pix2PixBrain(sb.core.Brain):
    def _move_batch(self, batch):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def compute_forward(self, batch, stage):
        batch = self._move_batch(batch)
        src, tgt = batch["src"], batch["tgt"]
        
        fake = self.modules.G(src)
        
        # Discriminator handles whether to use 'src' or not internally based on flag
        d_real = self.modules.D(src, tgt)
        d_fake = self.modules.D(src, fake.detach())
        
        return fake, d_real, d_fake, src, tgt

    def compute_objectives(self, predictions, batch, stage):
        fake, d_real, d_fake, src, tgt = predictions
        valid = torch.ones_like(d_real)
        fake_lbl = torch.zeros_like(d_fake)

        # 1. Generator Loss
        # Fool the discriminator
        pred_fake = self.modules.D(src, fake)
        gan_loss_G = self.hparams.bce_with_logits(pred_fake, valid)
        
        # L1 reconstruction loss (always used to guide the translation)
        l1_loss = F.l1_loss(fake, tgt) * self.hparams.lambda_L1
        g_loss = gan_loss_G + l1_loss

        # 2. Discriminator Loss
        d_loss = 0.5 * (
            self.hparams.bce_with_logits(d_real, valid) +
            self.hparams.bce_with_logits(d_fake, fake_lbl)
        )

        self.gan_loss = gan_loss_G.detach()
        self.l1_loss = (l1_loss.detach() / self.hparams.lambda_L1)  # log raw L1 (unscaled)
        self.d_loss = d_loss.detach()
        return g_loss, d_loss, fake, src, tgt

    def fit_batch(self, batch):
        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        fake, d_real, d_fake, src, tgt = self.compute_forward(batch, sb.Stage.TRAIN)
        g_loss, d_loss, *_ = self.compute_objectives((fake, d_real, d_fake, src, tgt), batch, sb.Stage.TRAIN)

        # AMP for stability (implementation choice, not a paper requirement)
        self.scaler.scale(g_loss).backward(retain_graph=True)
        self.scaler.step(self.optimizer_G)

        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.optimizer_D)

        self.scaler.update()
        return (g_loss + d_loss).detach()

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            fake, d_real, d_fake, src, tgt = self.compute_forward(batch, stage)
            g_loss, d_loss, *_ = self.compute_objectives((fake, d_real, d_fake, src, tgt), batch, stage)
            return (g_loss + d_loss).detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        stats = {
            "loss": float(stage_loss),
            "G_GAN": float(self.gan_loss.mean().cpu().item()) if hasattr(self, "gan_loss") else None,
            "G_L1": float(self.l1_loss.mean().cpu().item()) if hasattr(self, "l1_loss") else None,
            "D": float(self.d_loss.mean().cpu().item()) if hasattr(self, "d_loss") else None,
        }
        if stage == sb.Stage.TRAIN:
            self.train_stats = stats
            # pix2pix: linear learning-rate decay after epoch 100 (paper schedule)
            self.scheduler_G.step()
            self.scheduler_D.step()
        elif stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=stats,
            )
            # keep best by lowest valid L1 (proxy for image fidelity)
            key = -stats["G_L1"] if stats["G_L1"] is not None else -stats["loss"]
            self.checkpointer.save_and_keep_only(meta={"neg_L1": key, "epoch": epoch}, min_keys=["neg_L1"])
            if if_main_process():
                self._save_samples(epoch)

    def _save_samples(self, epoch):
        out_dir = Path(self.hparams.samples_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        dl = self.sample_loaders["valid"]
        try:
            batch = next(iter(dl))
        except StopIteration:
            return
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            fake = self.modules.G(batch["src"])
        denorm = lambda x: (x + 1.0) * 0.5
        grid = torch.cat([denorm(batch["src"][:4]), denorm(fake[:4]), denorm(batch["tgt"][:4])], dim=0)
        save_image(grid, out_dir / f"samples_valid_epoch{epoch:04d}.png", nrow=4)

    @torch.no_grad()
    def on_fit_start(self):
        self.sample_loaders = {}
        if "valid" in self.hparams.sample_loader_builders:
            self.sample_loaders["valid"] = self.hparams.sample_loader_builders["valid"]()


# ----------------------------- DataIO -----------------------------
def build_dataloaders(hparams):
    def make(split, shuffle):
        ds = MapsAlignedDataset(
            root=hparams["data_folder"],
            split=split,
            direction=hparams["direction"],
            load_size=hparams["load_size"],
            crop_size=hparams["crop_size"],
            random_flip=(split == "train"),
        )
        return DataLoader(
            ds,
            batch_size=hparams["batch_size"] if split == "train" else 1,
            shuffle=shuffle,
            num_workers=hparams["num_workers"],
            pin_memory=True,
            drop_last=(split == "train"),
        )

    train_loader = make("train", shuffle=True)
    valid_loader = make("val", shuffle=False)
    # used only to dump validation sample triplets (src|fake|tgt)
    sample_loader_builders = {"valid": (lambda: make("val", shuffle=False))}
    return train_loader, valid_loader, sample_loader_builders

def linear_decay(epoch):
    # pix2pix: keep LR constant for 100 epochs, linearly decay to 0 over next 100
    return 1.0 if epoch < 100 else max(0.0, 1 - (epoch - 100) / 100)

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    torch.backends.cudnn.benchmark = True

    train_loader, valid_loader, sample_loader_builders = build_dataloaders(hparams)
    hparams["sample_loader_builders"] = sample_loader_builders
    
    # --- CHECK FOR CONDITIONAL FLAG ---
    # Default to True (Pix2Pix Standard) if not present in yaml
    is_conditional = hparams.get("conditional", True)

    G = UNetGenerator(in_c=3, out_c=3, nf=hparams["ngf"])
    
    # Pass flag to D
    D = PatchDiscriminator(
        in_c=3, 
        nf=hparams["ndf"], 
        n_layers=hparams["n_layers_D"], 
        conditional=is_conditional
    )
    
    modules = {"G": G, "D": D}
    model_list = nn.ModuleList([G, D])

    # pix2pix: Adam with β1=0.5, β2=0.999
    optimizer_G = torch.optim.Adam(G.parameters(), lr=hparams["lr"], betas=(hparams["beta1"], 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=hparams["lr"], betas=(hparams["beta1"], 0.999))

    # pix2pix: linear LR decay schedule
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=linear_decay)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=linear_decay)

    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=hparams["save_folder"],
        recoverables={
            "model": model_list,
            "opt_G": optimizer_G,
            "opt_D": optimizer_D,
            "counter": hparams["epoch_counter"],
        },
    )

    brain = Pix2PixBrain(
        modules=modules,
        opt_class=None,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    brain.optimizer_G = optimizer_G
    brain.optimizer_D = optimizer_D
    brain.scheduler_G = scheduler_G
    brain.scheduler_D = scheduler_D
    brain.scaler = torch.cuda.amp.GradScaler(enabled=(hparams.get("precision", "fp32") != "fp32"))

    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_loader,
        valid_set=valid_loader,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    )