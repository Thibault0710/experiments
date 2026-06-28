from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from data import TinyImageNetMeshDataset, CIFAR10MeshDataset
from model_advanced import MeshColor, MeshColorAdaLn
from utils import lpips_loss, build_class_grid_batch, class_grid_wandb_image, EMA
import wandb
import torch.amp as amp
import random
from accelerate import Accelerator
from tqdm import tqdm
from conv_dit import ModelConv
from itertools import islice
from torchmetrics.image.fid  import FrechetInceptionDistance

PIXEL_STD  = torch.tensor([0.244, 0.237, 0.273]).mul(4).sqrt()  # CIFAR-10 per-channel "2·std" in [-1,1]
PIXEL_STD  = torch.tensor([1,1,1])
DEVICE     = 'cuda'
PATH       = "/home/thibault/experiments/data/cifar-10-batches-py"
T5_DIR     = "/home/thibault/experiments/data/cifar-10-batches-py/t5_embeddings"
IMAGE_SIZE = 32
LAMBDA_CLS = 0.03

def load_class_names(root=PATH) :
    return list(CIFAR10MeshDataset.CLASS_NAMES)

def step(model, b, device=DEVICE, cfg_drop=0.0, lambda_cls=0, gen=None) :
    B        = len(b.label)
    cfg_mask = torch.rand(B, device=device, generator=gen) < cfg_drop
    ts       = (torch.rand(B, device=device, generator=gen))
    eps      = torch.randn(b.y.shape, device=b.y.device, generator=gen) * PIXEL_STD.to(b.y.device)
    t_node   = ts[b.batch][..., None]
    xt       = (1-t_node)*b.y + t_node*eps
    b.x      = torch.cat([xt, b.pos], dim=-1)
    if isinstance(model ,MeshColor) :
        v_pred   = model(b, b.cond.masked_fill(cfg_mask[:, None, None], 0), ts, b.cond_mask)
    else :
        v_pred   = model(b, b.label.masked_fill(cfg_mask, 10), ts)
    v_gt     = eps - b.y
    mse      = torch.nn.functional.mse_loss(v_gt, v_pred)
    lpips    = torch.nn.functional.mse_loss(v_gt, v_gt) #lpips_loss(xt - t_node*v_pred, b, ts, image_size=IMAGE_SIZE)
    cls      = lambda_cls*torch.nn.functional.mse_loss(v_pred, v_gt.reshape(B, -1, 3)[torch.randperm(B)].flatten(0,1))
    
    #lpips    = lpips_loss(xt - t_node*v_pred, b, ts, image_size=IMAGE_SIZE)

    return mse + lpips - cls, mse, lpips, cls

def make_lr_schedule(warmup_steps) :
    def lr_lambda(step) :
        if step < warmup_steps :
            return step/warmup_steps
        return 1

    return lr_lambda

def inference(model, b, n_steps=20, t=1.0, eps=None, device=DEVICE) :
    B = len(b.label)
    if eps is None :
        eps    = torch.randn_like(b.y) * PIXEL_STD.to(b.y.device)

    with torch.no_grad() :
        xt        = (1-t)*b.y + t*eps
        ts        = torch.linspace(0, t, n_steps+1, device=device).unsqueeze(0).expand(B, -1)  # (B, n_steps+1)
        if n_steps <= 0 :
            return xt
        size_step = ts[0, -1] - ts[0, -2]
        for step in range(n_steps) :
            b.x    = torch.cat([xt, b.pos], dim=-1)
            if isinstance(model, MeshColor) :
                v_pred = model(b, b.cond, ts[:, -step-1], b.cond_mask)
            else : # asssume adaln0 vrsion
                v_pred = model(b, b.label, ts[:, -step-1])
            xt     = xt - size_step*v_pred
    return xt

def inference_cfg(model, b, n_steps=20, t=1.0, cfg=1.0, eps=None, device=DEVICE) : # default is no cfg
    B = len(b.label)
    if eps is None :
        eps    = torch.randn_like(b.y) * PIXEL_STD.to(b.y.device)

    with torch.no_grad() :
        xt        = (1-t)*b.y + t*eps
        ts        = torch.linspace(0, t, n_steps+1, device=device).unsqueeze(0).expand(B, -1)  # (B, n_steps+1)
        size_step = ts[0, -1] - ts[0, -2]
        for step in range(n_steps) :
            b.x         = torch.cat([xt, b.pos], dim=-1)
            if isinstance(model, MeshColor) :
                v_pred_cond = model(b, b.cond, ts[:, -step-1], b.cond_mask)
            else :
                v_pred_cond = model(b, b.label, ts[:, -step-1])
            
            cond = torch.zeros_like(b.cond)
            if isinstance(model, MeshColor) :
                v_pred_uncond = model(b, cond, ts[:, -step-1], b.cond_mask)
            else :
                v_pred_uncond = model(b, torch.ones_like(b.label)*(10), ts[:, -step-1])

            v_pred = v_pred_uncond + cfg*(v_pred_cond - v_pred_uncond)
            xt     = xt - size_step*v_pred
    return xt

def train_loop(model, train_dataloader, val_dataloader, epochs, lr, bs, grad_acc=1, cfg_drop=0.0, warump=500,
               device='cuda', val_log=1, save_ckpt=3, ckpt_dir="outputs/ckpt/pers_cifar",
               resume_ckpt=None, model_cfg=None, ema_viz: bool = False, disable_wandb=True, val_steps=100, compute_fid=40) :
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00)
    total_steps = 0
    start_epoch = 0
    ema = EMA(model, decay=0.9999)  # half-life ~7k steps (diffusion-standard)
    if resume_ckpt is not None :
        optimizer.load_state_dict(resume_ckpt["opt"])
        start_epoch = resume_ckpt["epoch"] + 1
        total_steps = resume_ckpt["step"]
        if "ema" in resume_ckpt:
            ema.load_state_dict(resume_ckpt["ema"])
            print("resumed EMA from ckpt")
        else:
            print("no EMA in ckpt — initializing EMA from current weights (cold start)")
    model_cfg = dict(model_cfg or {})  # safe default
    n_params  = sum(p.numel() for p in model.parameters())
    wandb_cfg = dict(
        epochs=epochs, lr=lr, bs=bs, grad_acc=grad_acc, cfg_drop=cfg_drop, warmup=warump,
        n_params=n_params, model_cls=type(model).__name__,
        **model_cfg,  # hidden_dim, n_heads, n_blocs, pe, expansion_factor, n_pe_freqs at top level
    )
    wandb.init(project="colorMesh", config=wandb_cfg, mode='online' if not disable_wandb else "disabled", id='rzbl03cf', resume='must'
               )
    ckpt_dir    = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # accelerator part
    accelerator=Accelerator(gradient_accumulation_steps=grad_acc)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,make_lr_schedule(warmup_steps=warump)
    )

    class_names = load_class_names()

    for epoch in range(start_epoch, epochs) :
        optimizer.zero_grad()  # discard leftover accumulated grads from previous epoch boundary
        accum_loss = accum_mse = accum_lpips = accum_cls = torch.tensor(0.0, device=device)
        accum_n = 0
        for b in tqdm(train_dataloader) :
            with accelerator.accumulate(model) :
                with torch.autocast(device_type=device, dtype=torch.bfloat16) :
                    loss, mse_loss, lpips_loss, cls_loss = step(model, b.to(device), cfg_drop=cfg_drop, lambda_cls=LAMBDA_CLS, device=device)
                accelerator.backward(loss)

                accum_loss  += loss.detach()
                accum_mse   += mse_loss.detach()
                accum_lpips += lpips_loss.detach()
                accum_cls   += cls_loss.detach()
                accum_n     += 1

                # gradient clipping if necessary; returns the pre-clip total L2 norm (the value compared to max_norm)
                if accelerator.sync_gradients :
                    total_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                total_steps += 1

                if accelerator.sync_gradients :
                    ema.update(model)
                    log_dict = {
                        "train_loss": accum_loss.item()  / accum_n,
                        "mse_loss":   accum_mse.item()   / accum_n,
                        "lpips_loss": accum_lpips.item() / accum_n,
                        "cls_loss": accum_cls.item()     / accum_n,
                        "epoch":      epoch,
                    }
                    if total_norm is not None :
                        log_dict["grad_norm"] = total_norm.item()
                    wandb.log(log_dict)
                    accum_loss = accum_mse = accum_lpips = accum_cls = 0.0
                    accum_n = 0
                    scheduler.step()
                    optimizer.zero_grad()

        if epoch % val_log == 0 :
            model.eval()
            with torch.no_grad() :
                rng         = torch.Generator(device=device).manual_seed(0)
                live_losses = [step(model, b.to(device), gen=rng)[0].item() for b in islice(val_dataloader, val_steps)]
                # Swap in EMA weights for val_loss_ema (always computed as a cheap diagnostic).
                ema_backup  = ema.apply_to(model)
                rng_ema     = torch.Generator(device=device).manual_seed(0)
                ema_losses  = [step(model, b.to(device), gen=rng_ema)[0].item() for b in islice(val_dataloader, val_steps)]
                # ema_viz=True keeps EMA loaded for the val_grid; False restores live weights first
                # (useful early in training when EMA is still close to init and viz looks broken).
                if not ema_viz :
                    ema.restore(model, ema_backup)
                grid_b      = build_class_grid_batch(val_dataloader.dataset, n_per_class=4, n_classes=len(class_names), device=device)
                wandb.log({
                    "val_loss":     sum(live_losses)/len(live_losses),
                    "val_loss_ema": sum(ema_losses)/len(ema_losses),
                    "val_grid_t10": class_grid_wandb_image(grid_b, inference(model, grid_b, n_steps=20, t=1.0), class_names, 4, IMAGE_SIZE),
                    "val_grid_cfg_3": class_grid_wandb_image(grid_b, inference_cfg(model, grid_b, n_steps=20, t=1.0, cfg=3), class_names, 4, IMAGE_SIZE),
                    "val_grid_cfg_5": class_grid_wandb_image(grid_b, inference_cfg(model, grid_b, n_steps=20, t=1.0, cfg=5), class_names, 4, IMAGE_SIZE),
                    "epoch":        epoch,
                })

                if ema_viz :
                    ema.restore(model, ema_backup)
            model.train()
        
        if epoch % compute_fid == 0 :
            model.eval()
            print("FID computation:")
            with torch.no_grad():
                fid = FrechetInceptionDistance(feature=2048).to(device)
                # Real images
                for x in val_dataloader:
                    fid.update(((1+x.y.reshape(bs, 32,32, 3).cuda())/2*255.0).to(torch.uint8).permute(0, 3, 1, 2), real=True)

                # Fake images
                for x in val_dataloader:
                    x_fake = inference_cfg(model, x.to(device), n_steps=10, t=1, cfg=4)
                    fid.update(((1+x_fake)/2*255.0).to(torch.uint8).reshape(bs, 32, 32, 3).permute(0,3,1,2), real=False)

                score = fid.compute()
                print('FID = ', score)
                wandb.log({'fid': score})
                del fid
            model.train()



        if epoch % save_ckpt == 0 :
            for old in ckpt_dir.glob("epoch_*.pt") :
                old.unlink()
            torch.save({"model": model.state_dict(),
                        "ema":   ema.state_dict(),
                        "opt":   optimizer.state_dict(),
                        "epoch": epoch,
                        "step":  total_steps,
                        "model_cls": type(model).__name__,
                        "model_cfg": model_cfg},
                    ckpt_dir / f"epoch_{epoch:02d}.pt")
        
def get_dataloader(path, t5_dir, bs=8, n_vert=2000, workers=8, mesh_type="grid", patch_size=32, same_grid=False, prefetch_factor=4, shuffle=True) :
    kw = dict(min_vertices=n_vert, max_vertices=n_vert, pe_type="fourier", mesh_type=mesh_type, grid_resolution=IMAGE_SIZE, patch_size=patch_size, same_grid=same_grid)
    tr = CIFAR10MeshDataset(path, t5_emb_dir=t5_dir, split="train", seed=0, **kw)
    va = CIFAR10MeshDataset(path, t5_emb_dir=t5_dir, split="val",   seed=1, **kw)
    dl_kw = dict(batch_size=bs, num_workers=workers, prefetch_factor=prefetch_factor if workers > 0 else None, drop_last=True)
    return (DataLoader(tr, shuffle=shuffle, **dl_kw),
            DataLoader(va, shuffle=False,   **dl_kw))

from unet import UNetModel
from conv_dit_32 import ModelConv32
from gat_dit_32 import ModelSAGE32
from PGNN import PGNN
import argparse

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_wandb', action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    torch.set_float32_matmul_precision("high")

    model_cfg  = dict(hidden_dim=128, sdim=512, n_blocs=4, n_pe_freqs=8, pe="none", n_heads=8, expansion_factor=4.0, res_connexions=False, n_sage_blocs_enc=4, n_sage_blocs_dec=8, patch_size=32, n_conv_blocs=4)
    #model      = MeshColorAdaLn(**model_cfg).to(DEVICE)
    #model = ModelConv32(**model_cfg).to(DEVICE)
    model = PGNN(**model_cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {type(model).__name__}  params: {n_params:,}")
    #model = torch.compile(model)
    # model = UNetModel(
    #     in_channels=3,
    #     model_channels=128,
    #     num_res_blocks=2,
    #     attention_resolutions=(16, 8),
    #     channel_mult=(1, 2, 2, 2),
    #     num_classes=11,
    #     num_heads=4,
    #     use_scale_shift_norm=True,
    # ).to(DEVICE)

    # Resume from a previous checkpoint: load model weights here; train_loop will
    # load the optimizer state (and EMA if present) via resume_ckpt.
    ckpt_path = "/home/thibault/experiments/colorMesh/outputs/ckpt/pers_cifar/epoch_42.pt"
    ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"resumed model weights from {ckpt_path}  (epoch={ckpt['epoch']}, step={ckpt['step']})")
    ckpt['ema'] = {key: ckpt['model'][key] for key in ckpt['ema'].keys()}

    lr=3e-4
    bs=32
    epochs=1000
    grad_acc=4
    warm_up=8000/grad_acc

    train_dataloader, val_dataloader = get_dataloader(PATH, T5_DIR, bs=bs, mesh_type='grid', patch_size=model_cfg['patch_size'], same_grid=True)
    try:
        train_loop(model, train_dataloader, val_dataloader, epochs=epochs, lr=lr, bs=bs, grad_acc=grad_acc, cfg_drop=0.1, warump=warm_up,
                   model_cfg=model_cfg, disable_wandb=args.disable_wandb, resume_ckpt=ckpt, ema_viz=True)
    finally:
        wandb.finish()

if __name__ == "__main__" :
    main()
