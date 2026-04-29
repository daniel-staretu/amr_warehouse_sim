"""
VirConvNet local training script.

Adapted from virconvnet_seg.ipynb for local GPU training.
Optimised for 4 GB VRAM: batch size 1 and AMP by default.

Usage
-----
python virconvnet_train.py --data_root path/to/training_data
python virconvnet_train.py --data_root path/to/training_data --resume checkpoints/virconvnet/epoch_030.pt
"""

import argparse
import glob
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)


# ── Configuration ─────────────────────────────────────────────────────────────

class Config:
    # Paths — overridden by parse_args()
    DATA_ROOT = "training_data"

    # Classes
    NUM_CLASSES = 4
    CLASS_NAMES = ["background", "shelf", "crate", "forklift"]

    # Camera intrinsics (IMAGE_COLLECTOR 640×480, VFOV=90°)
    IMG_W, IMG_H = 640, 480
    FX = FY = 240.0
    CX, CY = 320.0, 240.0
    FEAT_W = IMG_W // 4   # 160
    FEAT_H = IMG_H // 4   # 120

    # Pillar geometry (sensor frame: +x fwd, +y left, +z up)
    X_MIN, X_MAX = -20.0, 20.0
    Y_MIN, Y_MAX = -20.0, 20.0
    Z_MIN, Z_MAX =  -0.5,  3.0
    VX,    VY    =  0.25,  0.25
    MAX_PTS_PER_PILLAR = 32
    MAX_PILLARS        = 8000
    BEV_H = int((X_MAX - X_MIN) / VX)   # 160
    BEV_W = int((Y_MAX - Y_MIN) / VY)   # 160

    # Image feature dimension
    IMG_FEAT_DIM = 32
    # 4 (lidar) + 3 (Δ centroid) + 2 (pillar ctr) + IMG_FEAT_DIM + 1 (is_virtual)
    PFN_IN_CH    = 4 + 3 + 2 + IMG_FEAT_DIM + 1   # 42
    PILLAR_FEAT  = 64

    # Virtual points
    MAX_VIRTUAL_PTS     = 1500
    VIRTUAL_DROPOUT     = 0.3
    MAX_DEPTH_SEARCH_PX = 12

    # Training — defaults tuned for 4 GB VRAM
    BATCH_SIZE    = 1
    NUM_EPOCHS    = 60
    LR            = 1e-3
    WEIGHT_DECAY  = 1e-4
    LR_MILESTONES = [30, 50]
    MAX_PTS       = 15_000
    AUX_LOSS_WEIGHT = 0.5
    NUM_WORKERS   = 4     # increase if loading is a bottleneck
    AMP           = True  # mixed-precision; set False if you hit NaN losses

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


def apply_args(args: argparse.Namespace) -> None:
    cfg.DATA_ROOT   = args.data_root
    cfg.BATCH_SIZE  = args.batch_size
    cfg.NUM_EPOCHS  = args.epochs
    cfg.NUM_WORKERS = args.workers
    cfg.AMP         = not args.no_amp


# ── Camera projection utilities ───────────────────────────────────────────────

def lidar_to_image(pts):
    """Project (N,4) LiDAR sensor-frame points to image pixel coords."""
    lx, ly, lz = pts[:, 0], pts[:, 1], pts[:, 2]
    depth    = lx
    in_front = depth > 0.1
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(in_front, cfg.FX * (-ly) / depth + cfg.CX, -1.0)
        v = np.where(in_front, cfg.FY * (-lz) / depth + cfg.CY, -1.0)
    valid = in_front & (u >= 0) & (u < cfg.IMG_W) & (v >= 0) & (v < cfg.IMG_H)
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32), valid


def sample_feat_at_pixels(feat_map_np, u_img, v_img):
    """Bilinear-sample (C, fH, fW) feature map at original-image pixel positions."""
    C, fH, fW = feat_map_np.shape
    u_f = u_img / 4.0
    v_f = v_img / 4.0
    u0  = np.clip(np.floor(u_f).astype(int), 0, fW - 1)
    v0  = np.clip(np.floor(v_f).astype(int), 0, fH - 1)
    u1  = np.clip(u0 + 1, 0, fW - 1)
    v1  = np.clip(v0 + 1, 0, fH - 1)
    wu  = (u_f - u0).astype(np.float32).clip(0, 1)
    wv  = (v_f - v0).astype(np.float32).clip(0, 1)
    out = (
        feat_map_np[:, v0, u0] * (1 - wu) * (1 - wv) +
        feat_map_np[:, v0, u1] * wu       * (1 - wv) +
        feat_map_np[:, v1, u0] * (1 - wu) * wv       +
        feat_map_np[:, v1, u1] * wu       * wv
    ).T
    return out


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    def __init__(self, split="train", augment=False):
        self.augment = augment and (split == "train")
        scan_dir = os.path.join(cfg.DATA_ROOT, "lidar",  split, "scans")
        lbl_dir  = os.path.join(cfg.DATA_ROOT, "lidar",  split, "labels")
        img_dir  = os.path.join(cfg.DATA_ROOT, "images", split, "images")
        msk_dir  = os.path.join(cfg.DATA_ROOT, "images", split, "masks")

        all_scans = sorted(glob.glob(os.path.join(scan_dir, "*.npy")))
        self.scans, self.labels, self.images, self.masks = [], [], [], []
        for sf in all_scans:
            stem  = os.path.splitext(os.path.basename(sf))[0]
            img_f = os.path.join(img_dir, stem + ".png")
            msk_f = os.path.join(msk_dir, stem + ".png")
            if os.path.exists(img_f) and os.path.exists(msk_f):
                self.scans.append(sf)
                self.labels.append(os.path.join(lbl_dir, stem + ".npy"))
                self.images.append(img_f)
                self.masks.append(msk_f)
        print(f"[{split}] {len(self.scans)} paired LiDAR+image frames")

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        pts = np.load(self.scans[idx]).astype(np.float32)
        lbl = np.load(self.labels[idx]).astype(np.int64)
        img = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        msk = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE).astype(np.int64)

        keep = (
            (pts[:, 0] >= cfg.X_MIN) & (pts[:, 0] < cfg.X_MAX) &
            (pts[:, 1] >= cfg.Y_MIN) & (pts[:, 1] < cfg.Y_MAX) &
            (pts[:, 2] >= cfg.Z_MIN) & (pts[:, 2] < cfg.Z_MAX)
        )
        pts, lbl = pts[keep], lbl[keep]

        if len(pts) > cfg.MAX_PTS:
            sel = np.random.choice(len(pts), cfg.MAX_PTS, replace=False)
            pts, lbl = pts[sel], lbl[sel]

        if self.augment and len(pts) > 0:
            angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            c, s  = np.cos(angle), np.sin(angle)
            xy    = pts[:, :2] @ np.array([[c, s], [-s, c]], np.float32)
            pts   = np.concatenate([xy, pts[:, 2:]], axis=1)

        img_t = torch.from_numpy(img.transpose(2, 0, 1))
        return pts, lbl, img_t, msk


def collate_mm(batch):
    pts_list = [b[0] for b in batch]
    lbl_list = [b[1] for b in batch]
    img_t    = torch.stack([b[2] for b in batch])
    msk_list = [b[3] for b in batch]
    return pts_list, lbl_list, img_t, msk_list


# ── Model components ──────────────────────────────────────────────────────────

class ImageBackbone(nn.Module):
    """Lightweight encoder: (B,3,H,W) → (B,32,H/4,W/4)."""
    def __init__(self, out_ch=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(64, out_ch, 1, bias=False)

    def forward(self, img):
        return self.proj(self.enc(img))


class VirPillarFeatureNet(nn.Module):
    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch, bias=False)
        self.bn  = nn.BatchNorm1d(out_ch)

    def forward(self, pillars):
        BP, K, C = pillars.shape
        x = self.lin(pillars.reshape(BP * K, C))
        x = F.relu(self.bn(x), inplace=True)
        return x.view(BP, K, -1).max(dim=1)[0]


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, depth=4):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BEVBackbone(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.b1  = _ConvBlock(in_ch, 64,  stride=1, depth=4)
        self.b2  = _ConvBlock(64,   128,  stride=2, depth=6)
        self.b3  = _ConvBlock(128,  256,  stride=2, depth=6)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.out_ch = 64 + 128 + 128

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        return torch.cat([f1, self.up2(f2), self.up3(f3)], dim=1)


class VirConvNetSeg(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.img_bb   = ImageBackbone(out_ch=cfg.IMG_FEAT_DIM)
        self.pfn      = VirPillarFeatureNet(in_ch=cfg.PFN_IN_CH, out_ch=cfg.PILLAR_FEAT)
        self.backbone = BEVBackbone(in_ch=cfg.PILLAR_FEAT)
        self.head     = nn.Sequential(
            nn.Conv2d(self.backbone.out_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1),
        )
        self.aux_head    = nn.Conv2d(cfg.IMG_FEAT_DIM, num_classes, 1)
        self.num_classes = num_classes

    def encode_images(self, imgs):
        return self.img_bb(imgs)

    def forward_aux(self, feat_map):
        return self.aux_head(feat_map)

    def forward_pillars(self, pillars, coords, n_pil):
        B, P, K, C = pillars.shape
        H, W = cfg.BEV_H, cfg.BEV_W
        pf  = self.pfn(pillars.view(B * P, K, C)).view(B, P, cfg.PILLAR_FEAT)
        bev = pillars.new_zeros(B, cfg.PILLAR_FEAT, H, W)
        for b in range(B):
            n  = int(n_pil[b])
            xi = coords[b, :n, 0].long()
            yi = coords[b, :n, 1].long()
            bev[b, :, xi, yi] = pf[b, :n].T
        feat = self.backbone(bev)
        return self.head(feat)


# ── Virtual point generation ──────────────────────────────────────────────────

def generate_virtual_points(pts, mask_np, feat_map_np, training=False):
    C, fH, fW = feat_map_np.shape
    if len(pts) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, C), np.float32)

    u_img, v_img, depth, valid = lidar_to_image(pts)
    depth_buf = np.zeros((fH, fW), np.float32)
    if valid.any():
        u_f = np.clip((u_img[valid] / 4).astype(int), 0, fW - 1)
        v_f = np.clip((v_img[valid] / 4).astype(int), 0, fH - 1)
        for ui, vi, d in zip(u_f, v_f, depth[valid]):
            if depth_buf[vi, ui] == 0 or d < depth_buf[vi, ui]:
                depth_buf[vi, ui] = d

    msk_small = mask_np[::4, ::4][:fH, :fW]
    fg_mask   = (msk_small > 0) & (depth_buf == 0)
    fg_rows, fg_cols = np.where(fg_mask)
    if len(fg_rows) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, C), np.float32)

    occ_rows, occ_cols = np.where(depth_buf > 0)
    if len(occ_rows) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, C), np.float32)

    tree        = cKDTree(np.stack([occ_rows, occ_cols], axis=1).astype(np.float32))
    dists, idxs = tree.query(np.stack([fg_rows, fg_cols], axis=1).astype(np.float32), k=1)
    near        = dists <= cfg.MAX_DEPTH_SEARCH_PX
    fg_rows     = fg_rows[near];  fg_cols  = fg_cols[near]
    ref_depth   = depth_buf[occ_rows[idxs[near]], occ_cols[idxs[near]]]

    if len(fg_rows) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, C), np.float32)

    if len(fg_rows) > cfg.MAX_VIRTUAL_PTS:
        sel = np.random.choice(len(fg_rows), cfg.MAX_VIRTUAL_PTS, replace=False)
        fg_rows, fg_cols, ref_depth = fg_rows[sel], fg_cols[sel], ref_depth[sel]

    if training and cfg.VIRTUAL_DROPOUT > 0:
        keep = np.random.rand(len(fg_rows)) > cfg.VIRTUAL_DROPOUT
        fg_rows, fg_cols, ref_depth = fg_rows[keep], fg_cols[keep], ref_depth[keep]

    if len(fg_rows) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, C), np.float32)

    u_virt = fg_cols * 4 + 1.5
    v_virt = fg_rows * 4 + 1.5
    cam_z  = ref_depth
    cam_x  = (u_virt - cfg.CX) * cam_z / cfg.FX
    cam_y  = (v_virt - cfg.CY) * cam_z / cfg.FY
    lx_v   = cam_z.astype(np.float32)
    ly_v   = (-cam_x).astype(np.float32)
    lz_v   = (-cam_y).astype(np.float32)

    virt_feat = sample_feat_at_pixels(feat_map_np,
                                       fg_cols * 4 + 1.5, fg_rows * 4 + 1.5)
    virt_pts  = np.stack([lx_v, ly_v, lz_v,
                           np.full(len(lx_v), 0.5, np.float32)], axis=1)
    return virt_pts, virt_feat


# ── Pillarization ─────────────────────────────────────────────────────────────

def augment_and_pillarize(pts_list, lbl_list, feat_maps_np, msk_list, training=False):
    B   = len(pts_list)
    P   = cfg.MAX_PILLARS
    K   = cfg.MAX_PTS_PER_PILLAR
    FC  = cfg.PFN_IN_CH
    C   = cfg.IMG_FEAT_DIM

    pillars = np.zeros((B, P, K, FC), np.float32)
    coords  = np.zeros((B, P, 2),    np.int32)
    n_pil   = np.zeros(B,            np.int32)
    pt2pil  = []

    for b in range(B):
        pts      = pts_list[b]
        feat_map = feat_maps_np[b]
        mask_np  = msk_list[b]
        N_real   = len(pts)

        img_feat_real = np.zeros((N_real, C), np.float32)
        if N_real > 0:
            u, v, _, vis = lidar_to_image(pts)
            if vis.any():
                img_feat_real[vis] = sample_feat_at_pixels(feat_map, u[vis], v[vis])

        virt_pts, virt_feat = generate_virtual_points(
            pts, mask_np, feat_map, training=training)
        N_virt = len(virt_pts)

        real_aug = np.concatenate(
            [pts, img_feat_real, np.zeros((N_real, 1), np.float32)], axis=1)
        if N_virt > 0:
            virt_aug = np.concatenate(
                [virt_pts, virt_feat, np.ones((N_virt, 1), np.float32)], axis=1)
            combined = np.concatenate([real_aug, virt_aug], axis=0)
        else:
            combined = real_aug

        total_pts = len(combined)
        if total_pts == 0:
            pt2pil.append(np.empty(0, np.int32))
            continue

        xi = np.clip(np.floor((combined[:, 0] - cfg.X_MIN) / cfg.VX).astype(np.int32),
                     0, cfg.BEV_H - 1)
        yi = np.clip(np.floor((combined[:, 1] - cfg.Y_MIN) / cfg.VY).astype(np.int32),
                     0, cfg.BEV_W - 1)
        key    = xi.astype(np.int64) * cfg.BEV_W + yi
        order  = np.argsort(key, kind="mergesort")
        key_s  = key[order]
        pts_s  = combined[order]
        bounds = np.where(np.diff(key_s, prepend=key_s[0] - 1) != 0)[0]
        ukeys  = key_s[bounds]
        n_keep = min(len(ukeys), P)

        p2p_real = np.full(N_real, -1, np.int32)
        for pi in range(n_keep):
            s         = bounds[pi]
            e         = bounds[pi + 1] if pi + 1 < len(bounds) else total_pts
            n         = min(e - s, K)
            orig_idx  = order[s: s + n]
            real_mask = orig_idx < N_real
            p2p_real[orig_idx[real_mask]] = pi

            pts_pi  = pts_s[s: s + n]
            mean    = pts_pi[:, :3].mean(0)
            xi_p    = int(ukeys[pi]) // cfg.BEV_W
            yi_p    = int(ukeys[pi]) %  cfg.BEV_W
            xp      = xi_p * cfg.VX + cfg.X_MIN + cfg.VX / 2
            yp      = yi_p * cfg.VY + cfg.Y_MIN + cfg.VY / 2

            feat = np.zeros((n, FC), np.float32)
            feat[:, :4]      = pts_pi[:, :4]
            feat[:, 4:7]     = pts_pi[:, :3] - mean
            feat[:, 7]       = xp
            feat[:, 8]       = yp
            feat[:, 9:9 + C] = pts_pi[:, 4:4 + C]
            feat[:, 9 + C]   = pts_pi[:, 4 + C]

            pillars[b, pi, :n] = feat
            coords[b, pi]      = [xi_p, yi_p]

        n_pil[b] = n_keep
        pt2pil.append(p2p_real)

    return (
        torch.from_numpy(pillars),
        torch.from_numpy(coords),
        n_pil,
        pt2pil,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def per_class_iou(pred_np, true_np):
    iou = []
    for c in range(cfg.NUM_CLASSES):
        tp = int(((pred_np == c) & (true_np == c)).sum())
        fp = int(((pred_np == c) & (true_np != c)).sum())
        fn = int(((pred_np != c) & (true_np == c)).sum())
        d  = tp + fp + fn
        iou.append(tp / d if d > 0 else float("nan"))
    return iou


def estimate_class_weights(dataset, n_sample=300):
    counts = np.zeros(cfg.NUM_CLASSES, np.int64)
    for i in np.random.choice(len(dataset), min(n_sample, len(dataset)), replace=False):
        _, lbl, _, _ = dataset[i]
        for c in range(cfg.NUM_CLASSES):
            counts[c] += int((lbl == c).sum())
    counts = np.maximum(counts, 1)
    w = 1.0 / counts.astype(np.float64)
    w = w / w.sum() * cfg.NUM_CLASSES
    return torch.tensor(w, dtype=torch.float32)


# ── Train / eval loops ────────────────────────────────────────────────────────

def gather_point_logits(logits_bev, coords_np, n_pil, pt2pil, lbl_list=None):
    all_logits, all_labels = [], []
    for b in range(logits_bev.shape[0]):
        p2p   = pt2pil[b]
        valid = p2p >= 0
        if not valid.any():
            continue
        xi = torch.from_numpy(coords_np[b, p2p[valid], 0].astype(np.int64)).to(logits_bev.device)
        yi = torch.from_numpy(coords_np[b, p2p[valid], 1].astype(np.int64)).to(logits_bev.device)
        all_logits.append(logits_bev[b, :, xi, yi].T)
        if lbl_list is not None:
            all_labels.append(
                torch.from_numpy(lbl_list[b][valid]).to(logits_bev.device))
    if not all_logits:
        return None, None
    return torch.cat(all_logits), (torch.cat(all_labels) if all_labels else None)


def train_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for pts_list, lbl_list, imgs, msk_list in loader:
        imgs = imgs.to(cfg.DEVICE)

        with autocast(enabled=cfg.AMP):
            feat_maps = model.encode_images(imgs)

            msk_batch = torch.stack([
                torch.from_numpy(
                    m[::4, ::4][:cfg.FEAT_H, :cfg.FEAT_W]
                     .astype(np.int64).clip(0, cfg.NUM_CLASSES - 1))
                for m in msk_list
            ]).to(cfg.DEVICE)
            aux_loss = criterion(model.forward_aux(feat_maps), msk_batch)

        feat_maps_np = feat_maps.detach().float().cpu().numpy()

        pil, coo, n_pil, p2p = augment_and_pillarize(
            pts_list, lbl_list, feat_maps_np, msk_list, training=True)
        pil = pil.to(cfg.DEVICE)
        coo = coo.to(cfg.DEVICE)

        optimizer.zero_grad()
        with autocast(enabled=cfg.AMP):
            logits_bev        = model.forward_pillars(pil, coo, n_pil)
            logits_pt, lbl_pt = gather_point_logits(
                logits_bev, coo.cpu().numpy(), n_pil, p2p, lbl_list)
            if logits_pt is None:
                continue
            loss = criterion(logits_pt, lbl_pt) + cfg.AUX_LOSS_WEIGHT * aux_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_pred, all_true = [], []
    for pts_list, lbl_list, imgs, msk_list in loader:
        imgs = imgs.to(cfg.DEVICE)
        with autocast(enabled=cfg.AMP):
            feat_maps    = model.encode_images(imgs)
        feat_maps_np = feat_maps.float().cpu().numpy()
        pil, coo, n_pil, p2p = augment_and_pillarize(
            pts_list, lbl_list, feat_maps_np, msk_list, training=False)
        pil = pil.to(cfg.DEVICE)
        coo = coo.to(cfg.DEVICE)
        with autocast(enabled=cfg.AMP):
            logits_bev        = model.forward_pillars(pil, coo, n_pil)
        logits_pt, lbl_pt = gather_point_logits(
            logits_bev.float(), coo.cpu().numpy(), n_pil, p2p, lbl_list)
        if logits_pt is None:
            continue
        all_pred.append(logits_pt.argmax(1).cpu().numpy())
        all_true.append(lbl_pt.cpu().numpy())
    if not all_pred:
        return [float("nan")] * cfg.NUM_CLASSES, float("nan")
    pred_np = np.concatenate(all_pred)
    true_np = np.concatenate(all_true)
    iou     = per_class_iou(pred_np, true_np)
    valid   = [v for v in iou if not math.isnan(v)]
    miou    = sum(valid) / len(valid) if valid else float("nan")
    return iou, miou


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VirConvNet local training")
    p.add_argument("--data_root", default="training_data",
                   help="Root folder containing lidar/ and images/ sub-dirs")
    p.add_argument("--ckpt_dir", default=None,
                   help="Checkpoint output dir (default: <data_root>/checkpoints/virconvnet)")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch_size", type=int,   default=1,
                   help="Batch size (default 1 for 4 GB VRAM)")
    p.add_argument("--workers",    type=int,   default=4,
                   help="DataLoader worker processes")
    p.add_argument("--no_amp",     action="store_true",
                   help="Disable automatic mixed precision")
    return p.parse_args()


def main():
    args = parse_args()
    apply_args(args)

    ckpt_dir = args.ckpt_dir or os.path.join(cfg.DATA_ROOT, "checkpoints", "virconvnet")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Device      : {cfg.DEVICE}")
    if cfg.DEVICE == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM        : {total_mem:.1f} GB")
    print(f"AMP         : {cfg.AMP}")
    print(f"Batch size  : {cfg.BATCH_SIZE}")
    print(f"Epochs      : {cfg.NUM_EPOCHS}")
    print(f"Checkpoints : {ckpt_dir}")
    print()

    train_ds = MultimodalDataset("train", augment=True)
    val_ds   = MultimodalDataset("val",   augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=True,  collate_fn=collate_mm,
                          num_workers=cfg.NUM_WORKERS,
                          pin_memory=(cfg.DEVICE == "cuda"),
                          persistent_workers=(cfg.NUM_WORKERS > 0))
    val_dl   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                          shuffle=False, collate_fn=collate_mm,
                          num_workers=cfg.NUM_WORKERS,
                          pin_memory=(cfg.DEVICE == "cuda"),
                          persistent_workers=(cfg.NUM_WORKERS > 0))
    print(f"Train batches: {len(train_dl)}   Val batches: {len(val_dl)}\n")

    print("Estimating class weights…")
    class_weights = estimate_class_weights(train_ds).to(cfg.DEVICE)
    print("Weights:", dict(zip(cfg.CLASS_NAMES, class_weights.cpu().numpy().round(3))), "\n")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model     = VirConvNetSeg(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.LR_MILESTONES, gamma=0.1)
    scaler    = GradScaler(enabled=cfg.AMP)

    start_epoch = 1
    best_miou   = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"Resumed from {args.resume}  (epoch {start_epoch - 1},"
              f" best mIoU so far {best_miou:.4f})\n")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    log_path = os.path.join(ckpt_dir, "train_log.csv")
    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,loss,miou," + ",".join(cfg.CLASS_NAMES) + "\n")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0   = time.time()
        loss = train_epoch(model, train_dl, optimizer, scaler, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        do_eval = (epoch % 5 == 0) or (epoch == 1) or (epoch == cfg.NUM_EPOCHS)
        if do_eval:
            iou_list, miou = evaluate(model, val_dl, criterion)
            iou_str = "  ".join(
                f"{cfg.CLASS_NAMES[c]}={v:.3f}" if not math.isnan(v)
                else f"{cfg.CLASS_NAMES[c]}=--"
                for c, v in enumerate(iou_list))
            print(f"Ep {epoch:3d}/{cfg.NUM_EPOCHS}  loss={loss:.4f}  "
                  f"mIoU={miou:.4f}  [{iou_str}]  ({elapsed:.0f}s)")

            with open(log_path, "a") as f:
                iou_csv = ",".join(
                    f"{v:.4f}" if not math.isnan(v) else "" for v in iou_list)
                miou_csv = f"{miou:.4f}" if not math.isnan(miou) else ""
                f.write(f"{epoch},{loss:.4f},{miou_csv},{iou_csv}\n")

            if not math.isnan(miou) and miou > best_miou:
                best_miou = miou
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "miou": miou,
                    "iou": iou_list,
                    "best_miou": best_miou,
                }, os.path.join(ckpt_dir, "best.pt"))
                print(f"  -> best checkpoint saved (mIoU={miou:.4f})")
        else:
            print(f"Ep {epoch:3d}/{cfg.NUM_EPOCHS}  loss={loss:.4f}  ({elapsed:.0f}s)")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_miou": best_miou,
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))
            print(f"  -> periodic checkpoint saved (epoch {epoch})")

    print(f"\nDone.  Best val mIoU = {best_miou:.4f}")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
