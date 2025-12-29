import os
import argparse
import numpy as np
import scipy.io as sio
from scipy.special import softmax
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# External library for Vision Transformers
import timm


# ==========================================
# 1. Configuration
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="PIAS Training Script for AWA2")

    # Dataset paths - YOU MUST UPDATE THESE PATHS FOR YOUR ENVIRONMENT
    parser.add_argument('--dataset_root', default='/data/ZSL/dataset/Animals_with_Attributes2/JPEGImages/',
                        help='Path to images')
    parser.add_argument('--dataset_avg_root', default='/data/ZSL/dataset/Animals_with_Attributes2/AVGImages/',
                        help='Path to average/prototype images')
    parser.add_argument('--xlsa_root', default='/data/ZSL/dataset/xlsa17/data/AWA2/',
                        help='Path to XLSA17 split files (res101.mat and att_splits.mat)')

    # Hyperparameters
    parser.add_argument('--out_features', type=int, default=85, help='Dimension of attributes (85 for AWA2)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for classification loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for structure/similarity loss')
    parser.add_argument('--gamma', type=float, default=0.1, help='Calibrated stacking hyperparameter')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--model_name', default='vit_large_patch16_224_in21k', help='TIMM model name')

    # System
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', default='./checkpoint/AWA2', help='Directory to save checkpoints')

    return parser.parse_args()


# ==========================================
# 2. Utilities
# ==========================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        if np.sum(idx) > 0:
            acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
        else:
            acc_per_class[i] = 0
    return np.mean(acc_per_class)


# ==========================================
# 3. Data Loading
# ==========================================
class ZSLDataset:
    def __init__(self, args):
        self.root = args.dataset_root
        self.root_avg = args.dataset_avg_root

        # Load metadata
        # Expects res101.mat and att_splits.mat in the XLSA root directory
        res101_path = os.path.join(args.xlsa_root, 'res101.mat')
        att_splits_path = os.path.join(args.xlsa_root, 'att_splits.mat')

        if not os.path.exists(res101_path) or not os.path.exists(att_splits_path):
            raise FileNotFoundError(f"Metadata files not found in {args.xlsa_root}")

        data = sio.loadmat(res101_path)
        attrs_mat_data = sio.loadmat(att_splits_path)

        image_files = data['image_files']
        # Clean up file paths for AWA2
        image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])

        labels = data['labels'].astype(int).squeeze() - 1

        train_idx = attrs_mat_data['trainval_loc'].squeeze() - 1
        test_seen_idx = attrs_mat_data['test_seen_loc'].squeeze() - 1
        test_unseen_idx = attrs_mat_data['test_unseen_loc'].squeeze() - 1

        # Attributes matrix: Classes x Attributes
        self.attrs_mat = attrs_mat_data["att"].astype(np.float32).T

        # Training data
        self.train_files = image_files[train_idx]
        self.train_labels = labels[train_idx]
        self.uniq_train_labels, self.train_labels_based0, self.counts_train_labels = np.unique(
            self.train_labels, return_inverse=True, return_counts=True
        )

        # Test Seen data
        self.test_seen_files = image_files[test_seen_idx]
        self.test_seen_labels = labels[test_seen_idx]

        # Test Unseen data
        self.test_unseen_files = image_files[test_unseen_idx]
        self.test_unseen_labels = labels[test_unseen_idx]


class CustomImageLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root = root
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        # Handle potential path issues
        img_path = os.path.join(self.root, self.image_files[idx])
        img_pil = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img_pil)

        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)


def get_transforms(mode='Train'):
    if mode == 'Train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


# ==========================================
# 4. Model Architecture
# ==========================================
class ViT_ZSL(nn.Module):
    def __init__(self, model_name="vit_large_patch16_224_in21k", out_features=85, pretrained=True):
        super(ViT_ZSL, self).__init__()
        # Create model using timm
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        # Replace head with Identity to get raw features, then add custom projection
        self.vit.head = nn.Identity()
        self.mlp_g = nn.Linear(in_features=1024, out_features=out_features, bias=False)

    def forward(self, x):
        # ViT Forward pass logic specific to timm implementation details
        # bs * CHW --> bs * 768 * 196 --> bs * 196 * 768
        x = self.vit.patch_embed(x)

        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)

        # Handle distinction between models with/without distillation tokens
        if hasattr(self.vit, 'dist_token') and self.vit.dist_token is not None:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)

        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        # Extract global feature (CLS token is usually index 0)
        x_g = x[:, 0]
        feat_g = self.mlp_g(x_g)

        return x_g, feat_g


# ==========================================
# 5. Training and Evaluation Engines
# ==========================================
def get_reprs(model, data_loader, device):
    model.eval()
    reprs = []
    # No gradients needed for feature extraction
    with torch.no_grad():
        for _, (data, _) in enumerate(data_loader):
            data = data.to(device)
            # only take the global feature (x_g is the 1024-dim feature)
            x, feat = model(data)
            reprs.append(feat.cpu().data.numpy())

    reprs = np.concatenate(reprs, 0)
    return reprs


def evaluate(model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels, attrs_mat, gamma,
             device):
    # Extract Representations
    with torch.no_grad():
        seen_reprs = get_reprs(model, test_seen_loader, device)
        unseen_reprs = get_reprs(model, test_unseen_loader, device)

    # Get Unique Labels
    uniq_test_seen_labels = np.unique(test_seen_labels)
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    # --- ZSL (Zero-Shot Learning) ---
    zsl_unseen_sim = unseen_reprs @ attrs_mat[uniq_test_unseen_labels].T
    predict_labels = np.argmax(zsl_unseen_sim, axis=1)
    zsl_unseen_predict_labels = uniq_test_unseen_labels[predict_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    # --- GZSL (Generalized Zero-Shot Learning) ---
    # Calibrated stacking
    Cs_mat = np.zeros(attrs_mat.shape[0])
    Cs_mat[uniq_test_seen_labels] = gamma

    # Calculate Accuracy for Seen Classes
    gzsl_seen_sim = softmax(seen_reprs @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels = np.argmax(gzsl_seen_sim, axis=1)
    gzsl_seen_acc = compute_accuracy(gzsl_seen_predict_labels, test_seen_labels, uniq_test_seen_labels)

    # Calculate Accuracy for Unseen Classes
    gzsl_unseen_sim = softmax(unseen_reprs @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels = np.argmax(gzsl_unseen_sim, axis=1)
    gzsl_unseen_acc = compute_accuracy(gzsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    # Harmonic Mean
    if (gzsl_unseen_acc + gzsl_seen_acc) > 0:
        H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc)
    else:
        H = 0

    print(f'ZSL: averaged per-class accuracy: {zsl_unseen_acc * 100:.2f}')
    print(f'GZSL Seen: averaged per-class accuracy: {gzsl_seen_acc * 100:.2f}')
    print(f'GZSL Unseen: averaged per-class accuracy: {gzsl_unseen_acc * 100:.2f}')
    print(f'GZSL: harmonic mean (H): {H * 100:.2f}')

    return zsl_unseen_acc, gzsl_seen_acc, gzsl_unseen_acc, H


def train_loop(model, train_loader, train_attrbs, test_loaders, attrs_mat, avg_img, args):
    test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels = test_loaders

    # Optimizer Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)

    # Metrics Tracking
    best_metrics = {
        "H": 0,
        "zsl_unseen": 0,
        "gzsl_seen": 0,
        "gzsl_unseen": 0
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Starting training on {args.device}...")

    for i in range(args.epoch):
        print(f"Epoch {i + 1}/{args.epoch}...")
        loss_meter = AverageMeter()
        model.train()

        # Progress bar
        tk = tqdm(train_loader, total=len(train_loader), desc="Training")

        for batch_idx, (data, label) in enumerate(tk):
            data = data.to(args.device)
            label = label.to(args.device)

            # Retrieve batch-specific attributes and average images
            bs_avg_img = avg_img[label].to(args.device)  # bs * 3 * 224 * 224
            bs_att = train_attrbs[label].to(args.device)  # bs * 85

            optimizer.zero_grad()

            # Forward pass: Real Images
            x_g, feat_g = model(data)  # bs * 1024

            # 1. Classification Loss (Cross Entropy)
            logit_g = feat_g @ train_attrbs.T.to(args.device)  # bs * 100
            loss_cls = args.alpha * nn.CrossEntropyLoss()(logit_g, label)

            # 2. Structure Loss & Similarity
            # Forward pass: Average/Prototype Images
            avg_x_g, avg_feat_g = model(bs_avg_img)

            # Attribute Correlation Matrix
            att_matrix = bs_att @ feat_g.T  # bs * bs
            att_matrix_m = F.softmax(att_matrix.squeeze(), dim=1)

            # Feature Correlation Matrix
            fea_matrix = avg_x_g @ x_g.T  # bs * bs
            fea_matrix_m = F.softmax(fea_matrix.squeeze(), dim=1)

            # MSE between correlation matrices
            struct_loss = args.beta * nn.MSELoss()(att_matrix_m, fea_matrix_m)

            # Cosine similarity loss
            simi_loss = args.beta * torch.mean(F.cosine_similarity(bs_att, avg_feat_g, dim=-1))

            total_loss = loss_cls + struct_loss + simi_loss

            total_loss.backward()
            optimizer.step()

            loss_meter.update(total_loss.item(), label.shape[0])
            tk.set_postfix({"loss": loss_meter.avg})

        print(f'Train: Average loss: {loss_meter.avg:.4f}')
        lr_scheduler.step()

        # Save Checkpoint (every 10 epochs or specific interval if needed)
        if (i + 1) % 10 == 0:
            print(' .... Saving model ...')
            save_file = os.path.join(args.save_path, f'Epoch_{i}.pt')
            torch.save(model.state_dict(), save_file)

        # Evaluation
        zsl_unseen, gzsl_seen, gzsl_unseen, H = evaluate(
            model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels,
            attrs_mat, args.gamma, args.device
        )

        # Update Bests
        if zsl_unseen > best_metrics["zsl_unseen"]:
            best_metrics["zsl_unseen"] = zsl_unseen
        if H > best_metrics["H"]:
            best_metrics["H"] = H
            best_metrics["gzsl_seen"] = gzsl_seen
            best_metrics["gzsl_unseen"] = gzsl_unseen

        print('=================================================')
        print(f'Best ZSL Unseen: {best_metrics["zsl_unseen"] * 100:.2f}')
        print(f'Best GZSL Harmonic Mean (H): {best_metrics["H"] * 100:.2f}')
        print('=================================================\n')


# ==========================================
# 6. Main Execution
# ==========================================
def main():
    args = get_args()

    # Initialize Dataset
    print(f"Loading Metadata for AWA2...")
    dataset = ZSLDataset(args)
    attrs_mat = dataset.attrs_mat

    # Prepare Weights for WeightedRandomSampler (to handle class imbalance)
    weights_ = 1. / dataset.counts_train_labels
    weights = weights_[dataset.train_labels_based0]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=dataset.train_labels_based0.shape[0], replacement=True
    )

    # Initialize DataLoaders
    train_data = CustomImageLoader(
        dataset.root, dataset.train_files, dataset.train_labels_based0,
        transform=get_transforms('Train')
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, drop_last=True
    )

    test_seen_data = CustomImageLoader(
        dataset.root, dataset.test_seen_files, dataset.test_seen_labels,
        transform=get_transforms('Test')
    )
    test_seen_loader = DataLoader(
        test_seen_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    test_unseen_data = CustomImageLoader(
        dataset.root, dataset.test_unseen_files, dataset.test_unseen_labels,
        transform=get_transforms('Test')
    )
    test_unseen_loader = DataLoader(
        test_unseen_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    # Load Average Images (Prototypes) into Memory
    print("Loading Average Images...")
    avg_img = torch.zeros(attrs_mat.shape[0], 3, 224, 224).float()
    avg_transform = get_transforms('Train')

    # We assume there is one avg image per class ID
    for i in range(attrs_mat.shape[0]):
        # Depending on file naming convention, might need adjustment
        p = os.path.join(dataset.root_avg, f'{i}.jpg')
        if os.path.exists(p):
            avg_img_pil = Image.open(p).convert("RGB")
            avg_img[i] = avg_transform(avg_img_pil)
        else:
            # Fallback or warning if avg image missing
            pass

            # Prepare Training Attributes
    train_attrbs = attrs_mat[dataset.uniq_train_labels]
    train_attrbs_tensor = torch.from_numpy(train_attrbs).float().to(args.device)

    # Initialize Model
    print(f"Initializing ViT Model: {args.model_name}")
    model = ViT_ZSL(model_name=args.model_name, out_features=args.out_features).to(args.device)

    # Bundle loaders for cleaner signature
    test_loaders = (test_seen_loader, dataset.test_seen_labels, test_unseen_loader, dataset.test_unseen_labels)

    # Start Training
    train_loop(model, train_loader, train_attrbs_tensor, test_loaders, attrs_mat, avg_img, args)


if __name__ == '__main__':
    main()