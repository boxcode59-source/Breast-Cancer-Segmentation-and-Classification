
"""
MS-SBOC: Squeeze-Excitation Dilated Multi-Branch Residual Network +
GVFESCM + Multi-Kernel RBF SVM Ensemble

Template implementation generated from methodology description.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk


class MammogramPreprocessor:
    def preprocess(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        th = threshold_otsu(image)
        mask = image > th
        mask = opening(mask, disk(3))

        image = image * mask.astype(np.uint8)

        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        image = ((image - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)

        image = cv2.resize(image, (256, 256))
        return image


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r),
            nn.ReLU(),
            nn.Linear(ch // r, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiBranchResidual(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.b1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)
        self.b2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4)

        self.se = SEBlock(ch * 3)

        self.fuse = nn.Conv2d(ch * 3, ch, 1)

    def forward(self, x):
        a = self.b1(x)
        b = self.b2(x)
        c = self.b3(x)

        z = torch.cat([a, b, c], dim=1)
        z = self.se(z)
        z = self.fuse(z)

        return x + z


class SDMRN(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.r1 = MultiBranchResidual(32)
        self.r2 = MultiBranchResidual(32)
        self.r3 = MultiBranchResidual(32)

        self.mask_head = nn.Conv2d(32, 1, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cls_head = nn.Linear(32, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)

        mask = torch.sigmoid(self.mask_head(x))

        feat = self.pool(x).flatten(1)

        cls = self.cls_head(feat)

        return mask, feat, cls


def dice_loss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    inter = (pred * target).sum()
    return 1 - ((2 * inter + 1) / (pred.sum() + target.sum() + 1))


class GVFESCM:
    def refine(self, image, coarse_mask):
        mask = (coarse_mask > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        return refined


def extract_handcrafted_features(mask, image):
    area = np.sum(mask)

    edges = cv2.Canny(image, 50, 150)
    edge_strength = np.mean(edges)

    texture = np.std(image)

    return np.array([area, edge_strength, texture])


class MRBSVE:
    def __init__(self):
        self.kernels = [
            SVC(kernel='rbf', gamma=0.1, probability=True),
            SVC(kernel='rbf', gamma=1.0, probability=True),
            SVC(kernel='rbf', gamma=10.0, probability=True)
        ]
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)

        for svm in self.kernels:
            svm.fit(X, y)

    def predict(self, X):
        X = self.scaler.transform(X)

        probs = []

        for svm in self.kernels:
            probs.append(svm.predict_proba(X)[:, 1])

        probs = np.mean(probs, axis=0)

        return (probs > 0.5).astype(int)


def train_sdmrn(model, loader, epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()

        for img, mask, label in loader:

            img = img.to(device)
            mask = mask.to(device)
            label = label.to(device)

            pred_mask, _, pred_cls = model(img)

            loss = dice_loss(pred_mask, mask) + ce(pred_cls, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}/{epochs} Loss={loss.item():.4f}")


if __name__ == "__main__":
    print("MS-SBOC Template Implementation Loaded")
