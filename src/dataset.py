"""
LayerLogic — PyTorch Dataset & Feature Extraction
==================================================
Multimodal dataset class for paired audio (mel-spectrogram) + image data.
Includes feature extraction utilities using frozen ResNet-18.

Usage:
    from dataset import LPBFMultimodalDataset, get_dataloaders
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import librosa
from PIL import Image
from pathlib import Path


# ─── Configuration ───────────────────────────────────────────────────────────

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / "data"
IMAGE_SIZE = 224
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# ImageNet normalization (used for pretrained ResNet-18)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── Image Transforms ───────────────────────────────────────────────────────

def get_image_transform(augment: bool = False):
    """Get image preprocessing transform for melt-pool images."""
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─── Audio → Mel-Spectrogram ────────────────────────────────────────────────

def audio_to_melspectrogram(wav_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Convert a .wav file to a Mel-spectrogram image (224×224, 3-channel).

    Pipeline:
        raw audio → librosa.feature.melspectrogram → power_to_db →
        normalize → resize → 3-channel (for ResNet compatibility)

    Returns:
        numpy array of shape (224, 224, 3), dtype uint8
    """
    # Load audio
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # Convert to dB scale
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 255]
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 255).astype(np.uint8)

    # Resize to 224×224
    mel_img = Image.fromarray(mel_norm)
    mel_img = mel_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    # Convert to 3-channel (repeat grayscale for ResNet compatibility)
    mel_arr = np.array(mel_img)
    mel_rgb = np.stack([mel_arr, mel_arr, mel_arr], axis=-1)

    return mel_rgb


def audio_to_tensor(wav_path: str, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Convert a .wav file to a normalized 3-channel tensor for ResNet."""
    mel_rgb = audio_to_melspectrogram(wav_path, sr)
    # Convert to PIL for transform pipeline
    mel_pil = Image.fromarray(mel_rgb)
    transform = get_image_transform(augment=False)
    return transform(mel_pil)


# ─── Multimodal Dataset ─────────────────────────────────────────────────────

class LPBFMultimodalDataset(Dataset):
    """
    PyTorch Dataset for paired multimodal LPBF data.

    Each sample returns:
        mel_spectrogram (Tensor): 3×224×224 mel-spectrogram from .wav
        image (Tensor):           3×224×224 melt-pool image
        label (int):              0 = healthy, 1 = defect
    """

    def __init__(self, data_dir: str = None, augment: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.augment = augment
        self.image_transform = get_image_transform(augment=augment)

        # Scan directories
        self.samples = []  # list of (audio_path, image_path, label)
        self._scan_data()

    def _scan_data(self):
        """Scan data directories and pair audio/image files by sorted index."""
        audio_dir = self.data_dir / "audio"
        image_dir = self.data_dir / "images"

        for label_idx, label_name in enumerate(["healthy", "defect"]):
            audio_subdir = audio_dir / label_name
            image_subdir = image_dir / label_name

            if not audio_subdir.exists() or not image_subdir.exists():
                print(f"  [!] Missing directory: {audio_subdir} or {image_subdir}")
                continue

            # Get sorted file lists
            audio_files = sorted([
                f for f in audio_subdir.iterdir()
                if f.suffix.lower() in ('.wav', '.flac', '.mp3')
            ])
            image_files = sorted([
                f for f in image_subdir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
            ])

            # Pair by index (take minimum of the two counts)
            num_pairs = min(len(audio_files), len(image_files))
            for i in range(num_pairs):
                self.samples.append((
                    str(audio_files[i]),
                    str(image_files[i]),
                    label_idx
                ))

        print(f"  Dataset: {len(self.samples)} paired samples "
              f"({sum(1 for _, _, l in self.samples if l == 0)} healthy, "
              f"{sum(1 for _, _, l in self.samples if l == 1)} defect)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        audio_path, image_path, label = self.samples[idx]

        # Audio → mel-spectrogram tensor
        mel_spec = audio_to_tensor(audio_path, SAMPLE_RATE)

        # Image → tensor
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        return mel_spec, image_tensor, label


# ─── DataLoader Factory ─────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str = None,
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Create train/val/test DataLoaders with stratified-ish splitting.

    Returns:
        (train_loader, val_loader, test_loader, dataset)
    """
    dataset = LPBFMultimodalDataset(data_dir=data_dir, augment=False)

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"  Split: {train_size} train / {val_size} val / {test_size} test")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, dataset


# ─── Feature Extraction Utility ─────────────────────────────────────────────

def extract_features_from_loader(
    acoustic_model: nn.Module,
    optical_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Extract concatenated feature vectors from both branches.

    Args:
        acoustic_model: Frozen ResNet-18 for spectrograms
        optical_model:  Frozen ResNet-18 for images
        dataloader:     DataLoader yielding (mel_spec, image, label)
        device:         torch device

    Returns:
        features (np.ndarray): shape (N, 1024) — 512 acoustic + 512 optical
        labels (np.ndarray):   shape (N,)
    """
    acoustic_model.eval()
    optical_model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (mel_specs, images, labels) in enumerate(dataloader):
            mel_specs = mel_specs.to(device)
            images = images.to(device)

            # Extract features from both branches
            acoustic_feats = acoustic_model(mel_specs)  # (B, 512)
            optical_feats = optical_model(images)        # (B, 512)

            # Concatenate
            fused = torch.cat([acoustic_feats, optical_feats], dim=1)  # (B, 1024)

            all_features.append(fused.cpu().numpy())
            all_labels.append(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                processed = min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
                print(f"    Extracted features: {processed}/{len(dataloader.dataset)}")

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


if __name__ == "__main__":
    # Quick test
    print("Testing LPBFMultimodalDataset...")
    dataset = LPBFMultimodalDataset()
    if len(dataset) > 0:
        mel, img, label = dataset[0]
        print(f"  Mel-spec shape: {mel.shape}")
        print(f"  Image shape:    {img.shape}")
        print(f"  Label:          {label}")
    else:
        print("  No data found. Run data_setup.py first.")
