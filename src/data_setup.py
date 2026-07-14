"""
LayerLogic — Data Acquisition & Synthetic Fallback
===================================================
Downloads real melt-pool images from Zenodo (EOS M290, 316L steel)
and generates physics-informed synthetic acoustic .wav files matched
to the image labels.

Fallback: if Zenodo download fails, generates synthetic images too.

Usage:
    python src/data_setup.py
"""

import os
import sys
import zipfile

# Fix Windows encoding for Unicode output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import shutil
import random
import math
import requests
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
IMAGE_DIR = DATA_DIR / "images"

# Zenodo dataset: "Annotated Image Dataset for defects detection in LPBF"
ZENODO_RECORD_ID = "14996805"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Synthetic data parameters
SAMPLE_RATE = 22050          # Hz — librosa default
AUDIO_DURATION = 1.0         # seconds per sample
NUM_SAMPLES_PER_CLASS = 250  # target count per class
IMAGE_SIZE = (224, 224)      # pixels
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── Zenodo Image Download ──────────────────────────────────────────────────

def download_zenodo_images(max_per_class: int = 250) -> bool:
    """
    Attempt to download and extract real LPBF melt-pool images from Zenodo.
    Returns True on success, False on failure (triggering synthetic fallback).
    """
    print("\n" + "=" * 60)
    print("  STEP 1a: Downloading Real Melt-Pool Images from Zenodo")
    print("=" * 60)
    print(f"  Record ID: {ZENODO_RECORD_ID}")
    print(f"  Source: EOS M290 machine, 316L stainless steel")

    try:
        # Query Zenodo API for file list
        print("\n  [1/4] Querying Zenodo API...")
        resp = requests.get(ZENODO_API_URL, timeout=30)
        resp.raise_for_status()
        record = resp.json()

        # Find the zip file containing preprocessed images
        files = record.get("files", [])
        target_file = None
        for f in files:
            fname = f.get("key", "").lower()
            if "preprocessed" in fname or "image" in fname or fname.endswith(".zip"):
                target_file = f
                break

        # If no preprocessed file, try the archive endpoint
        if target_file is None:
            print("  [!] No preprocessed zip found. Trying files-archive...")
            download_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files-archive"
        else:
            download_url = target_file.get("links", {}).get("self", "")
            print(f"  [2/4] Found: {target_file.get('key', 'unknown')} "
                  f"({target_file.get('size', 0) / 1e6:.1f} MB)")

        if not download_url:
            print("  [!] Could not resolve download URL.")
            return False

        # Download the file
        print(f"  [3/4] Downloading from Zenodo...")
        zip_path = DATA_DIR / "zenodo_download.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(download_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r    Progress: {pct:.1f}% ({downloaded / 1e6:.1f} MB)", end="")
            print()

        # Extract and organize
        print("  [4/4] Extracting and organizing images...")
        extract_dir = DATA_DIR / "zenodo_extracted"
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Find image files and sort into healthy/defect
        healthy_dir = IMAGE_DIR / "healthy"
        defect_dir = IMAGE_DIR / "defect"
        healthy_dir.mkdir(parents=True, exist_ok=True)
        defect_dir.mkdir(parents=True, exist_ok=True)

        healthy_count = 0
        defect_count = 0

        for root, dirs, filenames in os.walk(extract_dir):
            root_lower = root.lower()
            for fname in filenames:
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    continue

                src_path = Path(root) / fname

                # Classify by folder name
                if any(kw in root_lower for kw in ['good', 'healthy', 'normal', 'no_defect']):
                    if healthy_count < max_per_class:
                        dst = healthy_dir / f"zenodo_healthy_{healthy_count:04d}.jpg"
                        _convert_and_save(src_path, dst)
                        healthy_count += 1
                elif any(kw in root_lower for kw in ['defect', 'bad', 'anomal', 'flaw']):
                    if defect_count < max_per_class:
                        dst = defect_dir / f"zenodo_defect_{defect_count:04d}.jpg"
                        _convert_and_save(src_path, dst)
                        defect_count += 1

        # If the folder structure didn't have clear labels, try a flat split
        if healthy_count == 0 and defect_count == 0:
            print("  [!] Could not determine labels from folder structure.")
            print("      Splitting images 50/50 as healthy/defect for PoC purposes.")
            all_images = []
            for root, dirs, filenames in os.walk(extract_dir):
                for fname in filenames:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        all_images.append(Path(root) / fname)

            random.shuffle(all_images)
            half = min(len(all_images) // 2, max_per_class)
            for i, src_path in enumerate(all_images[:half]):
                dst = healthy_dir / f"zenodo_healthy_{i:04d}.jpg"
                _convert_and_save(src_path, dst)
                healthy_count += 1
            for i, src_path in enumerate(all_images[half:half * 2]):
                dst = defect_dir / f"zenodo_defect_{i:04d}.jpg"
                _convert_and_save(src_path, dst)
                defect_count += 1

        # Cleanup
        if zip_path.exists():
            os.remove(zip_path)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

        if healthy_count > 0 and defect_count > 0:
            print(f"\n  ✓ Downloaded: {healthy_count} healthy + {defect_count} defect images")
            return True
        else:
            print("  [!] No usable images found in download.")
            return False

    except Exception as e:
        print(f"\n  [!] Zenodo download failed: {e}")
        print("      Falling back to synthetic image generation.")
        # Cleanup partial downloads
        for p in [DATA_DIR / "zenodo_download.zip", DATA_DIR / "zenodo_extracted"]:
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    os.remove(p)
        return False


def _convert_and_save(src: Path, dst: Path):
    """Load an image, resize to 224×224, and save as JPEG."""
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        img.save(dst, "JPEG", quality=95)
    except Exception:
        pass


# ─── Synthetic Image Generation (Fallback) ──────────────────────────────────

def generate_synthetic_images(num_per_class: int = NUM_SAMPLES_PER_CLASS):
    """
    Generate synthetic melt-pool thermal images using OpenCV/PIL.
    Healthy: clean elliptical bright melt pool on dark background.
    Defect: irregular shape, keyhole depression, scattered spatter.
    """
    print("\n" + "=" * 60)
    print("  STEP 1b: Generating Synthetic Melt-Pool Images")
    print("=" * 60)

    healthy_dir = IMAGE_DIR / "healthy"
    defect_dir = IMAGE_DIR / "defect"
    healthy_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_per_class):
        # ── Healthy melt-pool image ──
        img = _create_meltpool_image(defect=False)
        img.save(healthy_dir / f"synth_healthy_{i:04d}.jpg", "JPEG", quality=95)

        # ── Defect melt-pool image ──
        img = _create_meltpool_image(defect=True)
        img.save(defect_dir / f"synth_defect_{i:04d}.jpg", "JPEG", quality=95)

        if (i + 1) % 50 == 0:
            print(f"    Generated {i + 1}/{num_per_class} image pairs...")

    print(f"  ✓ Generated {num_per_class} healthy + {num_per_class} defect images")


def _create_meltpool_image(defect: bool) -> Image.Image:
    """
    Create a single synthetic melt-pool thermal image.
    Returns a 224×224 RGB PIL Image.
    """
    w, h = IMAGE_SIZE
    img = Image.new("RGB", (w, h), color=(10, 10, 15))
    draw = ImageDraw.Draw(img)

    # Background thermal gradient (subtle) — vectorized for speed
    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx - w // 2) / (w // 2)
    dy = (yy - h // 2) / (h // 2)
    r = np.sqrt(dx * dx + dy * dy)
    intensity = np.clip(25 * (1 - r) + np.random.normal(0, 3, (h, w)), 0, 255).astype(np.uint8)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :, 0] = intensity
    bg[:, :, 1] = intensity
    bg[:, :, 2] = np.clip(intensity.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    cx, cy = w // 2 + random.randint(-15, 15), h // 2 + random.randint(-15, 15)

    if not defect:
        # ── HEALTHY: clean, symmetric elliptical melt pool ──
        rx = random.randint(35, 55)
        ry = random.randint(25, 40)

        # Outer glow (warm orange)
        for s in range(3, 0, -1):
            alpha = int(60 / s)
            draw.ellipse(
                [cx - rx - s * 8, cy - ry - s * 8,
                 cx + rx + s * 8, cy + ry + s * 8],
                fill=(alpha, alpha // 2, 0)
            )

        # Core melt pool (bright white-yellow)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                     fill=(255, 240, 200))

        # Inner hot spot
        draw.ellipse([cx - rx // 2, cy - ry // 2, cx + rx // 2, cy + ry // 2],
                     fill=(255, 255, 255))

        # Minimal spatter (1-3 tiny dots)
        for _ in range(random.randint(1, 3)):
            sx = cx + random.randint(-70, 70)
            sy = cy + random.randint(-70, 70)
            sr = random.randint(1, 2)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr],
                         fill=(200, 180, 100))

    else:
        # ── DEFECT: irregular, asymmetric, keyhole features ──
        rx = random.randint(30, 60)
        ry = random.randint(20, 50)
        # Asymmetric stretch
        rx2 = rx + random.randint(10, 25)
        ry2 = ry - random.randint(5, 15)

        # Irregular outer glow
        for s in range(4, 0, -1):
            alpha = int(80 / s)
            draw.ellipse(
                [cx - rx2 - s * 10, cy - ry - s * 6,
                 cx + rx - s * 3, cy + ry2 + s * 10],
                fill=(alpha, alpha // 3, 0)
            )

        # Distorted melt pool (elongated, off-center)
        draw.ellipse([cx - rx2, cy - ry, cx + rx, cy + ry2],
                     fill=(240, 200, 140))

        # Keyhole depression (dark spot in center)
        kx = cx + random.randint(-10, 5)
        ky = cy + random.randint(-8, 5)
        kr = random.randint(8, 18)
        draw.ellipse([kx - kr, ky - kr, kx + kr, ky + kr],
                     fill=(40, 20, 10))

        # Heavy spatter (8-20 particles scattered widely)
        for _ in range(random.randint(8, 20)):
            sx = cx + random.randint(-100, 100)
            sy = cy + random.randint(-100, 100)
            sr = random.randint(1, 4)
            brightness = random.randint(150, 255)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr],
                         fill=(brightness, brightness - 30, brightness // 2))

        # Surface roughness streaks
        for _ in range(random.randint(2, 5)):
            x1 = cx + random.randint(-60, 60)
            y1 = cy + random.randint(-40, 40)
            x2 = x1 + random.randint(-30, 30)
            y2 = y1 + random.randint(-20, 20)
            draw.line([x1, y1, x2, y2], fill=(120, 100, 60), width=2)

    # Apply slight Gaussian blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Add sensor noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# ─── Synthetic Audio Generation ─────────────────────────────────────────────

def generate_synthetic_audio(num_per_class: int = NUM_SAMPLES_PER_CLASS):
    """
    Generate physics-informed synthetic acoustic emission .wav files.

    Healthy: Broadband pink noise with stable spectral content.
    Defect:  Pink noise + transient bursts (spatter clicks, keyhole rumble,
             amplitude instability).
    """
    print("\n" + "=" * 60)
    print("  STEP 2: Generating Physics-Informed Synthetic Audio")
    print("=" * 60)
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Duration: {AUDIO_DURATION}s per sample")

    healthy_dir = AUDIO_DIR / "healthy"
    defect_dir = AUDIO_DIR / "defect"
    healthy_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(SAMPLE_RATE * AUDIO_DURATION)

    for i in range(num_per_class):
        # ── Healthy acoustic emission ──
        audio = _generate_healthy_audio(num_samples)
        sf.write(healthy_dir / f"ae_healthy_{i:04d}.wav", audio, SAMPLE_RATE)

        # ── Defect acoustic emission ──
        audio = _generate_defect_audio(num_samples)
        sf.write(defect_dir / f"ae_defect_{i:04d}.wav", audio, SAMPLE_RATE)

        if (i + 1) % 50 == 0:
            print(f"    Generated {i + 1}/{num_per_class} audio pairs...")

    print(f"  ✓ Generated {num_per_class} healthy + {num_per_class} defect .wav files")


def _pink_noise(n: int) -> np.ndarray:
    """Generate pink (1/f) noise via spectral shaping."""
    white = np.random.randn(n)
    # FFT approach
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    freqs[0] = 1  # avoid division by zero
    # 1/f weighting
    fft *= 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n)
    # Normalize
    pink = pink / (np.max(np.abs(pink)) + 1e-8)
    return pink


def _generate_healthy_audio(n: int) -> np.ndarray:
    """
    Healthy melt-pool: stable broadband pink noise.
    Now includes natural process variations (small clicks, minor amplitude fluctuations)
    so it's not trivially separable from defect audio.
    """
    # Base pink noise
    audio = _pink_noise(n) * 0.3

    # Natural melt-pool oscillation
    t = np.linspace(0, AUDIO_DURATION, n)
    mod_freq = random.uniform(5, 20)
    mod_depth = random.uniform(0.05, 0.15)
    audio *= (1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t))

    # Add occasional small clicks (normal spatter)
    num_clicks = random.randint(0, 3)
    for _ in range(num_clicks):
        click_idx = random.randint(0, n - int(0.01 * SAMPLE_RATE))
        click_len = int(random.uniform(0.002, 0.008) * SAMPLE_RATE)
        audio[click_idx:click_idx+click_len] += np.random.randn(click_len) * random.uniform(0.1, 0.2)

    # General process noise
    audio += np.random.randn(n) * 0.05

    # Normalize safely
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
    return audio.astype(np.float32)


def _generate_defect_audio(n: int) -> np.ndarray:
    """
    Defect melt-pool: pink noise + transient anomalies.
    Anomalies are more subtle to overlap with the healthy distribution,
    preventing the model from getting 1.0 F1 trivially.
    """
    t = np.linspace(0, AUDIO_DURATION, n)
    audio = _pink_noise(n) * random.uniform(0.25, 0.35)

    # 1. Spatter ejection clicks (slightly more/louder than healthy)
    num_clicks = random.randint(2, 6)
    for _ in range(num_clicks):
        click_time = random.uniform(0.05, AUDIO_DURATION - 0.05)
        click_idx = int(click_time * SAMPLE_RATE)
        click_duration = int(random.uniform(0.005, 0.015) * SAMPLE_RATE)
        click_freq = random.uniform(4000, 9000)
        click_amp = random.uniform(0.15, 0.4)  # Reduced from 0.8

        end_idx = min(click_idx + click_duration, n)
        click_t = np.arange(end_idx - click_idx) / SAMPLE_RATE

        envelope = np.exp(-click_t * random.uniform(100, 300))
        click_signal = click_amp * envelope * np.sin(2 * np.pi * click_freq * click_t)
        audio[click_idx:end_idx] += click_signal

    # 2. Keyhole collapse rumble (subtle)
    if random.random() > 0.3:
        rumble_freq = random.uniform(300, 600)
        rumble_amp = random.uniform(0.05, 0.15)  # Reduced
        rumble = rumble_amp * np.sin(2 * np.pi * rumble_freq * t)
        
        start_frac = random.uniform(0.1, 0.6)
        end_frac = start_frac + random.uniform(0.1, 0.3)
        mask = np.zeros(n)
        mask[int(start_frac * n):int(min(end_frac, 1.0) * n)] = 1.0
        audio += rumble * mask

    # 3. Amplitude spikes (subtle instability)
    num_spikes = random.randint(1, 4)
    for _ in range(num_spikes):
        spike_idx = random.randint(0, n - 200)
        spike_width = random.randint(50, 200)
        spike_amp = random.uniform(1.2, 2.0)  # Reduced from 5.0
        end_idx = min(spike_idx + spike_width, n)
        audio[spike_idx:end_idx] *= spike_amp

    # Normalize safely
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
    return audio.astype(np.float32)


# ─── Data Summary ────────────────────────────────────────────────────────────

def print_data_summary():
    """Print a summary of all generated/downloaded data."""
    print("\n" + "=" * 60)
    print("  DATA SUMMARY")
    print("=" * 60)

    for modality, base_dir in [("Images", IMAGE_DIR), ("Audio", AUDIO_DIR)]:
        print(f"\n  {modality}:")
        for label in ["healthy", "defect"]:
            d = base_dir / label
            if d.exists():
                files = list(d.iterdir())
                count = len(files)
                ext = files[0].suffix if files else "N/A"
                print(f"    {label:>10s}: {count:4d} files ({ext})")
            else:
                print(f"    {label:>10s}: 0 files (directory missing)")

    total_imgs = sum(
        len(list((IMAGE_DIR / l).iterdir()))
        for l in ["healthy", "defect"]
        if (IMAGE_DIR / l).exists()
    )
    total_audio = sum(
        len(list((AUDIO_DIR / l).iterdir()))
        for l in ["healthy", "defect"]
        if (AUDIO_DIR / l).exists()
    )
    print(f"\n  Total: {total_imgs} images + {total_audio} audio files")
    print(f"  Location: {DATA_DIR}")
    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "+" + "=" * 58 + "+")
    print("|" + "  LayerLogic - Data Acquisition Pipeline".center(58) + "|")
    print("|" + "  Multimodal LPBF Defect Detection".center(58) + "|")
    print("+" + "=" * 58 + "+")

    # Parse CLI args
    use_synthetic = "--synthetic" in sys.argv or "-s" in sys.argv

    # Step 1: Try Zenodo download for images; fall back to synthetic
    images_exist = (
        (IMAGE_DIR / "healthy").exists() and
        len(list((IMAGE_DIR / "healthy").iterdir())) >= 10 and
        (IMAGE_DIR / "defect").exists() and
        len(list((IMAGE_DIR / "defect").iterdir())) >= 10
    )

    if images_exist:
        print("\n  [OK] Image data already exists. Skipping download.")
    elif use_synthetic:
        print("\n  [--synthetic] Skipping Zenodo download, generating synthetic images...")
        generate_synthetic_images(NUM_SAMPLES_PER_CLASS)
    else:
        success = download_zenodo_images(max_per_class=NUM_SAMPLES_PER_CLASS)
        if not success:
            print("\n  [->] Falling back to synthetic image generation...")
            generate_synthetic_images(NUM_SAMPLES_PER_CLASS)

    # Step 2: Generate synthetic audio (always, since no public paired dataset)
    audio_exist = (
        (AUDIO_DIR / "healthy").exists() and
        len(list((AUDIO_DIR / "healthy").iterdir())) >= 10 and
        (AUDIO_DIR / "defect").exists() and
        len(list((AUDIO_DIR / "defect").iterdir())) >= 10
    )

    if audio_exist:
        print("\n  [OK] Audio data already exists. Skipping generation.")
    else:
        generate_synthetic_audio(NUM_SAMPLES_PER_CLASS)

    # Summary
    print_data_summary()

    print("\n  Done! Data pipeline complete. Ready for training.")
    print("    Next: python src/train.py\n")


if __name__ == "__main__":
    main()
