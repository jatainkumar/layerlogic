"""
LayerLogic -- FastAPI Server (Single Backend)
=============================================
Serves both the HTML dashboard and the REST API for inference.
No Streamlit required -- single `python src/api.py` to launch.

Endpoints:
    GET  /             -> HTML dashboard
    GET  /health       -> Health check
    POST /predict      -> Upload .wav + .jpg -> defect probability
    GET  /model-info   -> Model metadata + training summary
    GET  /sample-files -> List sample files for demo
    POST /predict-sample -> Inference on pre-loaded sample
    GET  /graphs/{name} -> Serve output graph PNGs

Usage:
    python src/api.py
    # or: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import io
import time
import base64
import numpy as np
import torch
import joblib
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import audio_to_melspectrogram, get_image_transform, SAMPLE_RATE, IMAGE_SIZE
from model import ResNetFeatureExtractor, get_model_info

import librosa
from PIL import Image
import soundfile as sf

# -- Configuration --

PROJ_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJ_ROOT / "outputs"
DATA_DIR = PROJ_ROOT / "data"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# -- FastAPI App --

app = FastAPI(
    title="LayerLogic",
    description="Real-time LPBF defect detection via multimodal deep learning",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# -- Global State --

state = {
    "acoustic_model": None,
    "optical_model": None,
    "pipeline": None,
    "device": None,
    "best_classifier": None,
    "loaded": False,
}


@app.on_event("startup")
async def load_models():
    """Load all model artifacts on server startup."""
    print("\n  [LayerLogic] Loading models...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["device"] = device

    # Load ResNet-18 feature extractors
    acoustic = ResNetFeatureExtractor(pretrained=True, freeze=True).to(device)
    optical = ResNetFeatureExtractor(pretrained=True, freeze=True).to(device)
    acoustic.eval()
    optical.eval()
    state["acoustic_model"] = acoustic
    state["optical_model"] = optical

    # Load trained classifier pipeline
    pipeline_path = OUTPUT_DIR / "best_pipeline.pkl"
    if pipeline_path.exists():
        state["pipeline"] = joblib.load(pipeline_path)
        print(f"    Loaded pipeline: {pipeline_path}")
    else:
        print(f"    Pipeline not found at {pipeline_path}")
        print(f"    Run 'python src/train.py' first.")

    # Load training summary for metadata
    summary_path = OUTPUT_DIR / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        state["best_classifier"] = summary.get("best_classifier", "Unknown")
    else:
        state["best_classifier"] = "Unknown"

    state["loaded"] = True
    print(f"    Device: {device}")
    print(f"    Best classifier: {state['best_classifier']}")
    print(f"    Dashboard: http://localhost:8000")
    print("  [LayerLogic] Ready.\n")


# -- HTML Dashboard --

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main HTML dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/graphs/{filename}")
async def serve_graph(filename: str):
    """Serve output graph PNGs."""
    graph_path = OUTPUT_DIR / filename
    if not graph_path.exists() or not filename.endswith(".png"):
        raise HTTPException(status_code=404, detail="Graph not found.")
    return FileResponse(str(graph_path), media_type="image/png")


# -- API Endpoints --

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if state["loaded"] else "loading",
        "model_loaded": state["loaded"],
        "device": str(state["device"]),
        "classifier": state["best_classifier"],
    }


@app.post("/predict")
async def predict(
    audio_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
):
    """
    Run multimodal inference on uploaded audio + image.

    Args:
        audio_file: .wav acoustic emission file
        image_file: .jpg/.png melt-pool image

    Returns:
        JSON with defect_probability, prediction, latency_ms
    """
    if not state["loaded"] or state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python src/train.py' first."
        )

    start_time = time.time()

    try:
        # -- Process Audio --
        audio_bytes = await audio_file.read()

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        mel_rgb = audio_to_melspectrogram(tmp_path, SAMPLE_RATE)
        mel_display_b64 = _mel_to_base64(mel_rgb)

        mel_pil = Image.fromarray(mel_rgb)
        transform = get_image_transform(augment=False)
        mel_tensor = transform(mel_pil).unsqueeze(0).to(state["device"])

        os.unlink(tmp_path)

        # -- Process Image --
        image_bytes = await image_file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(state["device"])

        # -- Feature Extraction --
        with torch.no_grad():
            acoustic_feats = state["acoustic_model"](mel_tensor)
            optical_feats = state["optical_model"](image_tensor)

        fused = torch.cat([acoustic_feats, optical_feats], dim=1)
        fused_np = fused.cpu().numpy()

        # -- Classification --
        prediction = int(state["pipeline"].predict(fused_np)[0])
        probability = float(state["pipeline"].predict_proba(fused_np)[0][1])

        latency_ms = (time.time() - start_time) * 1000

        return JSONResponse({
            "defect_probability": round(probability, 4),
            "prediction": "Defect" if prediction == 1 else "Healthy",
            "prediction_code": prediction,
            "classifier": state["best_classifier"],
            "latency_ms": round(latency_ms, 2),
            "mel_spectrogram_base64": mel_display_b64,
            "feature_dim": int(fused_np.shape[1]),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Return model architecture and training metadata."""
    info = get_model_info()

    info["device"] = str(state["device"])
    info["best_classifier"] = state["best_classifier"]
    info["model_loaded"] = state["loaded"]

    summary_path = OUTPUT_DIR / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            info["training_summary"] = json.load(f)

    return info


@app.get("/sample-files")
async def list_sample_files():
    """List available sample files for the live replay demo."""
    samples = {"healthy": [], "defect": []}

    for label in ["healthy", "defect"]:
        audio_dir = DATA_DIR / "audio" / label
        image_dir = DATA_DIR / "images" / label

        if audio_dir.exists() and image_dir.exists():
            audio_files = sorted([f.name for f in audio_dir.iterdir()
                                  if f.suffix == ".wav"])
            image_files = sorted([f.name for f in image_dir.iterdir()
                                  if f.suffix in ('.jpg', '.jpeg', '.png')])

            num_pairs = min(len(audio_files), len(image_files))
            samples[label] = [
                {"audio": audio_files[i], "image": image_files[i]}
                for i in range(min(num_pairs, 20))
            ]

    return samples


@app.post("/predict-sample")
async def predict_sample(label: str, index: int):
    """
    Run inference on a pre-loaded sample (for live replay mode).

    Args:
        label: "healthy" or "defect"
        index: sample index
    """
    if not state["loaded"] or state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    audio_dir = DATA_DIR / "audio" / label
    image_dir = DATA_DIR / "images" / label

    audio_files = sorted([f for f in audio_dir.iterdir() if f.suffix == ".wav"])
    image_files = sorted([f for f in image_dir.iterdir()
                          if f.suffix in ('.jpg', '.jpeg', '.png')])

    if index >= len(audio_files) or index >= len(image_files):
        raise HTTPException(status_code=404, detail="Sample index out of range.")

    start_time = time.time()

    # Process audio
    mel_rgb = audio_to_melspectrogram(str(audio_files[index]), SAMPLE_RATE)
    mel_display_b64 = _mel_to_base64(mel_rgb)

    mel_pil = Image.fromarray(mel_rgb)
    transform = get_image_transform(augment=False)
    mel_tensor = transform(mel_pil).unsqueeze(0).to(state["device"])

    # Process image
    image_pil = Image.open(image_files[index]).convert("RGB")

    img_buffer = io.BytesIO()
    image_pil_resized = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_pil_resized.save(img_buffer, format="JPEG", quality=90)
    image_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    image_tensor = transform(image_pil).unsqueeze(0).to(state["device"])

    # Feature extraction + classification
    with torch.no_grad():
        acoustic_feats = state["acoustic_model"](mel_tensor)
        optical_feats = state["optical_model"](image_tensor)

    fused = torch.cat([acoustic_feats, optical_feats], dim=1)
    fused_np = fused.cpu().numpy()

    prediction = int(state["pipeline"].predict(fused_np)[0])
    probability = float(state["pipeline"].predict_proba(fused_np)[0][1])

    latency_ms = (time.time() - start_time) * 1000

    return JSONResponse({
        "defect_probability": round(probability, 4),
        "prediction": "Defect" if prediction == 1 else "Healthy",
        "prediction_code": prediction,
        "ground_truth": label,
        "sample_index": index,
        "classifier": state["best_classifier"],
        "latency_ms": round(latency_ms, 2),
        "mel_spectrogram_base64": mel_display_b64,
        "image_base64": image_b64,
    })


# -- Helpers --

def _mel_to_base64(mel_rgb: np.ndarray) -> str:
    """Convert a mel-spectrogram numpy array to base64 PNG string."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(mel_rgb[:, :, 0], aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Time", fontsize=9, color="#ccc")
    ax.set_ylabel("Mel Frequency", fontsize=9, color="#ccc")
    ax.set_title("Mel-Spectrogram", fontsize=10, fontweight="bold", color="#e0e0e0")
    ax.tick_params(colors="#999")
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#0a0a0a")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


# -- Main --

if __name__ == "__main__":
    import uvicorn
    print("\n  LayerLogic -- Starting server...")
    print("  Dashboard: http://localhost:8000\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
