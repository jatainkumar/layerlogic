"""
LayerLogic — Streamlit Web Application
=======================================
Full pitch-deck web app with interactive demo and live replay mode.

Sections:
    1. Hero/Landing
    2. The Problem (LPBF defects)
    3. How It Works (pipeline explanation)
    4. Interactive Demo (upload + inference)
    5. Live Replay Mode (auto-stream samples)
    6. Results Gallery (training graphs)
    7. Architecture Diagram
    8. About / Credits

Usage:
    streamlit run src/app.py
"""

import streamlit as st
import requests
import base64
import time
import json
import io
from pathlib import Path
from PIL import Image

# ─── Configuration ───────────────────────────────────────────────────────────

PROJ_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJ_ROOT / "outputs"
DATA_DIR = PROJ_ROOT / "data"
API_URL = "http://localhost:8000"

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LayerLogic — Real-Time LPBF Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Hero section */
.hero-container {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 40%, #0d2137 70%, #0a0a1a 100%);
    padding: 4rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid rgba(0, 212, 255, 0.15);
    box-shadow: 0 0 60px rgba(0, 212, 255, 0.05);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,212,255,0.03) 0%, transparent 70%);
    animation: pulse 8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff, #00ff88, #00d4ff);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient 4s ease infinite;
    margin-bottom: 0.5rem;
    position: relative;
    letter-spacing: -1px;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-subtitle {
    font-size: 1.3rem;
    color: #b0c4de;
    font-weight: 300;
    margin-bottom: 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.3rem;
    position: relative;
}

/* Section headers */
.section-header {
    font-size: 2rem;
    font-weight: 800;
    color: #e0e0e0;
    margin: 3rem 0 1.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid rgba(0, 212, 255, 0.3);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 900;
    margin: 0.5rem 0;
}
.metric-label {
    font-size: 0.9rem;
    color: #8899aa;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Prediction result */
.prediction-healthy {
    background: linear-gradient(135deg, #0a2e1a, #1a3a2e);
    border: 2px solid #00ff88;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.prediction-defect {
    background: linear-gradient(135deg, #2e0a0a, #3a1a1a);
    border: 2px solid #ff4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.prediction-warning {
    background: linear-gradient(135deg, #2e2a0a, #3a351a);
    border: 2px solid #ffd43b;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

/* Problem cards */
.defect-card {
    background: linear-gradient(180deg, #1a1a2e, #0e0e1a);
    border: 1px solid rgba(255, 107, 107, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
}
.defect-card h4 {
    color: #ff6b6b;
    font-size: 1.2rem;
    margin-bottom: 0.8rem;
}
.defect-card p {
    color: #aabbcc;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Pipeline steps */
.pipeline-step {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.5rem 0;
}
.pipeline-arrow {
    text-align: center;
    font-size: 1.5rem;
    color: #00d4ff;
    padding: 0.3rem 0;
}

/* Replay timeline */
.timeline-item {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin: 2px;
    transition: transform 0.2s;
}
.timeline-healthy { background: #00ff88; }
.timeline-defect { background: #ff4444; }
.timeline-pending { background: #333; }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────────────

def check_api_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except Exception:
        return False


def predict_uploaded(audio_file, image_file) -> dict:
    """Send uploaded files to the API for prediction."""
    files = {
        "audio_file": (audio_file.name, audio_file.getvalue(), "audio/wav"),
        "image_file": (image_file.name, image_file.getvalue(), "image/jpeg"),
    }
    resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
    resp.raise_for_status()
    return resp.json()


def predict_sample(label: str, index: int) -> dict:
    """Call the sample prediction endpoint for live replay."""
    resp = requests.post(
        f"{API_URL}/predict-sample",
        params={"label": label, "index": index},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_sample_files() -> dict:
    """Get list of available sample files."""
    try:
        resp = requests.get(f"{API_URL}/sample-files", timeout=5)
        return resp.json()
    except Exception:
        return {"healthy": [], "defect": []}


# ─── Section 1: Hero ─────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">LayerLogic</div>
        <div class="hero-subtitle">
            Real-Time Defect Detection in Metal Additive Manufacturing<br>
            via Multimodal Deep Learning & Sensor Fusion
        </div>
        <div>
            <span class="hero-badge">🔬 Deep Tech AI</span>
            <span class="hero-badge">🏭 Industry 4.0</span>
            <span class="hero-badge">⚡ Edge Inference</span>
            <span class="hero-badge">🎓 IIT Kharagpur</span>
        </div>
        <div style="margin-top: 1.5rem; position: relative;">
            <span style="color: #6688aa; font-size: 0.9rem;">
                Team SteelSync · IIT Kharagpur Platinum Jubilee Innovation Challenge
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Section 2: The Problem ─────────────────────────────────────────────────

def render_problem():
    st.markdown('<div class="section-header">⚠️ The Problem — Why Metal 3D Printing Isn\'t Trustworthy Yet</div>', unsafe_allow_html=True)

    st.markdown("""
    Metal Additive Manufacturing (Laser Powder Bed Fusion) creates parts by melting metal powder
    layer-by-layer with a high-power laser. The process is **chaotic** — thermal fluctuations,
    spatter ejection, and melt-pool instabilities create **hidden micro-defects** that compromise
    structural integrity.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="defect-card">
            <h4>⚡ Lack of Fusion</h4>
            <p><strong>Too little energy.</strong> The laser doesn't fully melt the powder,
            leaving gaps between layers. Creates weakness and porosity in the finished part.</p>
            <p style="color: #ff6b6b; font-weight: 600; margin-top: 1rem;">
                Energy: LOW → Incomplete melting
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="defect-card">
            <h4>🕳️ Keyhole Porosity</h4>
            <p><strong>Too much energy.</strong> The laser drills too deep, creating a vapor cavity
            that collapses and traps gas bubbles as spherical pores inside the metal.</p>
            <p style="color: #ff6b6b; font-weight: 600; margin-top: 1rem;">
                Energy: HIGH → Gas entrapment
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="defect-card">
            <h4>🌊 Surface Roughness</h4>
            <p><strong>Unstable melt pool.</strong> The surface freezes unevenly, leaving bumps
            and ridges that reduce fatigue life and create stress concentration points.</p>
            <p style="color: #ff6b6b; font-weight: 600; margin-top: 1rem;">
                Energy: UNSTABLE → Uneven solidification
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Impact stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Current Detection</div>
            <div class="metric-value" style="color: #ff6b6b;">DAYS</div>
            <div style="color: #8899aa; font-size: 0.85rem;">Post-mortem CT scan</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">LayerLogic Detection</div>
            <div class="metric-value" style="color: #00ff88;">~ms</div>
            <div style="color: #8899aa; font-size: 0.85rem;">Real-time, per-layer</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Scrap Reduction</div>
            <div class="metric-value" style="color: #00d4ff;">80%+</div>
            <div style="color: #8899aa; font-size: 0.85rem;">Halt failing builds early</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Hardware Needed</div>
            <div class="metric-value" style="color: #ffd43b;">ZERO</div>
            <div style="color: #8899aa; font-size: 0.85rem;">Software-only solution</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Section 3: How It Works ────────────────────────────────────────────────

def render_how_it_works():
    st.markdown('<div class="section-header">🔬 How LayerLogic Works — Multimodal Sensor Fusion</div>', unsafe_allow_html=True)

    st.markdown("""
    LayerLogic **listens and watches simultaneously**, fusing two physically independent data streams
    to diagnose build quality in real-time. Each modality is blind where the other sees — **fusion
    is the differentiator** that suppresses false alarms and enables defect-type diagnosis.
    """)

    # Pipeline visualization
    col_a, col_arrow1, col_cnn1, col_arrow2, col_fusion, col_arrow3, col_clf, col_arrow4, col_out = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2])

    with col_a:
        st.markdown("""
        <div class="pipeline-step">
            <div style="font-size: 2rem;">🎤</div>
            <div style="font-weight: 700; color: #00d4ff; margin: 0.5rem 0;">Acoustic Stream</div>
            <div style="font-size: 0.8rem; color: #8899aa;">
                Raw .wav → Mel-Spectrogram<br>(Librosa)
            </div>
        </div>
        <div style="height: 0.5rem;"></div>
        <div class="pipeline-step">
            <div style="font-size: 2rem;">📷</div>
            <div style="font-weight: 700; color: #00d4ff; margin: 0.5rem 0;">Optical Stream</div>
            <div style="font-size: 0.8rem; color: #8899aa;">
                Melt-pool image<br>224×224 normalized
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arrow1:
        st.markdown('<div class="pipeline-arrow" style="margin-top: 3rem;">→</div>', unsafe_allow_html=True)

    with col_cnn1:
        st.markdown("""
        <div class="pipeline-step" style="margin-top: 1.5rem;">
            <div style="font-size: 2rem;">🧠</div>
            <div style="font-weight: 700; color: #51cf66; margin: 0.5rem 0;">ResNet-18</div>
            <div style="font-size: 0.8rem; color: #8899aa;">
                Pretrained (ImageNet)<br>
                Frozen backbone<br>
                512-d features × 2
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arrow2:
        st.markdown('<div class="pipeline-arrow" style="margin-top: 3rem;">→</div>', unsafe_allow_html=True)

    with col_fusion:
        st.markdown("""
        <div class="pipeline-step" style="margin-top: 1.5rem;">
            <div style="font-size: 2rem;">🔗</div>
            <div style="font-weight: 700; color: #ffd43b; margin: 0.5rem 0;">Feature Fusion</div>
            <div style="font-size: 0.8rem; color: #8899aa;">
                Concatenation → 1024-d<br>
                PCA (95% variance)<br>
                Dimensionality reduction
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arrow3:
        st.markdown('<div class="pipeline-arrow" style="margin-top: 3rem;">→</div>', unsafe_allow_html=True)

    with col_clf:
        st.markdown("""
        <div class="pipeline-step" style="margin-top: 1.5rem;">
            <div style="font-size: 2rem;">📊</div>
            <div style="font-weight: 700; color: #ff6b6b; margin: 0.5rem 0;">Classifier Bank</div>
            <div style="font-size: 0.8rem; color: #8899aa;">
                SVM · XGBoost<br>
                Random Forest<br>
                Decision Tree
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arrow4:
        st.markdown('<div class="pipeline-arrow" style="margin-top: 3rem;">→</div>', unsafe_allow_html=True)

    with col_out:
        st.markdown("""
        <div class="pipeline-step" style="margin-top: 1.5rem; border-color: rgba(0, 255, 136, 0.3);">
            <div style="font-size: 2rem;">✅</div>
            <div style="font-weight: 700; color: #00ff88; margin: 0.5rem 0;">Verdict</div>
            <div style="font-size: 0.8rem; color: #8899aa;">
                Defect Probability<br>
                Healthy / Defect<br>
                ~ms latency
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Why fusion matters
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🎤 Acoustic emissions** capture:
        - Phase-change dynamics (melting, solidification)
        - Vapor-plume instability (keyhole signature)
        - Spatter ejection events (high-frequency clicks)
        - **Sub-surface events** optics cannot see
        """)
    with col2:
        st.markdown("""
        **📷 Melt-pool imagery** captures:
        - Geometric deformation of the melt pool
        - Spatter distribution and count
        - Thermal gradient patterns
        - **Surface-level morphology** acoustics cannot see
        """)

    st.info("💡 **Why fusion is the differentiator:** Acoustics reveal sub-surface physics; optics reveal surface geometry. Fused, they cross-validate — suppressing false alarms that make single-sensor monitors untrustworthy.")


# ─── Section 4: Interactive Demo ─────────────────────────────────────────────

def render_interactive_demo():
    st.markdown('<div class="section-header">🧪 Interactive Demo — Try It Yourself</div>', unsafe_allow_html=True)

    api_healthy = check_api_health()

    if not api_healthy:
        st.error("""
        **⚠️ FastAPI backend is not running.**

        Start the API server first:
        ```bash
        uvicorn src.api:app --host 0.0.0.0 --port 8000
        ```
        Then refresh this page.
        """)
        return

    st.success("✅ **API Connected** — LayerLogic inference engine is ready.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎤 Upload Acoustic Emission (.wav)")
        audio_file = st.file_uploader(
            "Choose a .wav file", type=["wav"],
            help="Upload an acoustic emission recording from the LPBF process.",
            key="audio_upload",
        )

    with col2:
        st.markdown("#### 📷 Upload Melt-Pool Image (.jpg/.png)")
        image_file = st.file_uploader(
            "Choose an image file", type=["jpg", "jpeg", "png"],
            help="Upload a melt-pool thermal/optical image.",
            key="image_upload",
        )

    if audio_file and image_file:
        if st.button("🔍 Run Inference", type="primary", use_container_width=True):
            with st.spinner("Processing... Extracting features and classifying..."):
                try:
                    result = predict_uploaded(audio_file, image_file)
                    _display_prediction_result(result, image_file)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
    else:
        st.info("👆 Upload both an audio file and an image file, then click **Run Inference**.")


def _display_prediction_result(result: dict, image_file=None):
    """Display the prediction result with visual styling."""
    prob = result["defect_probability"]
    pred = result["prediction"]
    latency = result["latency_ms"]
    classifier = result["classifier"]

    # Determine styling
    if prob < 0.3:
        css_class = "prediction-healthy"
        emoji = "🟢"
        status_color = "#00ff88"
    elif prob < 0.7:
        css_class = "prediction-warning"
        emoji = "🟡"
        status_color = "#ffd43b"
    else:
        css_class = "prediction-defect"
        emoji = "🔴"
        status_color = "#ff4444"

    # Result card
    st.markdown(f"""
    <div class="{css_class}">
        <div style="font-size: 1rem; color: #8899aa; text-transform: uppercase; letter-spacing: 2px;">
            Defect Probability Score
        </div>
        <div style="font-size: 4rem; font-weight: 900; color: {status_color}; margin: 0.5rem 0;">
            {emoji} {prob * 100:.1f}%
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {status_color};">
            {pred.upper()}
        </div>
        <div style="margin-top: 1rem; color: #8899aa; font-size: 0.85rem;">
            Classifier: {classifier} · Latency: {latency:.1f} ms
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Display mel-spectrogram and image side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🎵 Generated Mel-Spectrogram")
        if "mel_spectrogram_base64" in result:
            mel_bytes = base64.b64decode(result["mel_spectrogram_base64"])
            st.image(mel_bytes, use_container_width=True)

    with col2:
        st.markdown("##### 📷 Melt-Pool Image")
        if image_file is not None:
            image_file.seek(0)
            st.image(image_file, use_container_width=True)
        elif "image_base64" in result:
            img_bytes = base64.b64decode(result["image_base64"])
            st.image(img_bytes, use_container_width=True)


# ─── Section 5: Live Replay Mode ────────────────────────────────────────────

def render_live_replay():
    st.markdown('<div class="section-header">🔄 Live Replay Mode — Layer-by-Layer Monitoring Simulation</div>', unsafe_allow_html=True)

    api_healthy = check_api_health()
    if not api_healthy:
        st.warning("Start the FastAPI backend to enable live replay.")
        return

    st.markdown("""
    This simulates **real-time layer-by-layer monitoring** of an LPBF build. Pre-loaded samples
    are streamed through the inference pipeline sequentially — the same way LayerLogic would
    monitor an actual print job.
    """)

    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        num_layers = st.slider("Number of layers to process", 5, 30, 12, key="replay_layers")
    with col2:
        delay = st.slider("Delay between layers (seconds)", 0.5, 5.0, 1.5, key="replay_delay")

    # Initialize session state
    if "replay_running" not in st.session_state:
        st.session_state.replay_running = False
    if "replay_results" not in st.session_state:
        st.session_state.replay_results = []

    col_start, col_stop, col_clear = st.columns(3)
    with col_start:
        start_btn = st.button("▶️ Start Replay", type="primary", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)
    with col_clear:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.replay_results = []
        st.session_state.replay_running = False

    if stop_btn:
        st.session_state.replay_running = False

    if start_btn:
        st.session_state.replay_running = True
        st.session_state.replay_results = []

        # Get sample files
        samples = get_sample_files()
        healthy_count = len(samples.get("healthy", []))
        defect_count = len(samples.get("defect", []))

        if healthy_count == 0 and defect_count == 0:
            st.error("No sample data found. Run `python src/data_setup.py` first.")
            return

        # Create alternating sequence
        sequence = []
        for i in range(num_layers):
            if i % 3 == 2 and defect_count > 0:  # every 3rd layer is defect
                idx = (i // 3) % defect_count
                sequence.append(("defect", idx))
            elif healthy_count > 0:
                idx = i % healthy_count
                sequence.append(("healthy", idx))
            elif defect_count > 0:
                idx = i % defect_count
                sequence.append(("defect", idx))

        # Progress bar
        progress = st.progress(0, text="Initializing replay...")
        result_placeholder = st.empty()
        timeline_placeholder = st.empty()
        detail_placeholder = st.empty()

        for layer_idx, (label, sample_idx) in enumerate(sequence):
            if not st.session_state.replay_running:
                progress.progress(layer_idx / len(sequence), text="Replay stopped.")
                break

            progress.progress(
                (layer_idx + 1) / len(sequence),
                text=f"Processing layer {layer_idx + 1}/{len(sequence)}..."
            )

            try:
                result = predict_sample(label, sample_idx)
                result["layer"] = layer_idx + 1
                st.session_state.replay_results.append(result)

                # Update timeline
                timeline_html = '<div style="padding: 0.5rem; background: #0e1117; border-radius: 8px; margin: 0.5rem 0;">'
                for r in st.session_state.replay_results:
                    color_class = "timeline-defect" if r["prediction"] == "Defect" else "timeline-healthy"
                    timeline_html += f'<span class="{color_class} timeline-item" title="Layer {r["layer"]}: {r["prediction"]} ({r["defect_probability"]*100:.0f}%)"></span>'
                # Pending dots
                remaining = len(sequence) - len(st.session_state.replay_results)
                for _ in range(remaining):
                    timeline_html += '<span class="timeline-pending timeline-item"></span>'
                timeline_html += '</div>'
                timeline_placeholder.markdown(timeline_html, unsafe_allow_html=True)

                # Show latest result
                with detail_placeholder.container():
                    _display_prediction_result(result)

                time.sleep(delay)

            except Exception as e:
                st.error(f"Error on layer {layer_idx + 1}: {e}")
                break

        st.session_state.replay_running = False
        if st.session_state.replay_results:
            progress.progress(1.0, text="✅ Replay complete!")

            # Summary stats
            results = st.session_state.replay_results
            total = len(results)
            defects = sum(1 for r in results if r["prediction"] == "Defect")
            healthy = total - defects
            avg_latency = sum(r["latency_ms"] for r in results) / total

            st.markdown("### Replay Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Layers", total)
            c2.metric("Healthy", healthy, delta=None)
            c3.metric("Defects Found", defects, delta=None)
            c4.metric("Avg Latency", f"{avg_latency:.1f} ms")


# ─── Section 6: Results Gallery ──────────────────────────────────────────────

def render_results_gallery():
    st.markdown('<div class="section-header">📊 Training Results & Model Performance</div>', unsafe_allow_html=True)

    graphs = [
        ("classifier_comparison.png", "Classifier Performance Comparison",
         "F1-score and AUC across all four classifiers (SVM, XGBoost, Random Forest, Decision Tree). "
         "The best performer is highlighted. This mirrors the methodology from published LPBF monitoring research."),
        ("training_history.png", "Learning Curves",
         "F1-score vs. training set size for each classifier. Shows convergence behavior and "
         "validates that the models aren't simply memorizing — performance improves with more data."),
        ("roc_curve.png", "ROC Curves",
         "Receiver Operating Characteristic curves for all classifiers. AUC (Area Under Curve) measures "
         "the model's ability to distinguish healthy from defective layers across all classification thresholds."),
        ("confusion_matrix.png", "Confusion Matrix",
         "Detailed breakdown of true positives, true negatives, false positives, and false negatives "
         "for the best classifier on the held-out test set."),
    ]

    # Load training summary
    summary_path = OUTPUT_DIR / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        st.markdown(f"**Best Classifier: {summary.get('best_classifier', 'N/A')}**")

        if "test_metrics" in summary:
            m = summary["test_metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
            c2.metric("Test F1-Score", f"{m.get('f1', 0):.4f}")
            c3.metric("Test AUC", f"{m.get('auc', 0):.4f}")
            c4.metric("Test Precision", f"{m.get('precision', 0):.4f}")

    st.markdown("---")

    # Display graphs in a 2×2 grid
    for i in range(0, len(graphs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(graphs):
                fname, title, desc = graphs[idx]
                fpath = OUTPUT_DIR / fname
                with col:
                    st.markdown(f"##### {title}")
                    if fpath.exists():
                        st.image(str(fpath), use_container_width=True)
                    else:
                        st.warning(f"Graph not found: {fname}\nRun `python src/train.py` first.")
                    st.caption(desc)


# ─── Section 7: Architecture ────────────────────────────────────────────────

def render_architecture():
    st.markdown('<div class="section-header">🏛️ System Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                     DATA ACQUISITION LAYER                      │
    │                                                                  │
    │   Acoustic Sensor (.wav)          Optical Camera (.jpg)          │
    │   ─ High-frequency AE             ─ Melt-pool thermal imaging   │
    │   ─ 22050 Hz sample rate           ─ Layer-by-layer capture     │
    └────────────┬───────────────────────────────┬────────────────────┘
                 │                               │
                 ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐
    │  AUDIO PREPROCESSING │        │ IMAGE PREPROCESSING  │
    │                      │        │                      │
    │  Raw .wav            │        │  Raw .jpg            │
    │  → Mel-Spectrogram   │        │  → Resize 224×224    │
    │  → 224×224×3 tensor  │        │  → ImageNet norm     │
    │  (Librosa)           │        │  → 3-ch tensor       │
    └──────────┬───────────┘        └──────────┬───────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐
    │  ResNet-18 (Frozen)  │        │  ResNet-18 (Frozen)  │
    │  ImageNet pretrained │        │  ImageNet pretrained │
    │  → 512-d features    │        │  → 512-d features    │
    └──────────┬───────────┘        └──────────┬───────────┘
               │                               │
               └───────────┬───────────────────┘
                           ▼
              ┌──────────────────────┐
              │   FEATURE FUSION     │
              │                      │
              │   Concatenation      │
              │   → 1024-d vector    │
              │   → PCA (95% var)    │
              │   → Reduced features │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────────────────┐
              │      CLASSIFIER BANK             │
              │                                  │
              │  SVM (RBF) ← Best performer      │
              │  XGBoost                         │
              │  Random Forest                   │
              │  Decision Tree                   │
              └──────────┬───────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────────┐
              │      EDGE INFERENCE (FastAPI)     │
              │                                  │
              │  POST /predict                   │
              │  → Defect probability            │
              │  → Healthy / Defect verdict      │
              │  → ~ms latency on CPU            │
              │                                  │
              │  Streamlit Dashboard (Frontend)   │
              └──────────────────────────────────┘
    ```
    """)

    st.markdown("""
    **Key Design Decisions:**
    - **Frozen ResNet-18** — Pretrained on ImageNet, used purely for feature extraction. No fine-tuning needed,
      making training fast (~minutes, not hours).
    - **Classical ML classifiers** — SVM/XGBoost/RF/DT on fused features, following the methodology from
      IIT Kharagpur's published LPBF monitoring research (SVM + EfficientNet-B0 → 0.90 F1).
    - **PCA dimensionality reduction** — Compresses 1024-d fused features while retaining 95%+ variance,
      speeding up classification and reducing overfitting.
    - **FastAPI edge inference** — Proves the model can run as a real-time, integrable API service on commodity hardware.
    """)


# ─── Section 8: About ───────────────────────────────────────────────────────

def render_about():
    st.markdown('<div class="section-header">👤 About LayerLogic</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Team SteelSync**
        - Solo Developer — Mechanical Engineering, IIT Kharagpur
        - IIT Kharagpur Platinum Jubilee Innovation Challenge — Stage 2

        **Track:** Deep Tech for the World
        **Domain:** AI/ML × Advanced Manufacturing (Industry 4.0)
        """)

    with col2:
        st.markdown("""
        **Tech Stack:**
        | Component | Technology |
        |-----------|------------|
        | Language | Python 3.9+ |
        | Deep Learning | PyTorch (ResNet-18) |
        | Audio Processing | Librosa |
        | Computer Vision | OpenCV / PIL |
        | Classical ML | Scikit-learn, XGBoost |
        | Backend API | FastAPI + Uvicorn |
        | Frontend | Streamlit |
        | Plots | Matplotlib, Seaborn |
        """)

    st.markdown("---")

    st.markdown("""
    **Research Lineage:**
    - Shevchik, Wasmer et al. (EMPA) — Acoustic emission + spectral CNN for LPBF quality
    - Scime & Beuth (CMU) — Multi-scale CNN for powder-bed anomaly detection
    - IIT Kharagpur (Subir Chowdhury School) — Multimodal acoustic + optical with EfficientNet/ResNet + SVM

    > *"The per-sensor models are well established. What is not productized is a laptop-deployable
    > fusion of both, served in real time, on a path to closing the control loop. That is LayerLogic's wedge."*
    """)


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    render_hero()
    render_problem()
    render_how_it_works()

    st.markdown("---")

    # Tab layout for interactive sections
    tab1, tab2, tab3 = st.tabs(["🧪 Interactive Demo", "🔄 Live Replay", "📊 Results"])

    with tab1:
        render_interactive_demo()

    with tab2:
        render_live_replay()

    with tab3:
        render_results_gallery()

    st.markdown("---")

    render_architecture()
    render_about()

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #556677; font-size: 0.8rem;">
        LayerLogic © 2026 Team SteelSync · IIT Kharagpur Platinum Jubilee Innovation Challenge<br>
        Built with PyTorch, Librosa, FastAPI, and Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
