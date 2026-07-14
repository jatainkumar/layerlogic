# LayerLogic
### Real-Time Defect Detection in Metal Additive Manufacturing via Multimodal Deep Learning

**Team:** SteelSync (Solo Developer — Mechanical Engineering) · **Event:** IIT Kharagpur Platinum Jubilee Innovation Challenge — Stage 2
**Track:** Deep Tech for the World · **Domain:** Artificial Intelligence / Machine Learning × Advanced Manufacturing (Industry 4.0)

---

## Executive Summary

Metal Additive Manufacturing (AM) is reshaping aerospace, biomedical, and automotive engineering by enabling lightweight, topology-optimized geometries impossible to cast or machine. Yet the technology is held back by a single, expensive failure mode: **stochastic micro-defects — porosity, lack-of-fusion, and keyhole collapse — that form invisibly during the build and are only discovered afterwards via costly X-ray CT scanning.** A single flawed build can waste tens of hours, kilograms of gas-atomized powder, and enormous energy — a reactive, post-mortem quality regime fundamentally incompatible with Industry 4.0.

**LayerLogic is a purely software, multimodal-AI "diagnostic brain" that detects these defects in real time, layer-by-layer, as the part is printed.** It fuses two physically independent data streams — the **acoustic emissions** of the laser–powder interaction and the **optical imagery** of the melt pool — through lightweight convolutional neural networks, and outputs a per-layer defect-probability verdict. This lets an operator **halt or, ultimately, auto-correct a failing build instantly**, converting AM quality assurance from days-late CT inspection into millisecond-latency, in-process control.

Critically, LayerLogic requires **no hardware modification to the printer** and runs on **edge-class compute** (demonstrated on a single laptop via an optimized FastAPI inference service). It is designed as a drop-in API for existing commercial LPBF machines — a scalable software layer, not a bespoke instrument.

---

## The Problem — Why Metal AM Is Not Yet Trustworthy

**Metal 3D printing has a physics problem that becomes an economics problem.**

- The **melt pool** — the microscopic pool of molten metal beneath the laser — is chaotic. Thermal fluctuations, spatter ejection, and **keyhole-mode collapse** trap gas and leave unmelted voids.
- These defects are **sub-surface and stochastic**: they cannot be seen from the top, and parameters validated on one build do not guarantee the next.
- **Detection is reactive and expensive.** Defects surface only after the multi-hour build, via micro-CT or destructive metallography. By then the part — and all its material and energy — is already scrapped.
- Geometry compounds the problem: a part with thin walls and thick hubs dissipates heat unevenly, so **one set of "correct" laser parameters produces defects in different zones.** One-size-fits-all monitoring fails on real parts.

The consequence: metal AM is trapped in low-volume prototyping when its true value lies in **qualified serial production of safety-critical parts.**

---

## The Solution — How LayerLogic Works

LayerLogic reframes quality control from *inspecting the object afterwards* to *reading the physics of the process as it happens.* It listens and watches simultaneously:

**Stream 1 — Acoustic (the process's sound).** High-frequency acoustic emissions carry the signatures of phase change, vapor-plume dynamics, and keyhole instability — sub-surface events that optics cannot see. Raw `.wav` signals are transformed into **Mel-spectrograms**, turning non-stationary time-series into rich time–frequency images.

**Stream 2 — Optical (the process's appearance).** High-speed melt-pool imagery captures geometric deformation, spatter, and thermal-gradient signatures — the spatial-domain view of build health.

**Fusion & Diagnosis.** Lightweight **Convolutional Neural Networks (ResNet-18-class)** extract compact feature vectors from *both* the spectrograms and the melt-pool images. These vectors are **concatenated and fused**, then classified to predict the structural integrity (defect probability and type) of the current layer.

**Why fusion is the differentiator:** each modality is blind where the other sees. Optics reveal surface geometry; acoustics reveal sub-surface events. Fused, they cross-validate — suppressing the false alarms that make single-sensor commercial monitors untrustworthy, and enabling *defect-type* diagnosis rather than a mere anomaly flag.

**Deployment.** The trained model is wrapped in an optimized **FastAPI edge-inference service** feeding a local dashboard, proving near-zero-latency operation on commodity hardware — no cloud dependency, no machine retrofit.

---

## Why This Is Deep Tech, Not Surface-Level

This project sits at the intersection of **thermodynamics, signal processing, and multimodal deep learning.** Processing non-stationary acoustic emissions and high-speed melt-pool imagery, and *fusing* two disparate data modalities to infer hidden thermodynamic anomalies like keyhole porosity, demands architectures far beyond simple classifiers. Engineering the pipeline for **real-time edge inference** attacks a genuine algorithmic bottleneck in smart manufacturing. Every AI decision is grounded in melt-pool physics — this is hard-tech research bridging mechanical engineering and artificial intelligence.

---

## Technical Feasibility & Development Stage

- **Approach validated in the literature and in prior LPBF research** (acoustic-emission monitoring, CNN-based melt-pool classification, multimodal fusion — see the companion demo document for the model lineage).
- **Software-only and laptop-trainable:** lightweight edge models (ResNet-18) rather than massive networks, deliberately chosen for the single-consumer-laptop constraint.
- **Trained on open-source data** (NIST AM material databases; Kaggle AM acoustic and melt-pool datasets), removing dependence on private machine time.
- **TRL trajectory:** working software proof-of-concept with a live inference dashboard → pilot integration with an AM machine / service bureau → closed-loop control.

---

## Impact — Scale, Depth, and Reach

- **Material & energy savings:** halting a failing build early prevents wasted powder and the energy of a full failed print — directly cutting the carbon footprint of Industry 4.0.
- **Cost of qualification:** replacing sampling-based CT with in-line certification collapses the dominant cost of producing safety-critical AM parts.
- **Breadth:** delivered as an API, LayerLogic can ride on the existing installed base of commercial LPBF machines rather than requiring new hardware — a software-scale distribution model.

---

## Expansion — Beyond LPBF and Beyond Metals

The core thesis — *multimodal sensing + fusion AI + closed-loop control* — is **process- and material-agnostic.** The same pipeline extends by re-training and re-weighting modalities:

**Materials (near-term):** Inconel/nickel superalloys → **Titanium, stainless steel, aluminium alloys**, and complex features such as **thin walls and lattice structures**.

**Other AM processes (platform play):**

| Process | What transfers | New signal emphasis |
|---|---|---|
| **Directed Energy Deposition (DED / LMD)** | Melt-pool optical + acoustic monitoring | Bead height/width, dilution |
| **Electron Beam Melting (EB-PBF / EBM)** | Layerwise imaging + thermal signatures | Backscatter-electron imaging |
| **Wire Arc AM (WAAM)** | Acoustic + thermal + optical fusion | Arc current/voltage, interpass temp |
| **Binder Jetting** | Layerwise powder-bed imaging | Spreading & saturation uniformity |
| **Fused Deposition Modeling (FDM/FFF, polymer)** | Layer imaging + acoustic anomaly detection | Under-extrusion, warping, delamination |
| **Vat Photopolymerization (SLA/DLP)** | Optical layer monitoring | Cure uniformity, detachment force |

This gives a **single-platform, multi-market story:** prove it on the hardest problem (LPBF metal porosity), then port the AI stack across the AM landscape.

---

## Viability & Path to Deployment

- **Who pays:** AM service bureaus, aerospace/medical primes, and machine OEMs seeking in-process qualification to escape the CT bottleneck.
- **Model:** software subscription / per-build licensing / OEM integration — no per-unit hardware cost.
- **Wedge:** the **fusion layer** and the **closed-loop roadmap** — turning passive monitoring logs into an active, trustworthy controller.

---

## Roadmap

- **Phase 1 — Detect (now):** open-source data, per-modality CNNs, fusion classifier, live FastAPI dashboard demonstrating real-time inference.
- **Phase 2 — Certify:** integration with real LPBF data streams; layer-resolved defect maps and a digital part certificate; pilot with a bureau/OEM.
- **Phase 3 — Control:** feed defect predictions back to laser power / scan speed to *prevent* defects, not merely detect them — autonomous, zero-defect manufacturing.

---

## The Vision

> **Detection makes additive manufacturing inspectable. Closed-loop control makes it dependable.** LayerLogic aims to be the quality operating system that finally makes metal 3D printing trustworthy enough for serial, safety-critical production.
