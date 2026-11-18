# Attendance Pro: Face Recognition Attendance System

An enterprise-grade, real-time automated attendance tracking system powered by Computer Vision and Deep Learning. Features OpenCV YuNet DNN face detection, CNN/FaceNet 128-dimensional facial embedding extraction, landmark-guided affine preprocessing, vectorized similarity matching (Cosine Similarity & Euclidean Distance), duplicate-prevention cooldown logic, and passive anti-spoofing protection.

---

## 🚀 One-Click Streamlit Cloud Deployment

1. Fork or push this repository to GitHub (`saiteja11189/face-recognisation-attendance-system`).
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"** and select:
   - **Repository:** `saiteja11189/face-recognisation-attendance-system`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!** — System packages (`libgl1`, `libglib2.0-0` in `packages.txt`) and Python headless dependencies (`requirements.txt`) are automatically pre-configured.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Live Camera / Image In    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Face Detection (OpenCV DNN) |
                      | - Model: YuNet ONNX         |
                      | - Bounding box + 5 landmarks|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Preprocessing & Alignment   |
                      | - 5-point similarity warp   |
                      | - CLAHE illumination boost  |
                      | - Target resize (112 x 112) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Feature Extraction (CNN)    |
                      | - Model: SFace / FaceNet    |
                      | - 128D Unit Latent Vector   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Similarity Matching Engine  |
                      | - Cosine Similarity: u . v  |
                      | - Euclidean Norm: ||u - v|| |
                      | - Thresholding & Ranking    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Attendance Business Logic   |
                      | - Cooldown deduplication    |
                      | - Present vs. Late status   |
                      | - SQLite DB + CSV / Excel   |
                      +-----------------------------+
```

---

## Core Technologies & Theoretical Concepts

### 1. Face Detection (`FaceDetectorYN`)
- **Detector**: Lightweight CNN (`YuNet`) running natively inside OpenCV's DNN graph engine.
- **Output**: 
  - Bounding rectangle coordinates $(x, y, w, h)$.
  - 5 facial landmarks: Right Eye $(x_{re}, y_{re})$, Left Eye $(x_{le}, y_{le})$, Nose Tip $(x_{nt}, y_{nt})$, Right Mouth Corner $(x_{rcm}, y_{rcm})$, and Left Mouth Corner $(x_{lcm}, y_{lcm})$.
  - Detection confidence score $\in [0.0, 1.0]$.

### 2. Preprocessing & Alignment
- **Landmark Alignment**: Head tilt and camera angle cause facial distortion. Using the eye coordinates, the rotation angle $\theta$ is derived:
  $$\theta = \arctan2(\Delta y, \Delta x) \times \frac{180}{\pi}$$
  An affine transformation rotates the face so the interpupillary line is horizontal.
- **Illumination Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE) applied to the luminance channel ($L$ in LAB color space) compensates for uneven classroom/office lighting.

### 3. CNN / FaceNet Embedding Extraction (`FaceRecognizerSF`)
- **Latent Space**: Transforms the canonical $112 \times 112 \times 3$ aligned face crop into a high-dimensional vector $\mathbf{v} \in \mathbb{R}^{128}$.
- **L2 Normalization**: Each embedding is mapped to the unit hypersphere:
  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} \quad \text{such that} \quad \|\hat{\mathbf{v}}\|_2 = 1.0$$
- **Burst Averaging**: During registration, multiple frames are captured, and their centroid is computed:
  $$\mathbf{v}_{\text{composite}} = \frac{\sum_{i=1}^{N} \hat{\mathbf{v}}_i}{\|\sum_{i=1}^{N} \hat{\mathbf{v}}_i\|_2}$$

### 4. Similarity Matching Metrics
The system compares a query embedding $\mathbf{q}$ against the stored gallery matrix $\mathbf{M} \in \mathbb{R}^{K \times 128}$:

#### A. Cosine Similarity (Default)
$$\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{g}}{\|\mathbf{q}\|_2 \|\mathbf{g}\|_2} = \mathbf{q} \cdot \mathbf{g} \quad (\text{since vectors are unit-normalized})$$
- **Vectorized Gallery Lookup**: $\mathbf{s} = \mathbf{M} \cdot \mathbf{q}$ (completed in $<0.1 \text{ ms}$).
- **Threshold**: $\ge 0.38$ indicates a confirmed match.

#### B. Euclidean (L2) Distance
$$d(\mathbf{q}, \mathbf{g}) = \|\mathbf{q} - \mathbf{g}\|_2 = \sqrt{\sum_{i=1}^{128} (q_i - g_i)^2}$$
- **Threshold**: $\le 1.10$ indicates a confirmed match.

### 5. Attendance Management & Cooldown Deduplication
- **Duplicate Prevention**: If a recognized person remains in front of the camera, a configurable cooldown window (default: 60 minutes) prevents logging redundant records.
- **Punctuality Tracking**: Compares timestamp against official hours (e.g. 9:15 AM) to automatically flag arrivals as `Present` or `Late`.
- **Exporting**: Automatically writes logs to SQLite (`data/attendance.db`), CSV (`data/attendance_logs/attendance_YYYY-MM-DD.csv`), and multi-sheet styled Excel files (`.xlsx`).

### 6. Passive Anti-Spoofing
- Rejects presentation attacks (printed paper portraits or smartphone screen replays) via:
  - **Texture Sharpness**: Laplacian frequency variance $\nabla^2 I$ filter.
  - **Dynamic Micro-Motion**: Temporal variance of landmark saccades across consecutive frames.

---

## Directory Structure

```
Attendace Pro/
├── core/
│   ├── __init__.py            # Clean module exports
│   ├── config.py              # System thresholds, file paths, model configs
│   ├── model_loader.py        # Automatic model downloader and verification
│   ├── detector.py            # OpenCV YuNet DNN face detection & landmarks
│   ├── preprocessor.py        # Affine eye-alignment, CLAHE, tensor normalization
│   ├── embedder.py            # SFace/FaceNet 128D embedding extraction
│   ├── matcher.py             # Vectorized Cosine Similarity & L2 Distance matcher
│   ├── database.py            # SQLite database manager for gallery & attendance
│   ├── attendance_manager.py  # Cooldown logic, KPIs, CSV/Excel export engine
│   ├── camera.py              # OpenCV VideoCapture wrapper with FPS estimation
│   └── anti_spoofing.py       # Passive liveness & spoof rejection filter
├── ui/
│   ├── __init__.py
│   └── components.py          # Modern CSS styling and KPI metric components
├── data/
│   ├── attendance.db          # SQLite database
│   ├── attendance_logs/       # Auto-saved CSV & Excel logs
│   └── faces/                 # Saved aligned face thumbnails
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── tests/
│   ├── test_preprocessor.py
│   ├── test_embedder_matcher.py
│   ├── test_database.py
│   └── test_attendance.py
├── app.py                     # Streamlit Interactive Web Application
├── cli.py                     # Command-Line Interface tool
├── requirements.txt           # Dependencies
└── README.md                  # Comprehensive Documentation
```

---

## Quickstart Guide

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Run the interactive dashboard in your browser:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The web interface features 5 comprehensive tabs:
1. **📸 Live Attendance Tracker**: Real-time webcam snapshot or group classroom photo processor with bounding boxes, confidence badges, and instant attendance status.
2. **👤 Register New Identity**: Form to enroll students/employees via webcam or image upload with 112x112 aligned preview.
3. **📊 Attendance Logs & Analytics**: Daily attendance rate KPIs, punctuality counts, date/department filters, and one-click **Download CSV** / **Download Excel** buttons.
4. **🗂️ Enrolled Identities**: Identity gallery cards with enrolled face photos, IDs, departments, and deletion actions.
5. **⚙️ Calibration & Settings**: Interactive sliders for Cosine/L2 thresholds, detection sensitivity, cooldown minutes, and anti-spoofing toggle.

---

## CLI Usage (Command Line Interface)

The standalone CLI (`cli.py`) allows terminal-based management and desktop camera monitoring:

### 1. Live Desktop Monitor
Launch a native OpenCV HUD window with real-time bounding boxes, recognition tags, and automatic attendance marking:
```bash
python cli.py live
```
*Options:*
- `--camera 0`: Specify webcam index.
- `--metric cosine` or `--metric l2`: Select similarity metric.
- `--no-spoof`: Disable anti-spoofing filter.
- Keybindings: Press `q` to quit, `r` to reload gallery.

### 2. Enroll a Person
Enroll via webcam burst capture (averages 5 samples for maximum accuracy):
```bash
python cli.py enroll --id EMP101 --name "Alex Morgan" --dept "Computer Science" --email "alex@org.edu"
```
Or enroll from a photo file:
```bash
python cli.py enroll --id EMP102 --name "Elena Rostova" --dept "AI Lab" --photo "path/to/portrait.jpg"
```

### 3. List Registered Identities
```bash
python cli.py list-users
```

### 4. Delete an Identity
```bash
python cli.py delete-user --id EMP101
```

### 5. Generate Attendance Reports
Display daily summary to terminal and export CSV/Excel:
```bash
python cli.py report --csv --excel
```

---

## Automated Test Suite

Run the full test suite with all assertions:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

All 10 unit test cases verify:
- Eye-landmark affine rotation and CLAHE illumination boost.
- Cosine similarity and Euclidean distance numerical correctness.
- Multi-sample embedding centroid averaging.
- Database transactions, SQLite serialized embedding storage, and gallery caching.
- Duplicate prevention cooldown and daily attendance summary metrics.
