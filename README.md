Real-Time Player Re-Identification & YOLO Tracking

## 📌 **Project Purpose**
This project demonstrates real-time player detection and stable ID assignment (Re-Identification, or ReID) in football videos.

It combines:
- **YOLOv11** (Ultralytics) for detecting players, referees, goalkeepers, etc.
- **OSNet** (Torchreid) for ReID, so the same player keeps the same global ID even when detection IDs change.
- Real-time visualization to see tracking quality.

---

## 📦 **Files & their purpose**
| File / Folder | Purpose |
|---------------|--------:|
| `player_re-identification.py` | Main script: real-time detection + ReID tracking with global IDs |
| `real_time_detection_yolo.py` | Run YOLOv11 detection only (no ReID) – see how good the model is |
| `yolo_detection_savetrue.py` | YOLO detection and save frames or video |
| `testcuda.py` | Check if CUDA (GPU) is available |
| `requirements.txt` | Python dependencies |
| `data_videos/` | Sample football videos to test |

> ℹ️ I discovered the custom class names (`player`, `referee`, etc.) by printing `box` in `real_time_detection_yolo.py`.

---

## 🛠️ How to Download, Set Up, and Run

```bash
# 1. Clone the repository
git clone <your-repo-link>
cd your-repo-folder
```
```bash
# 2. Create and activate virtual environment
# Windows:
python -m venv venv
venv\Scripts\activate
```
```bash
# Linux/macOS:
python3 -m venv venv
source venv/bin/activate
```
```bash
# 3. Install dependencies
pip install -r requirements.txt
```

### 4. Download YOLOv11 model weights

➡️ [Click here to download `best.pt`](https://drive.google.com/uc?export=download&id=1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD)

After downloading, place the `best.pt` file into the root of this project folder (same location as your `.py` files).

```bash
# 5. run file
python player_re-identification.py
```

