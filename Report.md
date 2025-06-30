# 📊 Report: Real-Time Football Player Re-Identification & YOLO Tracking

## ✅ Approach and Methodology

- **Objective:** Detect and track football players in real time, ensuring each player keeps a consistent global ID even if they leave and re-enter the frame, turn around, or change orientation.
- **Detection:** Used **YOLOv11** (Ultralytics) as the primary object detector to identify players, referees, goalkeepers, and the ball.
- **Re-Identification (ReID):** Integrated **OSNet** from Torchreid to extract appearance embeddings of detected players.  
- **Global ID assignment:** Maintained an `active_tracks` dictionary to track players currently visible and an `inactive_gallery` to store embeddings of players who have temporarily disappeared.  
- **Matching:** Used cosine similarity between new embeddings and the gallery to re-assign global IDs when players reappear, ensuring identity continuity.

---

## 🧪 Techniques Tried & Outcomes

- **Tuned parameters:**  
  - Lowered `SIM_THRESHOLD` to ~0.6 to allow more tolerant ReID matching, especially when player turns sideways or away.
  - Set `MAX_AGE` to 15 so a player can disappear briefly before losing their global ID.
  - Used deque to store a short history of embeddings (`MAX_FEATURE_HISTORY=5`) to smooth identity drift.

- **Visualization:**  
  - Drew bounding boxes and IDs on frames in real time.
  - Used different colors for players, referees, and goalkeepers.

- **Separate scripts:**  
  - `real_time_detection_yolo.py` to test YOLO detection only and observe detection confidence.
  - `yolo_detection_savetrue.py` to save detection results for offline analysis.
  - `testcuda.py` to confirm GPU acceleration is available.

**Outcome:**  
- Successfully achieved real-time tracking with consistent global IDs.
- Players kept the same global ID even when briefly occluded or turning.
- Short test videos ran smoothly at acceptable FPS on CUDA.

---

## ⚠️ Challenges Encountered

- **ID switches:** Despite ReID, rapid player turns or heavy occlusion still sometimes caused new IDs to be assigned.
- **Limited data:** Small sample videos and not training a domain-specific ReID model reduced overall robustness.
- **Feature drift:** Over long sequences, ReID embeddings can drift, causing mismatches.

---


## 📦 Project Files Overview

| File | Purpose |
|------|---------|
| `player_re-identification.py` | Main real-time detection + ReID tracking |
| `real_time_detection_yolo.py` | YOLO detection only |
| `yolo_detection_savetrue.py` | Save detection output |
| `testcuda.py` | Check CUDA availability |
| `requirements.txt` | Dependencies |
| `data_videos/` | Short test videos |
| `README.md` | Usage instructions |

---

✅ This project demonstrates a working prototype for real-time football player detection and re-identification, combining YOLO and deep metric learning.
