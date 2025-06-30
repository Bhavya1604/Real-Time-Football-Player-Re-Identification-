import cv2
import torch
import torchreid
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import deque

# --- CONFIG ---
VIDEO_PATH = "data_videos/15sec_input_720p.mp4"
SIM_THRESHOLD = 0.6           # slightly lower for tolerance
MIN_BOX_SIZE = 30
MAX_GALLERY_SIZE = 300
MAX_FEATURE_HISTORY = 5       # keep last 5 features per track
MAX_AGE = 15                  # tolerate 15 frames without detection

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MODELS ---
detector = YOLO("best.pt")
class_names = detector.names

reid_model = torchreid.models.build_model('osnet_x1_0', pretrained=True, num_classes=1000).to(device).eval()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- STATE ---
global_id_counter = 0
active_tracks = {}  # track_id -> {'global_id', 'features' (deque), 'age'}
inactive_gallery = []  # [{'global_id', 'features'}]

# --- FUNCTIONS ---
def extract_embedding(crop):
    try:
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = reid_model(tensor)
        return feat.cpu().numpy().flatten()
    except:
        return None

def average_features(features):
    return np.mean(features, axis=0)

def match_to_gallery(new_feat, used_ids):
    if not inactive_gallery:
        return None
    candidates = [g for g in inactive_gallery if g['global_id'] not in used_ids]
    if not candidates:
        return None
    gallery_feats = [c['features'] for c in candidates]
    ids = [c['global_id'] for c in candidates]
    sims = cosine_similarity([new_feat], gallery_feats)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] > SIM_THRESHOLD:
        return ids[best_idx]
    return None

def assign_global_id(track_id, bbox, frame, used_ids):
    global global_id_counter
    if track_id in active_tracks:
        # Update age counter
        active_tracks[track_id]['age'] = 0
        gid = active_tracks[track_id]['global_id']
        used_ids.add(gid)
        return gid
    # New or recovered track
    x1, y1, x2, y2 = map(int, bbox)
    if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
        return None
    crop = frame[y1:y2, x1:x2]
    feat = extract_embedding(crop)
    if feat is None:
        return None
    gid = match_to_gallery(feat, used_ids)
    if gid is None:
        global_id_counter += 1
        gid = global_id_counter
    # Create new active track with feature history
    active_tracks[track_id] = {
        'global_id': gid,
        'features': deque([feat], maxlen=MAX_FEATURE_HISTORY),
        'age': 0
    }
    inactive_gallery.append({'global_id': gid, 'features': feat})
    used_ids.add(gid)
    if len(inactive_gallery) > MAX_GALLERY_SIZE:
        inactive_gallery.pop(0)
    return gid

def update_active_track(track_id, bbox, frame):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    feat = extract_embedding(crop)
    if feat is not None:
        active_tracks[track_id]['features'].append(feat)

def retire_or_increment_age(current_ids):
    for tid in list(active_tracks.keys()):
        if tid not in current_ids:
            active_tracks[tid]['age'] += 1
            if active_tracks[tid]['age'] > MAX_AGE:
                # retire
                avg_feat = average_features(active_tracks[tid]['features'])
                inactive_gallery.append({'global_id': active_tracks[tid]['global_id'], 'features': avg_feat})
                del active_tracks[tid]
                if len(inactive_gallery) > MAX_GALLERY_SIZE:
                    inactive_gallery.pop(0)

def draw_annotations(frame, results, used_ids):
    color_map = {
        'player': (255, 255, 255),
        'referee': (150, 120, 0),
        'goalkeeper': (0, 0, 255)
    }
    current_track_ids = []
    if results.boxes.id is not None:
        boxes = results.boxes.data.cpu().numpy()
        for *xyxy, track_id, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = int(track_id)
            cls_id = int(cls_id)
            label = class_names.get(cls_id, f"class{cls_id}")
            if label == 'ball':
                continue
            current_track_ids.append(track_id)
            if track_id in active_tracks:
                update_active_track(track_id, (x1, y1, x2, y2), frame)
                gid = active_tracks[track_id]['global_id']
                used_ids.add(gid)
            else:
                gid = assign_global_id(track_id, (x1, y1, x2, y2), frame, used_ids)
                if gid is None:
                    continue
            color = color_map.get(label, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ID:{gid}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1+2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    retire_or_increment_age(current_track_ids)
    return frame

# --- MAIN LOOP ---
def run_realtime_tracking():
    cap = cv2.VideoCapture(VIDEO_PATH)
    print("Starting real-time tracking... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.track(source=frame, persist=True, verbose=False, conf=0.4)[0]
        used_ids = set()
        annotated = draw_annotations(frame, results, used_ids)
        cv2.imshow("ReID Stable Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

# --- ENTRY ---
if __name__ == "__main__":
    run_realtime_tracking()
