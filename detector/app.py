# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms
# from PIL import Image
# import cv2
# import os
# from ultralytics import YOLO
# from transformers import CLIPProcessor, CLIPModel
# from exif import Image as ExifImage  # Use exif library for robust EXIF parsing
# import logging
# import warnings
# warnings.filterwarnings('ignore')  # Suppress sklearn warnings

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Step 1: Generate Synthetic Dataset
# num_samples = 1000
# width_m = np.random.uniform(1, 15, num_samples)
# height_m = np.random.uniform(1, 7, num_samples)
# lat = np.random.uniform(30, 50, num_samples)
# long = np.random.uniform(-120, -70, num_samples)
# age_years = np.random.uniform(0, 10, num_samples)
# installation_score = np.random.uniform(50, 100, num_samples)
# placement_score = np.random.uniform(0.5, 1.0, num_samples)
# violence_prob = np.random.uniform(0, 1, num_samples)
# explicit_prob = np.random.uniform(0, 1, num_samples)

# size_violation = ((width_m > 10) | (height_m > 5)).astype(int)
# geo_violation = (placement_score < 0.8).astype(int)
# structural_violation = ((age_years > 5) | (installation_score < 80)).astype(int)
# content_violation = ((violence_prob > 0.7) | (explicit_prob > 0.7)).astype(int)

# df = pd.DataFrame({
#     'width_m': width_m,
#     'height_m': height_m,
#     'lat': lat,
#     'long': long,
#     'age_years': age_years,
#     'installation_score': installation_score,
#     'placement_score': placement_score,
#     'violence_prob': violence_prob,
#     'explicit_prob': explicit_prob,
#     'size_violation': size_violation,
#     'geo_violation': geo_violation,
#     'structural_violation': structural_violation,
#     'content_violation': content_violation
# })

# df.to_csv('synthetic_billboard_dataset.csv', index=False)

# # Step 2: Preprocess Data
# feature_columns = ['width_m', 'height_m', 'lat', 'long', 'age_years', 'installation_score', 'placement_score', 'violence_prob', 'explicit_prob']
# X = df[feature_columns]
# y = df[['size_violation', 'geo_violation', 'structural_violation', 'content_violation']].values

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# indices = np.arange(len(X))
# X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X_scaled, y, indices, test_size=0.2, random_state=42)

# # Step 3: Rule-Based Detectors
# def check_size(width, height):
#     reasons = []
#     if width > 10:
#         reasons.append(f"width={width:.2f}m >10")
#     if height > 5:
#         reasons.append(f"height={height:.2f}m >5")
#     if reasons:
#         return 1, f"Size violation: {' and '.join(reasons)}"
#     return 0, ""

# def check_geo(lat, long, placement_score):
#     if lat is None or long is None:
#         return 0, "Geolocation check skipped: No valid coordinates available"
#     if placement_score < 0.95:
#         return 1, f"Geo/placement violation: score={placement_score:.2f} <0.95 (lat={lat:.2f}, long={long:.2f})"
#     return 0, ""

# def check_structural(age, installation_score):
#     reasons = []
#     if age > 4:
#         reasons.append(f"age={age:.2f} >4")
#     if installation_score < 85:
#         reasons.append(f"install_score={installation_score:.2f} <85")
#     if reasons:
#         return 1, f"Structural violation: {' and '.join(reasons)}"
#     return 0, ""

# def check_content(violence_prob, explicit_prob):
#     reasons = []
#     if violence_prob > 0.65:
#         reasons.append(f"violence={violence_prob:.2f} >0.65")
#     if explicit_prob > 0.65:
#         reasons.append(f"explicit={explicit_prob:.2f} >0.65")
#     if reasons:
#         return 1, f"Content violation: {' and '.join(reasons)}"
#     return 0, ""

# # Step 4: Image/Video Processing
# def extract_exif_geo(image_path):
#     try:
#         with open(image_path, 'rb') as img_file:
#             img = ExifImage(img_file)
#             if img.has_exif and hasattr(img, 'gps_latitude') and hasattr(img, 'gps_longitude'):
#                 lat = img.gps_latitude[0] + img.gps_latitude[1]/60 + img.gps_latitude[2]/3600
#                 lon = img.gps_longitude[0] + img.gps_longitude[1]/60 + img.gps_longitude[2]/3600
#                 lat_ref = img.get('gps_latitude_ref', 'N')
#                 lon_ref = img.get('gps_longitude_ref', 'E')
#                 lat = lat if lat_ref == 'N' else -lat
#                 lon = lon if lon_ref == 'E' else -lon
#                 logger.info(f"Extracted GPS: lat={lat:.6f}, lon={lon:.6f}")
#                 return lat, lon
#             else:
#                 logger.warning("No GPS data found in EXIF")
#                 return None, None
#     except Exception as e:
#         logger.error(f"Error extracting EXIF data: {e}")
#         return None, None

# def process_frame(img, yolo_model, resnet_model, transform, clip_model, clip_processor):
#     # Convert PIL image to OpenCV format for YOLO
#     img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     results = yolo_model(img_cv)  # Detect billboards

#     # Assume the largest detected object is the billboard
#     billboard_box = None
#     max_area = 0
#     for result in results:
#         for box in result.boxes:
#             if box.cls == 0:  # Adjust class ID for billboard (fine-tune YOLO for billboards)
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 area = (x2 - x1) * (y2 - y1)
#                 if area > max_area:
#                     max_area = area
#                     billboard_box = (x1, y1, x2, y2)

#     if billboard_box is None:
#         logger.warning("No billboard detected in image")
#         return None

#     x1, y1, x2, y2 = billboard_box
#     width_px = x2 - x1
#     height_px = y2 - y1
#     logger.info(f"Detected Billboard: {width_px}x{height_px}px")

#     # Size calibration (assume 1000px = 10m; adjust based on real-world data)
#     calib_factor = 10.0 / 1000.0
#     width_m = width_px * calib_factor
#     height_m = height_px * calib_factor

#     # Content analysis with CLIP
#     violence_prob, explicit_prob = 0.0, 0.0
#     if clip_model and clip_processor:
#         inputs = clip_processor(text=["violent content", "explicit content", "neutral content"], images=img, return_tensors="pt", padding=True)
#         with torch.no_grad():
#             outputs = clip_model(**inputs)
#         probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
#         violence_prob, explicit_prob = probs[0], probs[1]
#     else:
#         # Fallback to ResNet
#         img_transformed = transform(img).unsqueeze(0)
#         with torch.no_grad():
#             resnet_features = resnet_model(img_transformed).numpy().flatten()
#         violence_prob = resnet_features[0] / max(resnet_features)
#         explicit_prob = resnet_features[1] / max(resnet_features)

#     # Structural analysis
#     img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(img_gray, 100, 200)
#     edge_density = np.sum(edges) / (img_gray.shape[0] * img_gray.shape[1])
#     age_years = min(max(edge_density * 50, 0), 10)  # Heuristic: high edges = older
#     installation_score = max(100 - edge_density * 100, 50)  # Heuristic: high edges = poor installation

#     # Placement score
#     img_array = np.array(img)
#     placement_score = 0.9 - (np.var(img_array) / 255**2) * 0.4
#     placement_score = max(min(placement_score, 1.0), 0.5)

#     return [width_m, height_m, violence_prob, explicit_prob, age_years, installation_score, placement_score]

# def extract_features_from_media(media_path, resnet_model, transform, clip_model=None, clip_processor=None, is_video=False, user_lat=None, user_lon=None):
#     features = {
#         'width_m': 0.0, 'height_m': 0.0, 'lat': 0.0, 'long': 0.0,
#         'age_years': 0.0, 'installation_score': 0.0, 'placement_score': 0.0,
#         'violence_prob': 0.0, 'explicit_prob': 0.0
#     }

#     yolo_model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model

#     if is_video:
#         frame_paths = [os.path.join(media_path, f) for f in os.listdir(media_path) if f.endswith(('.jpg', '.png'))] if os.path.isdir(media_path) else [media_path]
#         frame_features = []
#         for frame_path in frame_paths[:5]:
#             try:
#                 img = Image.open(frame_path).convert('RGB')
#                 frame_result = process_frame(img, yolo_model, resnet_model, transform, clip_model, clip_processor)
#                 if frame_result:
#                     frame_features.append(frame_result)
#             except Exception as e:
#                 logger.error(f"Error processing frame {frame_path}: {e}")
#         if frame_features:
#             avg_features = np.mean(frame_features, axis=0)
#             features.update({
#                 'width_m': avg_features[0], 'height_m': avg_features[1],
#                 'violence_prob': avg_features[2], 'explicit_prob': avg_features[3],
#                 'age_years': avg_features[4], 'installation_score': avg_features[5],
#                 'placement_score': avg_features[6]
#             })
#             lat, long = extract_exif_geo(frame_paths[0]) if frame_paths else (None, None)
#             features['lat'] = user_lat if user_lat is not None else lat if lat is not None else 40.7128  # Default: NYC
#             features['long'] = user_lon if user_lon is not None else long if long is not None else -74.0060
#     else:
#         img = Image.open(media_path).convert('RGB')
#         feature_vector = process_frame(img, yolo_model, resnet_model, transform, clip_model, clip_processor)
#         if feature_vector:
#             features.update({
#                 'width_m': feature_vector[0], 'height_m': feature_vector[1],
#                 'violence_prob': feature_vector[2], 'explicit_prob': feature_vector[3],
#                 'age_years': feature_vector[4], 'installation_score': feature_vector[5],
#                 'placement_score': feature_vector[6]
#             })
#             lat, long = extract_exif_geo(media_path)
#             features['lat'] = user_lat if user_lat is not None else lat if lat is not None else 40.7128  # Default: NYC
#             features['long'] = user_lon if user_lon is not None else long if long is not None else -74.0060

#     logger.info(f"Extracted Features for {media_path}: {features}")
#     return features

# # Initialize ResNet and CLIP
# resnet = models.resnet50(weights='DEFAULT')
# resnet = nn.Sequential(*list(resnet.children())[:-1])
# resnet.eval()
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Step 5: ML Model
# class BillboardDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# train_dataset = BillboardDataset(X_train, y_train)
# test_dataset = BillboardDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# class ViolationModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_size)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.sigmoid(self.fc3(x))
#         return x

# model = ViolationModel(input_size=len(feature_columns), output_size=y_train.shape[1])
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# epochs = 20
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# # Evaluate the model
# model.eval()
# y_pred = []
# y_true = []
# for inputs, labels in test_loader:
#     with torch.no_grad():
#         outputs = model(inputs)
#         pred = (outputs > 0.5).numpy().astype(int)
#         y_pred.append(pred)
#         y_true.append(labels.numpy())
# y_pred = np.vstack(y_pred)
# y_true = np.vstack(y_true)

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# accuracy = accuracy_score(y_true, y_pred)
# precision = precision_score(y_true, y_pred, average='macro')
# recall = recall_score(y_true, y_pred, average='macro')
# logger.info(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# torch.save(model.state_dict(), 'violation_model.pth')

# # Step 6: Detection Function
# def detect_violations(scaler, model, feature_columns, image_path=None, video_path=None, resnet=None, transform=None, clip_model=None, clip_processor=None, user_lat=None, user_lon=None):
#     is_video = video_path is not None
#     media_path = video_path if is_video else image_path
#     if media_path is None:
#         raise ValueError("Provide image_path or video_path")
    
#     features = extract_features_from_media(media_path, resnet, transform, clip_model, clip_processor, is_video, user_lat, user_lon)
    
#     # Debug: Print extracted features
#     logger.info(f"Extracted Features: { {k: f'{v:.2f}' for k, v in features.items()} }")
    
#     X_extracted = pd.DataFrame([features], columns=feature_columns)
#     X_scaled = scaler.transform(X_extracted)
#     X_torch = torch.tensor(X_scaled, dtype=torch.float32)
#     with torch.no_grad():
#         ml_preds = model(X_torch).numpy().flatten() > 0.5
    
#     logger.info(f"ML Predictions (size, geo, structural, content): {ml_preds.astype(int)}")
    
#     size_v, size_reason = check_size(features['width_m'], features['height_m'])
#     geo_v, geo_reason = check_geo(features['lat'], features['long'], features['placement_score'])
#     structural_v, structural_reason = check_structural(features['age_years'], features['installation_score'])
#     content_v, content_reason = check_content(features['violence_prob'], features['explicit_prob'])
    
#     reasons = [r for r in [size_reason, geo_reason, structural_reason, content_reason] if r and "skipped" not in r]
#     result = "violation" if reasons else "non-violating"
    
#     return result, reasons

# # Step 7: Test with Media
# test_image_path = r'C:\Users\KIIT\Downloads\billboard.jpg'
# model.load_state_dict(torch.load('violation_model.pth', weights_only=True))
# model.eval()
# # Provide user-defined coordinates if EXIF fails
# user_lat, user_lon = 40.7128, -74.0060  # Example: New York City
# result, reasons = detect_violations(scaler, model, feature_columns, 
#                                     image_path=test_image_path, 
#                                     resnet=resnet, transform=transform,
#                                     clip_model=clip_model, clip_processor=clip_processor,
#                                     user_lat=user_lat, user_lon=user_lon)

# # Step 8: Generate Chart Data
# violation_counts = df[['size_violation', 'geo_violation', 'structural_violation', 'content_violation']].sum()

# chart_config = {
#     "type": "bar",
#     "data": {
#         "labels": ["size", "geo", "structural", "content"],
#         "datasets": [{
#             "label": "Violation Counts",
#             "data": [
#                 int(violation_counts['size_violation']),
#                 int(violation_counts['geo_violation']),
#                 int(violation_counts['structural_violation']),
#                 int(violation_counts['content_violation'])
#             ],
#             "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"],
#             "borderColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"],
#             "borderWidth": 1
#         }]
#     },
#     "options": {
#         "scales": {
#             "y": {
#                 "beginAtZero": True,
#                 "title": {
#                     "display": True,
#                     "text": "Count"
#                 }
#             },
#             "x": {
#                 "title": {
#                     "display": True,
#                     "text": "Violation Type"
#                 }
#             }
#         },
#         "plugins": {
#             "title": {
#                 "display": True,
#                 "text": "Violation Counts in Synthetic Data"
#             },
#             "legend": {
#                 "display": False
#             }
#         }
#     }
# }

# # Step 9: Print Final Result
# print(f"Billboard Status: {result}")
# if result == "violation":
#     for reason in reasons:
#         print(f"- {reason}")

# # Step 10: Visualize Chart
# print("```chartjs\n" + str(chart_config) + "\n```")



# import cv2
# import numpy as np
# import os

# def check_violations(file_path):
#     violations = []
    
#     # Load image (works for video first frame too)
#     ext = os.path.splitext(file_path)[-1].lower()
#     is_video = ext in [".mp4", ".avi", ".mov"]

#     if is_video:
#         cap = cv2.VideoCapture(file_path)
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Error reading video")
#             return
#         img = frame
#         cap.release()
#     else:
#         img = cv2.imread()
#         if img is None:
#             print("❌ Error reading image")
#             return

#     height, width, _ = img.shape
#     print(f"Detected Billboard Dimensions → Width: {width}px, Height: {height}px")

#     # ---------- RULE 1: Size ----------
#     if width > 1000 or height > 800:
#         violations.append("Billboard too large (size violation)")

#     # ---------- RULE 2: Tilt Detection ----------
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         cnt = max(contours, key=cv2.contourArea)
#         rect = cv2.minAreaRect(cnt)  # gives (center, (w,h), angle)
#         (cx, cy), (w, h), angle = rect

#         # Normalize angle
#         if angle < -45:
#             angle = 90 + angle

#         print(f"Detected Tilt Angle: {angle:.2f} degrees")

#         if abs(angle) > 15:  # Dummy rule: more than 15° is violation
#             violations.append(f"Billboard tilted by {angle:.2f}° (structural hazard)")

#     # ---------- RULE 3: Age (Dummy) ----------
#     # We use image contrast/edges as a proxy for wear/tear
#     blur = cv2.Laplacian(gray, cv2.CV_64F).var()
#     if blur < 50:  
#         violations.append("Billboard appears old/faded (possible age issue)")

#     # ---------- RULE 4: Content Violation (Dummy) ----------
#     if "alcohol" in file_path.lower():
#         violations.append("Content violation (alcohol detected)")
#     if "nudity" in file_path.lower():
#         violations.append("Content violation (nudity detected)")

#     # ---------- OUTPUT ----------
#     if violations:
#         print("🚨 Violations Found:")
#         for v in violations:
#             print(" -", v)
#     else:
#         print("✅ Non-Violating Billboard")


# # ---------- Example Run ----------
# check_violations("billboard.jpg")  
# # check_violations("billboard_video.mp4")  # For video input


# app.py (same as ml_logic.py)
import cv2
import numpy as np
import os
from typing import Dict, Any

# Remove Tesseract usage (we'll use EasyOCR only)
TESS_AVAILABLE = False

# Optional EasyOCR (fallback/augment OCR)
try:
    import easyocr  # type: ignore
    EASY_AVAILABLE = True
    _EASY_READER = None  # lazy-init
except Exception:
    EASY_AVAILABLE = False
    _EASY_READER = None

# Optional EXIF altitude support
try:
    from exif import Image as ExifImage  # type: ignore
    EXIF_AVAILABLE = True
except Exception:
    EXIF_AVAILABLE = False

# Optional YOLO (Ultralytics) for person/car detection (scale reference)
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
    _YOLO_MODEL = None  # lazy-init
except Exception:
    YOLO_AVAILABLE = False
    _YOLO_MODEL = None

CFG = {
    "max_w": 1000,          # simple demo size rule
    "max_h": 800,
    "tilt_violation_deg": 25.0,  # relaxed threshold, combine with robust angle estimation
    "blur_var_thresh": 50.0,
    "output_dir": "uploads",  # annotated outputs go here
    
    # Size calibration and meter limits
    "px_to_meter": 0.01,     # 1000 px ≈ 10 m => 0.01 m/px (heuristic)
    "max_w_m": 8.0,          # width must not exceed 8 meters
    "max_h_m": 5.0,          # height must not exceed 5 meters
    
    # New features configuration
    "min_distance_from_road": 50,      # meters from road center
    "min_distance_from_intersection": 100,  # meters from intersection
    "max_height_from_ground": 20,      # meters above ground (not used for violations now)
    "min_distance_from_residential": 200,   # meters from residential areas
    "max_distance_from_permitted_location": 500,  # meters from permitted location
    "zoning_violation_distance": 100,  # meters from non-commercial zones
}

# --- Lightweight feedback model persistence (learn meters-per-pixel) ---
def _get_learned_m_per_px_fallback() -> float:
    """Static fallback: return configuration px_to_meter."""
    return float(CFG["px_to_meter"])

def _ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _annotate_tilt(img, rect):
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    (cx, cy), (w, h), angle = rect
    cv2.putText(img, f"tilt:{angle:.1f} deg",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return img

def _get_exif_altitude_m(file_path: str):
    """Return EXIF GPS altitude in meters if present, else None."""
    if not EXIF_AVAILABLE:
        return None
    try:
        with open(file_path, 'rb') as f:
            exif_img = ExifImage(f)
        if not getattr(exif_img, 'has_exif', False):
            return None
        # EXIF altitude may be a rational; convert to float meters
        if hasattr(exif_img, 'gps_altitude'):
            alt = exif_img.gps_altitude
            try:
                return float(alt)
            except Exception:
                return None
        return None
    except Exception:
        return None

def _load_yolo():
    global _YOLO_MODEL
    if not YOLO_AVAILABLE:
        return None
    if _YOLO_MODEL is None:
        try:
            # Use a tiny model for speed; falls back to default if local pt is unavailable
            _YOLO_MODEL = YOLO('yolov8n.pt')
        except Exception:
            _YOLO_MODEL = None
    return _YOLO_MODEL

def _detect_reference_scale(img_bgr):
    """Detect person or car to estimate meters-per-pixel.
    Returns (m_per_px, ref_label, ref_px_height). None if unavailable.
    """
    model = _load_yolo()
    if model is None:
        return None, None, None
    try:
        results = model.predict(img_bgr, verbose=False)
        m_per_px = None
        ref_label = None
        ref_px = 0
        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            for b in r.boxes:
                try:
                    cls = int(b.cls)
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    hpx = max(0.0, y2 - y1)
                except Exception:
                    continue
                # COCO class ids: 0=person, 2=car
                if cls == 0 and hpx > ref_px:
                    m_per_px = 1.7 / max(hpx, 1e-6)
                    ref_label = 'person'
                    ref_px = hpx
                elif cls == 2 and ref_label is None and hpx > 0:
                    # fallback if no person; approximate 1.5m for car body height
                    m_per_px = 1.5 / max(hpx, 1e-6)
                    ref_label = 'car'
                    ref_px = hpx
        return m_per_px, ref_label, ref_px
    except Exception:
        return None, None, None

def _anonymize_people(img_bgr):
    """Blur detected persons in the image. Returns (anonymized_img, num_people)."""
    model = _load_yolo()
    if model is None:
        return img_bgr, 0
    try:
        results = model.predict(img_bgr, verbose=False)
        people = []
        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue
            for b in r.boxes:
                try:
                    cls = int(b.cls)
                    if cls != 0:  # 0 = person
                        continue
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    people.append((max(0, x1), max(0, y1), max(0, x2), max(0, y2)))
                except Exception:
                    continue
        if not people:
            return img_bgr, 0
        anon = img_bgr.copy()
        for (x1, y1, x2, y2) in people:
            roi = anon[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # Apply strong blur to the ROI
            k = max(15, (max(y2 - y1, x2 - x1) // 10) * 2 + 1)
            roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
            anon[y1:y2, x1:x2] = roi_blur
        return anon, len(people)
    except Exception:
        return img_bgr, 0

def _estimate_ground_y(img_gray):
    """Estimate ground line y using edges + Hough on bottom quarter; fallback to image bottom."""
    h, w = img_gray.shape[:2]
    roi = img_gray[int(h*0.75):, :]
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=int(w*0.3), maxLineGap=15)
    ground_y = h - 1
    if lines is not None and len(lines) > 0:
        # Choose the lowest nearly horizontal line
        best = h - 1
        for l in lines[:,0,:]:
            x1, y1, x2, y2 = l
            if abs(y2 - y1) <= 10:  # horizontal
                y = int(min(y1, y2) + h*0.75)
                if y > best:
                    best = y
        ground_y = best
    return ground_y

def check_violations(file_path: str) -> Dict[str, Any]:
    """
    Pure function (no prints). Returns a JSON-serializable dict.
    - Handles image and video (uses first frame for video).
    - Draws annotated preview if tilt rect found.
    """
    result: Dict[str, Any] = {
        "input_file": os.path.basename(file_path),
        "media_type": "image",
        "width": None,
        "height": None,
        "tilt_angle_deg": None,
        "blur_var": None,
        "violations": [],
        "verdict": "Non-Violating",
        "annotated_path": None,
        "notes": []
    }

    ext = os.path.splitext(file_path)[-1].lower()
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    result["media_type"] = "video" if is_video else "image"

    # Read frame
    try:
        if is_video:
            cap = cv2.VideoCapture(file_path)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                result["notes"].append("Unable to read video")
                result["verdict"] = "Non-Violating"
                return result
            img = frame
        else:
            img = cv2.imread(file_path)
            if img is None:
                result["notes"].append("Unable to read image")
                return result
    except Exception as e:
        result["notes"].append(f"Media reading failed: {str(e)}")
        return result

    h, w = img.shape[:2]
    result["width"] = w
    result["height"] = h

    # Rule 1: Size (in meters). Actual billboard size computed after tilt step.

    # Rule 2: Tilt (improved detection for billboard-like shapes)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use multiple edge detection and morphological close to stabilize rectangles
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        edges = cv2.bitwise_or(edges1, edges2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours with better filtering
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = None
        
        if cnts:
            # Filter contours by area and aspect ratio to find billboard-like shapes
            valid_cnts = []
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area > 1500:  # Slightly higher minimum area threshold
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = max(w, h) / min(w, h)
                    if aspect_ratio > 1.3:  # More rectangular constraint
                        valid_cnts.append(cnt)
            
            if valid_cnts:
                # Use the largest valid contour
                cnt = max(valid_cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (bw, bh), angle = rect
                
                # Normalize angle to be between -45 and 45 degrees using min-area rect
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                
                result["tilt_angle_deg"] = float(angle)
                
                # Only flag as violation if significantly tilted AND large enough to be a real billboard
                if abs(angle) > CFG["tilt_violation_deg"] and cv2.contourArea(cnt) > 7000:
                    result["violations"].append(f"Billboard tilted by {angle:.2f}° (> {CFG['tilt_violation_deg']}°)")
                # Do not add tilt notes; only record violations
                    
    except Exception as e:
        rect = None

    # Rule 3: Aging proxy via blur/texture
    try:
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        result["blur_var"] = float(blur)
        if blur < CFG["blur_var_thresh"]:
            result["violations"].append("Billboard appears old/faded (low Laplacian variance)")
    except Exception as e:
        result["blur_var"] = 0.0
        result["notes"].append(f"Blur analysis failed: {str(e)}")

    # Rule 4: Content rule (filename + OCR if available)
    fname = os.path.basename(file_path).lower()
    filename_alcohol_keywords = [
        "alcohol", "liquor", "spirits", "beer", "lager", "ale", "ipa",
        "stout", "pilsner", "wine", "winery", "whisky", "whiskey",
        "scotch", "bourbon", "vodka", "rum", "gin", "tequila",
        "brandy", "cognac"
    ]
    if any(k in fname for k in filename_alcohol_keywords):
        result["violations"].append("Content violation (alcohol)")
    if "nudity" in fname:
        result["violations"].append("Content violation (nudity)")

    # OCR-based keyword scan (works even if filename doesn't contain keywords)
    try:
        if TESS_AVAILABLE:
            # Tesseract disabled in this build; skip
            pass
        # If no Tesseract or no hit, try EasyOCR as fallback/augment
        if EASY_AVAILABLE:
            global _EASY_READER
            if _EASY_READER is None:
                try:
                    _EASY_READER = easyocr.Reader(['en'], gpu=False)
                except Exception:
                    _EASY_READER = None
            if _EASY_READER is not None:
                # EasyOCR expects RGB image arrays
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                eo_results = _EASY_READER.readtext(image_rgb, detail=1)
                detected_lines = []
                hits = []
                for item in eo_results:
                    try:
                        bbox, txt, conf = item
                    except ValueError:
                        # Some versions return (bbox, txt)
                        bbox, txt = item[0], item[1]
                        conf = 1.0
                    if conf is None:
                        conf = 0.0
                    if conf < 0.3:
                        continue
                    detected_lines.append(txt)
                    tlow = str(txt).lower()
                    alcohol_keywords = [
                        "alcohol", "liquor", "spirits", "bar", "pub", "brewery",
                        "beer", "lager", "ale", "ipa", "stout", "pilsner",
                        "wine", "winery", "red wine", "white wine", "rose",
                        "whisky", "whiskey", "scotch", "bourbon", "single malt",
                        "vodka", "rum", "gin", "tequila", "brandy", "cognac"
                    ]
                    if any(k in tlow for k in alcohol_keywords):
                        hits.append(txt)
                if hits:
                    result["violations"].append("Content violation (alcohol-related text detected via EasyOCR)")
                if detected_lines:
                    preview_eo = " ".join(detected_lines)[:120]
                    result["notes"].append(f"EasyOCR preview: '{preview_eo}...'")
        else:
            result["notes"].append("OCR not available; install Tesseract+pytesseract or easyocr for text-based content checks")
    except Exception as e:
        result["notes"].append(f"OCR content analysis failed: {str(e)}")

    # NEW FEATURES: Advanced violation detection
    # Note: These will be populated when coordinates are provided
    result["geolocation_violations"] = []
    result["zoning_violations"] = []
    result["permitted_location_violations"] = []
    result["placement_score"] = 100  # 100 = perfect placement, 0 = worst

    # Compute billboard size & ground clearance using scale if available
    try:
        width_m = None
        height_m = None
        clearance_m = None
        # Scale from YOLO (person/car)
        m_per_px, ref_label, ref_px = _detect_reference_scale(img)
        if rect is not None:
            # rect provides (bw, bh) in pixels
            (_, _), (bw, bh), _ = rect
            # Prefer learned scale if available; else heuristic px_to_meter
            if m_per_px is not None:
                width_m = float(bw) * m_per_px
                height_m = float(bh) * m_per_px
                # Estimate ground line and billboard bottom y
                gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ground_y = _estimate_ground_y(gray2)
                # billboard bottom from rect box
                box = cv2.boxPoints(rect).astype(np.int32)
                bottom_y = int(max(pt[1] for pt in box))
                clearance_px = max(0, ground_y - bottom_y)
                clearance_m = clearance_px * m_per_px
            else:
                learned = _get_learned_m_per_px_fallback()
                width_m = float(bw) * learned
                height_m = float(bh) * learned
        else:
            # Fallback estimate from full image bounds
            if m_per_px is not None:
                width_m = float(w) * m_per_px
                height_m = float(h) * m_per_px
            else:
                learned = _get_learned_m_per_px_fallback()
                width_m = float(w) * learned
                height_m = float(h) * learned
            

        result["width_m"] = width_m
        result["height_m"] = height_m
        if clearance_m is not None:
            result["clearance_m"] = float(clearance_m)
        # expose scale used for potential feedback
        # do not export scale for learning when admin approves

        # Check meter-based limits
        exceeded = []
        if width_m is not None and width_m > CFG["max_w_m"]:
            exceeded.append(f"width {width_m:.2f}m > {CFG['max_w_m']}m")
        if height_m is not None and height_m > CFG["max_h_m"]:
            exceeded.append(f"height {height_m:.2f}m > {CFG['max_h_m']}m")
        if exceeded:
            result["violations"].append("Size violation (meters): " + ", ".join(exceeded))
        # Clearance rule: < 3m from ground is violation (only if we computed clearance)
        if clearance_m is not None and clearance_m < 3.0:
            result["violations"].append(f"Low clearance: {clearance_m:.2f}m (< 3.0m)")
    except Exception as e:
        result["notes"].append(f"Meter size computation failed: {str(e)}")

    # Annotated preview (if rect found)
    if rect is not None:
        try:
            # Anonymize people before drawing annotations
            annotated, num_people = _anonymize_people(img)
            annotated = _annotate_tilt(annotated, rect)
            _ensure_dir(CFG["output_dir"])
            out_name = f"annotated_{os.path.splitext(os.path.basename(file_path))[0]}.jpg"
            out_path = os.path.join(CFG["output_dir"], out_name)
            cv2.imwrite(out_path, annotated)
            result["annotated_path"] = out_path
            if num_people > 0:
                result["notes"].append(f"Anonymized {num_people} person(s) in preview")
        except Exception as e:
            result["notes"].append(f"Annotation failed: {str(e)}")
    else:
        pass

    # Final verdict
    if result["violations"]:
        result["verdict"] = "Violating"
    else:
        result["verdict"] = "Non-Violating"

    return result


def check_geolocation_violations(lat: float, long: float, file_path: str | None = None) -> Dict[str, Any]:
    """
    Check for poor placement violations based on coordinates
    """
    violations = []
    placement_score = 100
    
    # Simulate distance calculations (in production, use actual GIS data)
    # For demo purposes, we'll use some mock data
    
    # Check distance from road (simulated)
    road_distance = abs(lat - 40.7128) * 111000  # Rough conversion to meters
    if road_distance < CFG["min_distance_from_road"]:
        violations.append(f"Too close to road: {road_distance:.1f}m (< {CFG['min_distance_from_road']}m)")
        placement_score -= 20
    
    # Check distance from intersection (simulated)
    intersection_distance = abs(long - (-74.0060)) * 111000
    if intersection_distance < CFG["min_distance_from_intersection"]:
        violations.append(f"Too close to intersection: {intersection_distance:.1f}m (< {CFG['min_distance_from_intersection']}m)")
        placement_score -= 25
    
    # Height-from-ground note (no violation here; clearance violation handled by image-based pipeline)
    exif_alt_m = _get_exif_altitude_m(file_path) if file_path else None
    height_from_ground_m = exif_alt_m if exif_alt_m is not None else None
    
    # Check residential area proximity (simulated)
    residential_distance = abs(lat - 40.7589) * 111000  # Times Square area
    if residential_distance < CFG["min_distance_from_residential"]:
        violations.append(f"Too close to residential area: {residential_distance:.1f}m (< {CFG['min_distance_from_residential']}m)")
        placement_score -= 30
    
    return {
        "violations": violations,
        "placement_score": max(0, placement_score),
        "road_distance": road_distance,
        "intersection_distance": intersection_distance,
        "height_from_ground_m": height_from_ground_m,
        "residential_distance": residential_distance
    }


def check_zoning_violations(lat: float, long: float) -> Dict[str, Any]:
    """
    Check for zoning violations based on coordinates
    """
    violations = []
    
    # Simulate zoning data (in production, integrate with city GIS)
    # For demo, we'll use coordinate-based mock zoning
    
    # Commercial zones (allowed)
    commercial_zones = [
        (40.7128, -74.0060),  # Downtown Manhattan
        (40.7589, -73.9851),  # Times Square
        (40.7505, -73.9934),  # Penn Station
    ]
    
    # Check if location is in commercial zone
    in_commercial_zone = False
    min_distance_to_commercial = float('inf')
    
    for cz_lat, cz_long in commercial_zones:
        distance = ((lat - cz_lat) ** 2 + (long - cz_long) ** 2) ** 0.5 * 111000
        min_distance_to_commercial = min(min_distance_to_commercial, distance)
        if distance < CFG["zoning_violation_distance"]:
            in_commercial_zone = True
            break
    
    if not in_commercial_zone:
        violations.append(f"Not in commercial zone (closest: {min_distance_to_commercial:.1f}m)")
    
    # Check for specific restricted zones
    restricted_zones = [
        (40.7484, -73.9857),  # Grand Central Terminal (historic)
        (40.7527, -73.9772),  # UN Headquarters
    ]
    
    for rz_lat, rz_long in restricted_zones:
        distance = ((lat - rz_lat) ** 2 + (long - rz_long) ** 2) ** 0.5 * 111000
        if distance < 500:  # 500m buffer
            violations.append(f"Too close to restricted zone: {distance:.1f}m")
    
    return {
        "violations": violations,
        "in_commercial_zone": in_commercial_zone,
        "distance_to_commercial": min_distance_to_commercial,
        "zoning_status": "Commercial" if in_commercial_zone else "Non-Commercial"
    }


def check_permitted_database(lat: float, long: float) -> Dict[str, Any]:
    """
    Check against permitted billboard database
    """
    violations = []
    
    # Simulate permitted billboard database (in production, use real database)
    permitted_billboards = [
        {"id": 1, "lat": 40.7128, "long": -74.0060, "permit_number": "BB001", "expiry": "2025-12-31"},
        {"id": 2, "lat": 40.7589, "long": -73.9851, "permit_number": "BB002", "expiry": "2025-12-31"},
        {"id": 3, "lat": 40.7505, "long": -73.9934, "permit_number": "BB003", "expiry": "2025-12-31"},
    ]
    
    # Check if location is too close to existing permitted billboard
    too_close_to_permitted = False
    closest_permitted_distance = float('inf')
    closest_permitted = None
    
    for billboard in permitted_billboards:
        distance = ((lat - billboard["lat"]) ** 2 + (long - billboard["long"]) ** 2) ** 0.5 * 111000
        if distance < CFG["max_distance_from_permitted_location"]:
            too_close_to_permitted = True
            if distance < closest_permitted_distance:
                closest_permitted_distance = distance
                closest_permitted = billboard
    
    if too_close_to_permitted:
        violations.append(f"Too close to permitted billboard #{closest_permitted['id']}: {closest_permitted_distance:.1f}m")
    
    # Check if location already has a permit
    has_existing_permit = False
    for billboard in permitted_billboards:
        distance = ((lat - billboard["lat"]) ** 2 + (long - billboard["long"]) ** 2) ** 0.5 * 111000
        if distance < 50:  # 50m threshold for same location
            has_existing_permit = True
            violations.append(f"Location already has permit: {billboard['permit_number']}")
            break
    
    return {
        "violations": violations,
        "too_close_to_permitted": too_close_to_permitted,
        "closest_permitted_distance": closest_permitted_distance,
        "closest_permitted": closest_permitted,
        "has_existing_permit": has_existing_permit,
        "permit_status": "Permitted" if has_existing_permit else "Not Permitted"
    }


def check_all_violations_with_coordinates(file_path: str, lat: float, long: float) -> Dict[str, Any]:
    """
    Enhanced violation checker that includes all new features
    """
    # Get basic violations first
    basic_result = check_violations(file_path)
    
    # Add coordinate-based violations
    geolocation_result = check_geolocation_violations(lat, long)
    # Poor placement via excessive tilt (>25°): use computed tilt from basic_result
    try:
        tilt = basic_result.get("tilt_angle_deg")
        if tilt is not None and abs(float(tilt)) > 25.0:
            geolocation_result["violations"].append(
                f"Poor placement: excessive tilt {float(tilt):.2f}° (> 25°)"
            )
    except Exception:
        pass
    permitted_result = check_permitted_database(lat, long)
    
    # Combine all violations
    all_violations = basic_result["violations"].copy()
    all_violations.extend(geolocation_result["violations"])
    all_violations.extend(permitted_result["violations"])
    
    # Update result with new data
    result = basic_result.copy()
    result["violations"] = all_violations
    result["geolocation_violations"] = geolocation_result["violations"]
    result["permitted_location_violations"] = permitted_result["violations"]
    result["placement_score"] = geolocation_result["placement_score"]
    result["permit_status"] = permitted_result["permit_status"]
    
    # Update verdict
    if all_violations:
        result["verdict"] = "Violating"
    else:
        result["verdict"] = "Non-Violating"
    
    return result
