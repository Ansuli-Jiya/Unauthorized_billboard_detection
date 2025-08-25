# from flask import Flask, render_template, request, jsonify
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import numpy as np
# from PIL import Image
# import os
# import logging
# from exif import Image as ExifImage
# from ultralytics import YOLO
# from torchvision import models, transforms
# from transformers import CLIPProcessor, CLIPModel
# import cv2
# import sqlite3
# from datetime import datetime
# import shutil

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
# app.config['MEDIA_FOLDER'] = os.path.join(os.path.dirname(__file__), 'media')
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['MEDIA_FOLDER'], exist_ok=True)

# # Database setup
# conn = sqlite3.connect('cases.db', check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS cases (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         media_path TEXT,
#         lat REAL,
#         long REAL,
#         reasons TEXT,
#         user_name TEXT,
#         user_email TEXT,
#         status TEXT DEFAULT 'pending',
#         submitted_at TIMESTAMP
#     )
# ''')
# conn.commit()

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load scaler and feature columns from synthetic data
# df = pd.read_csv('synthetic_billboard_dataset.csv')
# feature_columns = ['width_m', 'height_m', 'lat', 'long', 'age_years', 'installation_score', 'placement_score', 'violence_prob', 'explicit_prob']
# X = df[feature_columns]
# scaler = StandardScaler()
# scaler.fit(X)

# # Load models
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

# model = ViolationModel(input_size=len(feature_columns), output_size=4)
# model.load_state_dict(torch.load('violation_model.pth', weights_only=True))
# model.eval()

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

# # Define functions
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

# # def process_frame(img, yolo_model, resnet_model, transform, clip_model, clip_processor):
# #     img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# #     results = yolo_model(img_cv)

# #     billboard_box = None
# #     max_area = 0
# #     for result in results:
# #         for box in result.boxes:
# #             if box.cls == 0:  # Adjust class ID for billboard
# #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
# #                 area = (x2 - x1) * (y2 - y1)
# #                 if area > max_area:
# #                     max_area = area
# #                     billboard_box = (x1, y1, x2, y2)

# #     if billboard_box is None:
# #         logger.warning("No billboard detected in image")
# #         return None

# #     x1, y1, x2, y2 = billboard_box
# #     width_px = x2 - x1
# #     height_px = y2 - y1
# #     logger.info(f"Detected Billboard: {width_px}x{height_px}px")

# #     calib_factor = 10.0 / 1000.0
# #     width_m = width_px * calib_factor
# #     height_m = height_px * calib_factor

# #     violence_prob, explicit_prob = 0.0, 0.0
# #     if clip_model and clip_processor:
# #         inputs = clip_processor(text=["violent content", "explicit content", "neutral content"], images=img, return_tensors="pt", padding=True)
# #         with torch.no_grad():
# #             outputs = clip_model(**inputs)
# #         probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
# #         violence_prob, explicit_prob = probs[0], probs[1]
# #     else:
# #         img_transformed = transform(img).unsqueeze(0)
# #         with torch.no_grad():
# #             resnet_features = resnet_model(img_transformed).numpy().flatten()
# #         violence_prob = resnet_features[0] / max(resnet_features)
# #         explicit_prob = resnet_features[1] / max(resnet_features)

# #     img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
# #     edges = cv2.Canny(img_gray, 100, 200)
# #     edge_density = np.sum(edges) / (img_gray.shape[0] * img_gray.shape[1])
# #     age_years = min(max(edge_density * 50, 0), 10)
# #     installation_score = max(100 - edge_density * 100, 50)

# #     img_array = np.array(img)
# #     placement_score = 0.9 - (np.var(img_array) / 255**2) * 0.4
# #     placement_score = max(min(placement_score, 1.0), 0.5)

# #     return [width_m, height_m, violence_prob, explicit_prob, age_years, installation_score, placement_score]

# # def process_frame(img, yolo_model, resnet_model, transform, clip_model, clip_processor):
# #     # Convert PIL image to OpenCV format for YOLO
# #     img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# #     logger.info(f"Processing frame with shape: {img_cv.shape}")
# #     results = yolo_model(img_cv)  # Detect billboards

# #     # Assume the largest detected object is the billboard
# #     billboard_box = None
# #     max_area = 0
# #     for result in results:
# #         for box in result.boxes:
# #             logger.info(f"Detected box: cls={box.cls}, xyxy={box.xyxy[0].cpu().numpy()}")
# #             if box.cls == 0:  # Adjust class ID for billboard (fine-tune YOLO for billboards)
# #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
# #                 area = (x2 - x1) * (y2 - y1)
# #                 logger.info(f"Box area: {area}")
# #                 if area > max_area:
# #                     max_area = area
# #                     billboard_box = (x1, y1, x2, y2)

# #     if billboard_box is None:
# #         logger.warning("No billboard detected in image")
# #         return None

# #     x1, y1, x2, y2 = billboard_box
# #     width_px = x2 - x1
# #     height_px = y2 - y1
# #     logger.info(f"Detected Billboard: {width_px}x{height_px}px")

# #     # Size calibration (assume 1000px = 10m; adjust based on real-world data)
# #     calib_factor = 10.0 / 1000.0
# #     width_m = width_px * calib_factor
# #     height_m = height_px * calib_factor
# #     logger.info(f"Converted to meters: width={width_m:.2f}m, height={height_m:.2f}m")

# #     # Content analysis with CLIP
# #     violence_prob, explicit_prob = 0.0, 0.0
# #     if clip_model and clip_processor:
# #         inputs = clip_processor(text=["violent content", "explicit content", "neutral content"], images=img, return_tensors="pt", padding=True)
# #         with torch.no_grad():
# #             outputs = clip_model(**inputs)
# #         probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
# #         violence_prob, explicit_prob = probs[0], probs[1]
# #     else:
# #         # Fallback to ResNet
# #         img_transformed = transform(img).unsqueeze(0)
# #         with torch.no_grad():
# #             resnet_features = resnet_model(img_transformed).numpy().flatten()
# #         violence_prob = resnet_features[0] / max(resnet_features)
# #         explicit_prob = resnet_features[1] / max(resnet_features)

# #     # Structural analysis
# #     img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
# #     edges = cv2.Canny(img_gray, 100, 200)
# #     edge_density = np.sum(edges) / (img_gray.shape[0] * img_gray.shape[1])
# #     age_years = min(max(edge_density * 50, 0), 10)  # Heuristic: high edges = older
# #     installation_score = max(100 - edge_density * 100, 50)  # Heuristic: high edges = poor installation

# #     # Placement score
# #     img_array = np.array(img)
# #     placement_score = 0.9 - (np.var(img_array) / 255**2) * 0.4
# #     placement_score = max(min(placement_score, 1.0), 0.5)

# #     return [width_m, height_m, violence_prob, explicit_prob, age_years, installation_score, placement_score]

# def process_frame(img, yolo_model, resnet_model, transform, clip_model, clip_processor):
#     # # Convert PIL image to OpenCV format for YOLO
#     # img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     # logger.info(f"Processing frame with shape: {img_cv.shape}")
#     # results = yolo_model(img_cv)  # Detect objects

# #    yolo_model = YOLO('runs/detect/train10/weights/best.pt')  # Update 'train10' if different

# #     # Convert PIL image to OpenCV format
# #     img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# #     results = yolo_model(img_cv)
    
#     yolo_model = YOLO('runs/detect/train10/weights/best.pt')  # Update 'train10' if different

#     # Convert PIL image to OpenCV format
#     img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     results = yolo_model(img_cv)

#     # # Extract the largest billboard detection
#     # billboard_box = None
#     # max_area = 0
#     # for result in results:
#     #     for box in result.boxes:
#     #         cls = int(box.cls[0])  # Class 0 = billboard
#     #         if cls == 0:  # Ensure it's a billboard
#     #             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#     #             area = (x2 - x1) * (y2 - y1)
#     #             if area > max_area:
# #                 max_area = area
# #                 billboard_box = (x1, y1, x2, y2)

# #     # Temporary workaround: Use the largest box of any class if no class 0 is found
#     if billboard_box is None:
#         logger.warning("No billboard (class 0) detected, using largest object as fallback")
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 area = (x2 - x1) * (y2 - y1)
#                 if area > max_area:
#                     max_area = area
#                     billboard_box = (x1, y1, x2, y2)

#     if billboard_box is None:
#         logger.error("No objects detected in image")
#         return None

#     x1, y1, x2, y2 = billboard_box
#     width_px = x2 - x1
#     height_px = y2 - y1
#     logger.info(f"Detected object (fallback): {width_px}x{height_px}px")

#     # Size calibration (assume 1000px = 10m; adjust based on real-world data)
#     calib_factor = 10.0 / 1000.0
#     width_m = width_px * calib_factor
#     height_m = height_px * calib_factor
#     logger.info(f"Converted to meters: width={width_m:.2f}m, height={height_m:.2f}m")

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
#         'width_m': 0.0, 'height_m': 0.0, 'lat': user_lat, 'long': user_lon,
#         'age_years': 0.0, 'installation_score': 0.0, 'placement_score': 0.0,
#         'violence_prob': 0.0, 'explicit_prob': 0.0
#     }

#     yolo_model = YOLO('yolov8n.pt')

#     frame_features = []
#     if is_video or media_path.lower().endswith(('.mp4', '.avi', '.mov')):
#         cap = cv2.VideoCapture(media_path)
#         if not cap.isOpened():
#             logger.error(f"Could not open video file: {media_path}")
#             return features
#         frame_count = 0
#         while cap.isOpened() and frame_count < 5:  # Limit to 5 frames
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             frame_result = process_frame(img, yolo_model, resnet_model, transform, clip_model, clip_processor)
#             if frame_result:
#                 frame_features.append(frame_result)
#             frame_count += 1
#         cap.release()
#     else:
#         # Handle all image formats
#         try:
#             img = Image.open(media_path).convert('RGB')
#             # Verify the image can be processed
#             img.verify()  # Check if the file is a valid image
#             frame_result = process_frame(img, yolo_model, resnet_model, transform, clip_model, clip_processor)
#             if frame_result:
#                 frame_features.append(frame_result)
#         except (IOError, SyntaxError, ValueError) as e:
#             logger.error(f"Error processing image {media_path}: {e}")
#             return features  # Return default features if image fails

#     if frame_features:
#         avg_features = np.mean(frame_features, axis=0)
#         features.update({
#             'width_m': avg_features[0], 'height_m': avg_features[1],
#             'violence_prob': avg_features[2], 'explicit_prob': avg_features[3],
#             'age_years': avg_features[4], 'installation_score': avg_features[5],
#             'placement_score': avg_features[6]
#         })

#     # Ensure lat and long are set, using user-provided values as fallback
#     if features['lat'] is None:
#         lat, lon = extract_exif_geo(media_path) if not is_video else (None, None)
#         features['lat'] = user_lat if user_lat is not None else lat if lat is not None else 0.0
#         features['long'] = user_lon if user_lon is not None else lon if lon is not None else 0.0

#     logger.info(f"Extracted Features for {media_path}: {features}")
#     return features

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

# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'media' not in request.files:
#         return jsonify({'error': 'No media provided'}), 400
    
#     media = request.files['media']
#     lat = float(request.form.get('lat', 0.0))
#     lon = float(request.form.get('long', 0.0))
    
#     if media.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     # Save media temporarily
#     temp_path = os.path.join(app.config['UPLOAD_FOLDER'], media.filename)
#     media.save(temp_path)
#     logger.info(f"Saved temp file at: {temp_path}")
    
#     # Determine if it's a video
#     is_video = temp_path.lower().endswith(('.mp4', '.avi', '.mov'))
    
#     try:
#         features = extract_features_from_media(temp_path, resnet, transform, clip_model, clip_processor, is_video=is_video, user_lat=lat, user_lon=lon)
#         logger.info(f"Extracted features: {features}")
        
#         X_extracted = pd.DataFrame([features])[feature_columns]
#         X_scaled = scaler.transform(X_extracted)
#         X_torch = torch.tensor(X_scaled, dtype=torch.float32)
#         with torch.no_grad():
#             ml_preds = model(X_torch).numpy().flatten() > 0.5
        
#         size_v, size_reason = check_size(features['width_m'], features['height_m'])
#         geo_v, geo_reason = check_geo(features['lat'], features['long'], features['placement_score'])
#         structural_v, structural_reason = check_structural(features['age_years'], features['installation_score'])
#         content_v, content_reason = check_content(features['violence_prob'], features['explicit_prob'])
        
#         reasons = [r for r in [size_reason, geo_reason, structural_reason, content_reason] if r and "skipped" not in r]
#         result = "violation" if reasons else "non-violating"
        
#         logger.info(f"Detection result: {result}, reasons: {reasons}, media_path: {temp_path}")
#         return jsonify({
#             'result': result,
#             'reasons': reasons,
#             'media_path': temp_path,
#             'width_m': features['width_m'],
#             'height_m': features['height_m']
#         })
#     except Exception as e:
#         logger.error(f"Error in detection: {e}")
#         return jsonify({'error': str(e)}), 500
#     finally:
#         pass

# # @app.route('/detect', methods=['POST'])
# # def detect():
# #     if 'media' not in request.files:
# #         return jsonify({'error': 'No media provided'}), 400
    
# #     media = request.files['media']
# #     lat = float(request.form.get('lat', 0.0))
# #     lon = float(request.form.get('long', 0.0))
    
# #     if media.filename == '':
# #         return jsonify({'error': 'No selected file'}), 400
    
# #     # Save media temporarily
# #     temp_path = os.path.join(app.config['UPLOAD_FOLDER'], media.filename)
# #     media.save(temp_path)
    
# #     try:
# #         # Detect violations (update to handle video if needed)
# #         features = extract_features_from_media(temp_path, resnet, transform, clip_model, clip_processor, is_video=temp_path.endswith(('.mp4', '.avi')), user_lat=lat, user_lon=lon)
        
# #         X_extracted = pd.DataFrame([features])[feature_columns]
# #         X_scaled = scaler.transform(X_extracted)
# #         X_torch = torch.tensor(X_scaled, dtype=torch.float32)
# #         with torch.no_grad():
# #             ml_preds = model(X_torch).numpy().flatten() > 0.5
        
# #         size_v, size_reason = check_size(features['width_m'], features['height_m'])
# #         geo_v, geo_reason = check_geo(features['lat'], features['long'], features['placement_score'])
# #         structural_v, structural_reason = check_structural(features['age_years'], features['installation_score'])
# #         content_v, content_reason = check_content(features['violence_prob'], features['explicit_prob'])
        
# #         reasons = [r for r in [size_reason, geo_reason, structural_reason, content_reason] if r and "skipped" not in r]
# #         result = "violation" if reasons else "non-violating"
        
# #         # Return media path for submission (temp for now; will move on submit)
# #         return jsonify({'result': result, 'reasons': reasons, 'media_path': temp_path})
# #     except Exception as e:
# #         logger.error(f"Error in detection: {e}")
# #         return jsonify({'error': str(e)}), 500
# #     finally:
# #         # Do not remove temp_path yet; remove after submission or if no violation
# #         pass

# @app.route('/submit_report', methods=['POST'])
# def submit_report():
#     media_path = request.form.get('mediaPath')
#     lat = request.form.get('lat', '0.0')
#     lon = request.form.get('long', '0.0')
#     reasons = request.form.get('reasons')
#     user_name = request.form.get('userName')
#     user_email = request.form.get('userEmail')
    
#     logger.info(f"Received data: media_path={media_path}, lat={lat}, lon={lon}, reasons={reasons}, user_name={user_name}, user_email={user_email}")
    
#     if not media_path:
#         logger.error("media_path is empty")
#         return jsonify({'error': 'No media path provided'}), 400
#     if not os.path.exists(media_path):
#         logger.error(f"media_path does not exist: {media_path}")
#         return jsonify({'error': 'Media file missing or invalid path'}), 400
    
#     try:
#         lat = float(lat)
#         lon = float(lon)
#     except ValueError:
#         logger.error(f"Invalid lat/lon: lat={lat}, lon={lon}")
#         return jsonify({'error': 'Invalid latitude or longitude'}), 400
    
#     # Move media to permanent storage
#     permanent_path = os.path.join(app.config['MEDIA_FOLDER'], os.path.basename(media_path))
#     shutil.move(media_path, permanent_path)
    
#     # Save to database
#     cursor.execute('''
#         INSERT INTO cases (media_path, lat, long, reasons, user_name, user_email, submitted_at)
#         VALUES (?, ?, ?, ?, ?, ?, ?)
#     ''', (permanent_path, lat, lon, reasons, user_name, user_email, datetime.now()))
#     conn.commit()
#     case_id = cursor.lastrowid
    
#     logger.info(f"Case submitted with ID: {case_id}")
#     return jsonify({'case_id': case_id})

# @app.route('/admin', methods=['GET', 'POST'])
# def admin():
#     if request.method == 'POST':
#         case_id = request.form.get('case_id')
#         new_status = request.form.get('status')
#         cursor.execute('UPDATE cases SET status = ? WHERE id = ?', (new_status, case_id))
#         conn.commit()
    
#     cursor.execute('SELECT * FROM cases ORDER BY submitted_at DESC')
#     cases = cursor.fetchall()
#     return render_template('admin.html', cases=cases)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import sqlite3
from app import check_violations, check_geolocation_violations, check_permitted_database, CFG
import json
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Health check endpoint for Render
@app.route('/health')
def health():
    return 'ok', 200

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# SQLite database setup
DB_PATH = 'database.db'

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                userId TEXT NOT NULL,
                input_file TEXT NOT NULL,
                lat REAL NOT NULL,
                long REAL NOT NULL,
                violations TEXT NOT NULL,
                status TEXT NOT NULL,
                annotated_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        # Ensure created_at exists for older tables
        try:
            cursor.execute("PRAGMA table_info(reports)")
            cols = [r[1] for r in cursor.fetchall()]
            if 'created_at' not in cols:
                cursor.execute('ALTER TABLE reports ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP')
                conn.commit()
        except Exception:
            pass

def _reports_has_created_at() -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(reports)")
            cols = [r[1] for r in cur.fetchall()]
            return 'created_at' in cols
    except Exception:
        return False

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/check-violations', methods=['POST'])
def check_violations_endpoint():
    try:
        if 'file' not in request.files or 'lat' not in request.form or 'long' not in request.form:
            return jsonify({'error': 'Missing file or coordinates'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get coordinates from form
        lat = float(request.form['lat'])
        long = float(request.form['long'])
        
        # Process with ml_logic
        result = check_violations(file_path)
        
        # Add geolocation analysis if coordinates are provided
        if lat and long:
            geo_result = check_geolocation_violations(lat, long, file_path)
            placement_issues = geo_result.get('violations', [])

            # Add tilt/broken (blur) analysis into Placement Issues as requested
            try:
                tilt = result.get('tilt_angle_deg')
                if tilt is not None and abs(float(tilt)) > float(CFG.get('tilt_violation_deg', 25.0)):
                    placement_issues.append(f"Poor placement: excessive tilt {float(tilt):.2f}° (> {CFG.get('tilt_violation_deg', 25.0)}°)")
            except Exception:
                pass
            try:
                blur_var = result.get('blur_var')
                if blur_var is not None and float(blur_var) < float(CFG.get('blur_var_thresh', 50.0)):
                    placement_issues.append("Possible broken/old display (low texture/blur analysis)")
            except Exception:
                pass

            result['geolocation_violations'] = placement_issues
            result['placement_score'] = geo_result.get('placement_score')
            
            # Add permit checks (zoning removed per requirements)
            permit_result = check_permitted_database(lat, long)
            result['permitted_location_violations'] = permit_result.get('violations', [])
            result['permit_status'] = permit_result.get('permit_status')

            # Merge all violations so they appear in the main list
            merged_extra = []
            merged_extra.extend(result['geolocation_violations'])
            merged_extra.extend(result['permitted_location_violations'])
            if merged_extra:
                # Ensure base list exists
                if not isinstance(result.get('violations'), list):
                    result['violations'] = []
                result['violations'].extend(merged_extra)
                # Update verdict if any violations
                if result['violations']:
                    result['verdict'] = 'Violating'

        # Update annotated_path to be URL-accessible
        if result['annotated_path']:
            result['annotated_path'] = f'/uploads/{os.path.basename(result["annotated_path"])}'

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/submit-report', methods=['POST'])
def submit_report():
    if not all(key in request.form for key in ['userId', 'lat', 'long', 'violations', 'status']):
        return jsonify({'error': 'Missing required fields'}), 400

    user_id = request.form['userId']
    lat = float(request.form['lat'])
    long = float(request.form['long'])
    violations = request.form['violations']
    status = request.form['status']
    
    # Get the filename from the mediaPath (which contains the filename)
    media_path = request.form.get('mediaPath', '')
    if not media_path:
        return jsonify({'error': 'No media path provided'}), 400
    
    filename = os.path.basename(media_path)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({'error': 'Media file not found'}), 400
    
    # Get annotated path from check_violations
    result = check_violations(file_path)
    annotated_path = f'/uploads/{os.path.basename(result["annotated_path"])}' if result['annotated_path'] else None

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if _reports_has_created_at():
            cursor.execute('''
                INSERT INTO reports (userId, input_file, lat, long, violations, status, annotated_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, filename, lat, long, violations, status, annotated_path, datetime.utcnow().isoformat()))
        else:
            cursor.execute('''
                INSERT INTO reports (userId, input_file, lat, long, violations, status, annotated_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, filename, lat, long, violations, status, annotated_path))
        conn.commit()

    return jsonify({'message': 'Report submitted successfully'})

@app.route('/api/user-reports/<userId>', methods=['GET'])
def get_user_reports(userId):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM reports WHERE userId = ?', (userId,))
        rows = cursor.fetchall()
        reports = []
        for row in rows:
            created = row[8] if len(row) > 8 else None
            reports.append({
                'id': row[0],
                'userId': row[1],
                'input_file': row[2],
                'lat': row[3],
                'long': row[4],
                'violations': json.loads(row[5]),
                'status': row[6],
                'annotated_path': row[7],
                'created_at': created
            })
        return jsonify(reports)

@app.route('/api/admin-reports', methods=['GET'])
def get_admin_reports():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM reports')
        rows = cursor.fetchall()
        reports = []
        for row in rows:
            created = row[8] if len(row) > 8 else None
            reports.append({
                'id': row[0],
                'userId': row[1],
                'input_file': row[2],
                'lat': row[3],
                'long': row[4],
                'violations': json.loads(row[5]),
                'status': row[6],
                'annotated_path': row[7],
                'created_at': created
            })
        return jsonify(reports)

@app.route('/api/admin-reports/<int:reportId>', methods=['PATCH'])
def update_report_status(reportId):
    data = request.get_json()
    if 'status' not in data:
        return jsonify({'error': 'Missing status'}), 400

    status = data['status']
    if status not in ['Pending', 'Approved', 'Rejected']:
        return jsonify({'error': 'Invalid status'}), 400

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Update status
        cursor.execute('UPDATE reports SET status = ? WHERE id = ?', (status, reportId))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({'error': 'Report not found'}), 404

        # If approved, log to feedback csv (no model learning)
        if status == 'Approved':
            cursor.execute('SELECT id, userId, input_file, lat, long, violations, status, annotated_path FROM reports WHERE id = ?', (reportId,))
            row = cursor.fetchone()
            if row:
                try:
                    feedback_dir = 'feedback'
                    if not os.path.exists(feedback_dir):
                        os.makedirs(feedback_dir)
                    feedback_csv = os.path.join(feedback_dir, 'approved_feedback.csv')
                    # Prepare CSV line
                    import csv, datetime
                    write_header = not os.path.exists(feedback_csv)
                    with open(feedback_csv, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(['id','userId','input_file','lat','long','violations_json','approved_at'])
                        writer.writerow([
                            row[0], row[1], row[2], row[3], row[4], row[5], datetime.datetime.utcnow().isoformat()
                        ])
                except Exception:
                    # Do not fail the API if logging fails
                    pass

    return jsonify({'message': 'Status updated successfully'})

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
