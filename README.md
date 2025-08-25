# Billboard Violation Detection System

A comprehensive web application for detecting and reporting billboard violations using computer vision and machine learning techniques.

## 🚀 Features

### Core Detection Capabilities
- **Size Violations**: Detects billboards exceeding maximum dimensions (8m width × 5m height)
- **Tilt Detection**: Identifies billboards tilted beyond 25° threshold
- **Content Analysis**: OCR-based detection of prohibited content (e.g., alcohol advertisements)
- **Height from Ground**: Analyzes billboard elevation using EXIF GPS data
- **Geolocation Validation**: Checks placement against zoning regulations and permitted locations
- **Privacy Protection**: Automatically blurs detected persons in annotated images

### User Interface
- **User Portal**: Upload images/videos, detect violations, submit reports
- **Admin Portal**: Review, manage, and update violation report statuses
- **Report Tracking**: Users can view their submitted reports and current status
- **Mobile-Friendly**: Responsive design with camera capture support

### Technical Features
- **Multi-format Support**: Handles images (JPEG, PNG) and videos
- **Real-time Processing**: Fast violation detection using YOLOv8 and OpenCV
- **Database Management**: SQLite backend for storing violation reports
- **API-driven**: RESTful endpoints for frontend-backend communication

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework for API endpoints and server logic
- **OpenCV**: Computer vision for image processing and analysis
- **YOLOv8**: Object detection for billboard and person identification
- **EasyOCR**: Optical Character Recognition for text content analysis
- **NumPy**: Numerical operations and array processing
- **SQLite**: Local database for report storage

### Frontend
- **HTML5/CSS3**: Modern, responsive web interface
- **JavaScript/jQuery**: Dynamic interactions and AJAX calls
- **Bootstrap**: UI framework for consistent styling
- **Font Awesome**: Icon library for enhanced user experience

### Deployment
- **Docker**: Containerization for consistent deployment
- **Gunicorn**: WSGI server for production deployment
- **Render**: Cloud platform hosting

## 📋 Prerequisites

- Python 3.8+
- OpenCV dependencies
- CUDA-compatible GPU (optional, for faster processing)

## 🚀 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd detector
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app_flask.py
   ```

5. **Access the application**
   - User Portal: http://localhost:5000
   - Admin Portal: http://localhost:5000/admin

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t billboard-detector .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:8000 billboard-detector
   ```

## 📁 Project Structure

```
detector/
├── app_flask.py              # Flask application entry point
├── app.py                    # Core violation detection logic
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── render.yaml              # Render deployment config
├── templates/               # HTML templates
│   ├── index.html          # User portal
│   └── admin.html          # Admin portal
├── uploads/                 # User uploaded media
├── train/                   # Training dataset
├── val/                     # Validation dataset
├── runs/                    # YOLO training outputs
└── media/                   # Sample media files
```

## 🔧 Configuration

### Environment Variables
- `YOLO_CONFIG_DIR`: Directory for YOLO configuration files
- `FLASK_ENV`: Flask environment (development/production)

### Detection Parameters
Key configuration values in `app.py`:
- `max_w_m`: Maximum billboard width (8 meters)
- `max_h_m`: Maximum billboard height (5 meters)
- `tilt_violation_deg`: Tilt threshold (25 degrees)
- `min_distance_from_road`: Minimum distance from road
- `max_height_from_ground`: Maximum height from ground level

## 📊 API Endpoints

### Core Endpoints
- `GET /`: User portal homepage
- `GET /admin`: Admin portal
- `GET /health`: Health check endpoint

### API Endpoints
- `POST /api/check-violations`: Analyze uploaded media for violations
- `POST /api/submit-report`: Submit violation report
- `GET /api/admin-reports`: Fetch all reports (admin)
- `GET /api/user-reports/<userId>`: Fetch user-specific reports

## 🎯 Usage

### For Users
1. **Upload Media**: Select image or video file
2. **Set Location**: Enter coordinates or use auto-detection
3. **Detect Violations**: System analyzes media for rule violations
4. **Submit Report**: Fill form and submit violation report
5. **Track Status**: View report status in "My Reports" tab

### For Administrators
1. **Review Reports**: View all submitted violation reports
2. **Update Status**: Change report status (Pending, Approved, Rejected)
3. **View Details**: Access detailed violation information and media
4. **Manage Database**: Monitor system usage and report statistics

## 🔍 Violation Detection Rules

### Size Regulations
- **Width**: Maximum 8 meters
- **Height**: Maximum 5 meters
- **Measurement**: Uses pixel-to-meter calibration with YOLO detection

### Placement Standards
- **Tilt**: Maximum 25° from vertical
- **Height from Ground**: Analyzed via EXIF GPS altitude
- **Distance from Road**: Minimum safe distance requirements
- **Zoning Compliance**: Commercial vs. residential area validation

### Content Restrictions
- **Prohibited Content**: Alcohol, tobacco, adult content detection
- **OCR Analysis**: Uses EasyOCR for text recognition
- **Keyword Filtering**: Automated content violation identification

### Location Validation
- **Permit Database**: Cross-reference with authorized billboard locations
- **Geolocation**: GPS coordinate validation
- **Zoning Laws**: Area-specific regulation compliance

## Workflow 

1. On clicking Detect Violation , it will detect using the logic and it will generate this
2. And then you can submit the report to authorities
3. Then it will be visible to the authorities on admin-portal
4. Also there is feature where you can see advance violance analysis of your report 

## 🚀 Deployment

### Render Platform
1. **Connect Repository**: Link your GitHub repository
2. **Configure Service**: Set as web service with Docker
3. **Environment Variables**: Configure production settings
4. **Deploy**: Automatic deployment on code push

### Docker Configuration
- **Base Image**: Python 3.9-slim
- **System Dependencies**: OpenCV libraries
- **Port**: 8000 (internal), configurable via $PORT
- **Health Check**: /health endpoint for monitoring

## 🐛 Troubleshooting

### Common Issues
1. **Port Already in Use**: Ensure no other service uses port 5000
2. **Memory Issues**: Reduce Gunicorn workers for low-memory environments
3. **OpenCV Errors**: Verify system dependencies are installed
4. **YOLO Warnings**: Set YOLO_CONFIG_DIR environment variable

### Debug Mode
- Enable Flask debug mode for detailed error messages
- Check console logs for backend errors
- Use browser DevTools for frontend debugging

## 📈 Performance

### Optimization Tips
- **GPU Acceleration**: Use CUDA-compatible GPU for faster processing
- **Image Resizing**: Large images are automatically resized for processing
- **Caching**: YOLO models are cached after first load
- **Background Processing**: Long operations run asynchronously

### Resource Requirements
- **Minimum RAM**: 2GB
- **Recommended RAM**: 4GB+
- **Storage**: 1GB+ for models and uploads
- **CPU**: Multi-core processor recommended

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Object detection framework
- **OpenCV**: Computer vision library
- **EasyOCR**: OCR functionality
- **Flask**: Web framework
- **Bootstrap**: UI framework


