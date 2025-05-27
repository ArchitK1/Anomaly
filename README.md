# Anomaly Detection System

This project implements a video-based anomaly detection system using a Long-term Recurrent Convolutional Network (LRCN) model. The system can analyze video footage and detect potential anomalous behavior, with real-time notifications via email for high-probability anomalies.

## Project Structure

```
your_project/
│
├── app.py                      # Flask backend
├── shoplifting_model.py        # Model implementation
├── requirements.txt            # Python dependencies
├── lrcn_160S_90_90Q.h5        # Trained LRCN model file
│
├── templates/                  # Frontend templates
│   └── index.html             
│
├── static/                     # Static files
│   ├── uploads/               # Uploaded videos
│   └── outputs/               # Processed videos
│
├── data/                      # Data directory
│   ├── input/                # Original input videos
│   └── output/               # Reference outputs
│
├── notebooks/                 # Jupyter notebooks
│   └── run.ipynb             # Development notebook
│
└── models/                    # Model storage
    └── lrcn_160S_90_90Q.h5   # Current model
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the web interface at `http://localhost:5000`

## Features

- Video upload and processing interface
- Real-time anomaly detection
- Email notifications for high-probability anomalies
- Video visualization with detection results

## Model Details

The system uses a Long-term Recurrent Convolutional Network (LRCN) model trained on video sequences. The model processes video frames in sequences of 160 frames, with each frame resized to 90x90 pixels.

## Note

Make sure to update the email configuration in `shoplifting_model.py` with your own email credentials for notifications. 