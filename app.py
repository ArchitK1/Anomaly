from flask import Flask, render_template, request, send_file, url_for, jsonify
import os
from shoplifting_model import ShopliftingPrediction

app = Flask(__name__)

# Configure folders with absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lrcn_160S_90_90Q.h5')
DATA_FOLDER = os.path.join(BASE_DIR, 'data', 'input')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

# Ensure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded video
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(input_path)
        
        # Process the video
        output_filename = f'output_{video.filename}'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Initialize and run the model
        model = ShopliftingPrediction(
            model_path=MODEL_PATH,
            frame_width=90,
            frame_height=90,
            sequence_length=160
        )
        model.load_model()
        model.Predict_Video(input_path, output_path)
        
        # Verify the output file exists
        if not os.path.exists(output_path):
            raise Exception("Output video file was not created")
            
        # Get file size for debugging
        file_size = os.path.getsize(output_path)
        print(f"Output video created at: {output_path}")
        print(f"File size: {file_size} bytes")
        
        # Return the video URL for download
        video_url = url_for('static', filename=f'outputs/{output_filename}', _external=True)
        return jsonify({
            'video_url': video_url,
            'file_size': file_size
        })
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 