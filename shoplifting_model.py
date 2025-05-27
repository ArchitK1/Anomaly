import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import math
import smtplib
from email.message import EmailMessage

class ShopliftingPrediction:
    def __init__(self, model_path, frame_width, frame_height, sequence_length):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sequence_length = sequence_length
        self.model_path = model_path
        
    def load_model(self):
        try:
            # Try loading with custom_objects to handle potential compatibility issues
            custom_objects = {
                'Conv2D': keras.layers.Conv2D,
                'LSTM': keras.layers.LSTM,
                'TimeDistributed': keras.layers.TimeDistributed,
                'Dense': keras.layers.Dense
            }
            self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def generate_message_content(self, probability, label):
        if label == 0:  # Anomaly
            if probability <= 75:
                self.message = "There is little chance of anomaly"
            elif probability <= 85:
                self.message = "High probability of anomaly"
            else:
                self.message = "Very high probability of anomaly"
        elif label == 1:  # Normal
            if probability <= 75:
                self.message = "The movement is confusing, watch"
            elif probability <= 85:
                self.message = "I think it's normal, but it's better to watch"
            else:
                self.message = "Movement is normal"
          
    def Pre_Process_Video(self, current_frame, previous_frame):
        diff = cv2.absdiff(current_frame, previous_frame)
        diff = cv2.GaussianBlur(diff,(3,3), 0)
        resized_frame = cv2.resize(diff, (self.frame_height, self.frame_width))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = gray_frame / 255
        return normalized_frame
      
    def Read_Video(self, filePath):
        self.video_reader = cv2.VideoCapture(filePath)
        self.original_video_width = int(self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_video_height = int(self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video_reader.get(cv2.CAP_PROP_FPS)
      
    def Single_Frame_Predict(self, frames_queue):
        frames_queue = np.array(frames_queue)
        frames_queue = np.expand_dims(frames_queue, axis=0)
        probabilities = self.model.predict(frames_queue)[0]
        predicted_label = np.argmax(probabilities)
        probability = math.floor(max(probabilities[0], probabilities[1])*100)
        return [probability, predicted_label]
      
    def send_email_notification(self, message):
        sender_email = "asmit79tyagi@gmail.com"
        receiver_email = "asmit.2125cs1019@kiet.edu"
        app_password = "wgsl lgxc gmwv noin"
        
        msg = EmailMessage()
        msg['Subject'] = 'Urgent: Anomaly Detected'
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg.set_content(message)
        
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, app_password)
            smtp.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email sent successfully!")
    
    def Predict_Video(self, video_file_path, output_file_path):
        message = 'I will start analysing video now'
        self.Read_Video(video_file_path)
        video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                    self.fps, (self.original_video_width, self.original_video_height))
        success, frame = self.video_reader.read()
        previous = frame.copy()
        frames_queue = []
        
        while self.video_reader.isOpened():
            ok, frame = self.video_reader.read()
            if not ok:
                break
                
            normalized_frame = self.Pre_Process_Video(frame, previous)
            previous = frame.copy()
            frames_queue.append(normalized_frame)
                
            if len(frames_queue) == self.sequence_length:
                [probability, predicted_label] = self.Single_Frame_Predict(frames_queue)
                self.generate_message_content(probability, predicted_label)
                message = "{}:{}%".format(self.message, probability)
                frames_queue = []
                print(message)
                
            cv2.rectangle(frame, (0,0), (640, 40), (255, 255, 255), -1)
            cv2.putText(frame, message, (1,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            video_writer.write(frame)
            
        self.video_reader.release()
        video_writer.release()
        
        if probability > 90:
            message = "High anomaly detected in video: "
            self.send_email_notification(message) 