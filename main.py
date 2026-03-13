from fastapi import FastAPI, File, UploadFile
import shutil
import os
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_actual_architecture(seq_len=20, img_size=128):
    # Load with 'imagenet' weights initially to ensure the structure 
    # and layer naming match exactly what was in Colab.
    base_cnn = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        pooling='avg', 
        input_shape=(img_size, img_size, 3)
    )
    
    # We build the sequential wrapper
    model = models.Sequential([
        layers.Input(shape=(seq_len, img_size, img_size, 3)),
        layers.TimeDistributed(base_cnn),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
# Initialize the "Skeleton"
model = build_actual_architecture()

# Load only the "Brains" (the learned weights)
# Ensure 'bilstm_model_1.h5' is in your Mac folder
model.load_weights('bilstm_model_1.keras')

print("✅ MobileNet-BiLSTM Architecture rebuilt and weights loaded!")

# 2. Preprocessing Function 
def preprocess_video(video_path, target_frames=20, img_size=128):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // target_frames)
    frames = []
    
    for i in range(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame / 255.0)
        else:
            frames.append(np.zeros((img_size, img_size, 3)))
    cap.release()
    return np.expand_dims(np.array(frames), axis=0) # Add Batch Dim

@app.post("/predict")
async def predict_violence(file: UploadFile = File(...)):
    # Save the uploaded video temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess and Predict
        input_data = preprocess_video(temp_path)
        prediction = model.predict(input_data)[0][0]
        
        # Logic: 0 = Violence, 1 = Non-Violence 
        is_violent = bool(prediction < 0.5)
        confidence = float(1 - prediction if is_violent else prediction)
        
        return {
            "filename": file.filename,
            "is_violent": is_violent,
            "confidence": f"{confidence:.2%}",
            "status": "Alert" if is_violent else "Safe"
        }
    
    finally:
        # Cleanup: Delete the temp video file
        if os.path.exists(temp_path):
            os.remove(temp_path)