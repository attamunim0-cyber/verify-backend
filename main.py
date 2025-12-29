from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
import os
import shutil

app = FastAPI()

def analyze_face_texture(face_img):
    """
    Checks if the face skin is 'too smooth' (AI) or has 'natural noise' (Real).
    Returns: is_fake (bool), texture_score (float)
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the variance of the Laplacian (Measure of focus/texture)
    # Real skin/video has noise/grain (Variance > 150 usually)
    # AI/Deepfake skin is often blurry/smooth (Variance < 100)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # If variance is LOW, it's smooth (Suspicious/AI)
    # If variance is HIGH, it's textured (Real Camera)
    is_fake = variance < 100
    return is_fake, variance

def analyze_frame(frame):
    """
    Detects face AND analyzes its texture.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return False, 0.0, None # No face found
    
    # Get the largest face
    (x, y, w, h) = faces[0]
    face_roi = frame[y:y+h, x:x+w] # Crop the face
    
    # Run the Texture Check on the face
    is_fake, texture_score = analyze_face_texture(face_roi)
    
    if is_fake:
        # Found a face, but it's too smooth!
        return True, 0.15, "Face detected, but skin is unnaturally smooth (AI Signature)."
    else:
        # Found a face, and it has natural grain.
        return True, 0.98, "Face detected with natural camera noise."

# --- ROUTER ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        image_frame = cv2.imread(temp_path)
        frames_to_analyze = []

        if image_frame is not None:
            frames_to_analyze = [image_frame]
        else:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return JSONResponse(content={"verdict": "ERROR", "score": 0.0, "message": "Could not open media"})
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Analyze 3 frames to be sure
            for i in [0, total_frames // 2, total_frames - 5]:
                if i < 0: continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames_to_analyze.append(frame)
            cap.release()

        # --- ANALYSIS PHASE ---
        fake_votes = 0
        real_votes = 0
        message = ""
        
        for frame in frames_to_analyze:
            has_face, score, msg = analyze_frame(frame)
            if has_face:
                if score < 0.5: # Low score means Fake
                    fake_votes += 1
                    message = msg
                else:
                    real_votes += 1
            else:
                # If no face, run general artifact check
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                if variance < 100:
                    fake_votes += 1
                    message = "No face, image too smooth."

        # --- FINAL VERDICT ---
        if fake_votes > real_votes:
            return {"verdict": "FAKE", "score": 0.10, "message": message}
        elif real_votes > 0:
            return {"verdict": "REAL", "score": 0.98, "message": "Natural face detected."}
        else:
            return {"verdict": "UNKNOWN", "score": 0.0, "message": "No clear subject found."}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
