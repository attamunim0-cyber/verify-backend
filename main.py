from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
import os
import shutil

app = FastAPI()

def get_variance(image):
    """
    Standardizes image and checks for 'Digital Smoothness'.
    Real cameras have high noise (variance > 300).
    AI generators are often too perfect (variance < 300).
    """
    # 1. Resize to standard width (640px) so 4K images don't throw us off
    height, width = image.shape[:2]
    target_width = 640
    scale = target_width / width
    dim = (target_width, int(height * scale))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. Calculate Laplacian Variance (The "Crunchiness" Score)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def analyze_frame(frame):
    # Load Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Check for Faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Calculate Global "Crunchiness" (Noise)
    noise_score = get_variance(frame)
    print(f"Noise Score: {noise_score}") # Check logs to see this number!

    # --- STRICTER RULES ---
    
    # Rule 1: The "Too Smooth" Trap
    # We raised the threshold from 100 to 300. 
    # Most real phone videos are 400-800. AI is usually 50-250.
    if noise_score < 300:
        return True, 0.1, f"Suspiciously smooth (Score: {int(noise_score)}). Likely AI."

    # Rule 2: The "Ghost" Trap (No Face)
    if len(faces) == 0:
        # If it's noisy but has no face, it's just 'Unknown' or 'Fake' depending on your preference.
        # Let's call it Fake for safety if you only expect human videos.
        return True, 0.3, "No human face detected."

    # Rule 3: Face Exists + Good Noise = Probably Real
    return False, 0.98, f"Natural noise detected (Score: {int(noise_score)})."

# --- ROUTER ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        # Try to open as image first
        image_frame = cv2.imread(temp_path)
        frames_to_analyze = []

        if image_frame is not None:
            frames_to_analyze = [image_frame]
        else:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return JSONResponse(content={"verdict": "ERROR", "score": 0.0, "message": "Could not open media"})
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Check 3 random frames
            for i in [0, total_frames // 2, total_frames - 5]:
                if i < 0: continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames_to_analyze.append(frame)
            cap.release()

        # --- VOTING SYSTEM ---
        fake_votes = 0
        real_votes = 0
        final_message = ""
        
        for frame in frames_to_analyze:
            is_fake, score, msg = analyze_frame(frame)
            if is_fake:
                fake_votes += 1
                final_message = msg
            else:
                real_votes += 1
                final_message = msg

        # --- FINAL VERDICT ---
        # If even ONE frame is suspicious, we flag it. (Paranoid Mode)
        if fake_votes > 0:
            return {"verdict": "FAKE", "score": 0.15, "message": final_message}
        else:
            return {"verdict": "REAL", "score": 0.96, "message": final_message}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
