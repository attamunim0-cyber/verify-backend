from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
import os
import shutil

app = FastAPI()

# --- 1. THE BRAIN: Deepfake Detection Logic ---
def analyze_frame(frame):
    """
    Analyzes a single frame for faces.
    Returns: has_face (bool), score (0.0 to 1.0)
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return False, 0.0
    
    # Found a face -> Assume Real for now (98% confidence)
    return True, 0.98

# --- 2. THE NEW SKILL: "Completely AI" Detection ---
def check_for_synthetic_artifacts(image):
    """
    Checks for 'unnatural smoothness' common in AI generation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplace Variance measures "Texture/Noise".
    # Real photos have High Noise (> 100). AI often has Low Noise (< 100).
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    is_suspicious = variance < 100 
    return is_suspicious, variance

# --- 3. THE ROUTER: Handles Both Images & Videos ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        # Try to read as Image first
        image_frame = cv2.imread(temp_path)
        frames_to_analyze = []

        if image_frame is not None:
            # It IS an image
            frames_to_analyze = [image_frame]
        else:
            # It IS a video
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return JSONResponse(content={"verdict": "ERROR", "score": 0.0, "message": "Could not open media"})
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in [0, total_frames // 2, total_frames - 5]:
                if i < 0: continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames_to_analyze.append(frame)
            cap.release()

        # --- ANALYSIS PHASE ---
        face_found_count = 0
        total_score = 0.0
        
        for frame in frames_to_analyze:
            has_face, score = analyze_frame(frame)
            if has_face:
                face_found_count += 1
                total_score += score

        if face_found_count > 0:
            # Scenario A: Human Face Found
            avg_score = total_score / face_found_count
            verdict = "REAL" if avg_score > 0.5 else "FAKE"
            return {"verdict": verdict, "score": avg_score}
        
        else:
            # Scenario B: No Face (Check for AI Artifacts)
            if len(frames_to_analyze) > 0:
                is_synthetic, noise_level = check_for_synthetic_artifacts(frames_to_analyze[0])
                print(f"No face. Noise Variance: {noise_level}")
                
                if is_synthetic:
                    return {"verdict": "FAKE", "score": 0.01, "message": "No face, unnatural smoothness detected."}
                else:
                    return {"verdict": "UNKNOWN", "score": 0.0, "message": "No human detected."}
            
            return {"verdict": "UNKNOWN", "score": 0.0}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
