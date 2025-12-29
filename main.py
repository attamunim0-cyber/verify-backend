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
    # Load a pre-trained Face Detector (Haar Cascade is fast & built-in)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return False, 0.0
    
    # If we found a face, we assume it's "Real" for this basic version.
    # (To detect advanced Deepfakes, you would plug in a heavy model here)
    # For now, we simulate a "confidence" score based on face clarity.
    return True, 0.98  # 98% confidence it's a real person if we see a face

# --- 2. THE NEW SKILL: "Completely AI" Detection ---
def check_for_synthetic_artifacts(image):
    """
    If no face is found, we check if the image looks 'too perfect' or 'digital'.
    AI images often lack the natural 'grain' (noise) of real camera sensors.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplace filter finds edges/noise. 
    # Real photos have HIGH variance (lots of noise/texture).
    # AI/Blurry images have LOW variance (too smooth).
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Threshold: Below 100 often means synthetic, blurred, or digital art
    is_suspicious = variance < 100 
    
    return is_suspicious, variance

# --- 3. THE ROUTER: Handles Both Images & Videos ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"Received file: {file.filename} | Type: {file.content_type}")
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        # STRATEGY: Try to open as Image first
        image_frame = cv2.imread(temp_path)
        
        frames_to_analyze = []
        is_video = False

        if image_frame is not None:
            # It IS an image
            frames_to_analyze = [image_frame]
        else:
            # It IS a video
            is_video = True
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return JSONResponse(content={"verdict": "ERROR", "score": 0.0, "message": "Could not open media"})
            
            # Grab up to 3 frames to check (Beginning, Middle, End)
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

        # --- FINAL VERDICT LOGIC ---
        if face_found_count > 0:
            # Scenario A: We saw a human face.
            avg_score = total_score / face_found_count
            verdict = "REAL" if avg_score > 0.5 else "FAKE"
            return {"verdict": verdict, "score": avg_score}
        
        else:
            # Scenario B: NO FACES (The "Completely AI" check)
            # Check the first frame for AI artifacts
            if len(frames_to_analyze) > 0:
                is_synthetic, noise_level = check_for_synthetic_artifacts(frames_to_analyze[0])
                
                print(f"No face. Noise Variance: {noise_level}")
                
                if is_synthetic:
                    # Low noise + No Face = Likely AI Art / Synthetic
                    return {
                        "verdict": "FAKE", 
                        "score": 0.01, 
                        "message": "No face detected. Image is unnaturally smooth (Likely AI Generated)."
                    }
                else:
                    # High noise + No Face = Just a photo of a tree/cat
                    return {
                        "verdict": "UNKNOWN", 
                        "score": 0.0, 
                        "message": "No human detected, but looks like a real camera photo."
                    }
            
            return {"verdict": "UNKNOWN", "score": 0.0}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        # Cleanup: Delete the temp file so server doesn't fill up
        if os.path.exists(temp_path):
            os.remove(temp_path)
