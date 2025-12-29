from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import tempfile
import os
import shutil
import sqlite3
import hashlib

app = FastAPI()
DB_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

init_db()

class UserCredentials(BaseModel):
    username: str
    password: str = None # Optional for Google Login

class GoogleLoginRequest(BaseModel):
    email: str

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- STANDARD LOGIN ---
@app.post("/register")
async def register(user: UserCredentials):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        if c.execute("SELECT * FROM users WHERE username=?", (user.username,)).fetchone():
            return JSONResponse(content={"status": "error", "message": "Username taken"}, status_code=400)
        c.execute("INSERT INTO users VALUES (?, ?)", (user.username, hash_password(user.password)))
        conn.commit()
        return {"status": "success", "message": "Created!"}
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    finally:
        conn.close()

@app.post("/login")
async def login(user: UserCredentials):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    result = c.execute("SELECT * FROM users WHERE username=? AND password=?", (user.username, hash_password(user.password))).fetchone()
    conn.close()
    if result: return {"status": "success", "message": "Login successful"}
    else: return JSONResponse(content={"status": "error", "message": "Invalid credentials"}, status_code=401)

# --- NEW: GOOGLE LOGIN ENDPOINT ---
@app.post("/google_login")
async def google_login(request: GoogleLoginRequest):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Check if this email exists
        user = c.execute("SELECT * FROM users WHERE username=?", (request.email,)).fetchone()
        
        if user:
            # User exists, log them in
            return {"status": "success", "message": "Welcome back!"}
        else:
            # New user! Create account automatically with a dummy password
            c.execute("INSERT INTO users VALUES (?, ?)", (request.email, "GOOGLE_AUTH_USER"))
            conn.commit()
            return {"status": "success", "message": "Account created via Google!"}
    finally:
        conn.close()

# --- SCANNER LOGIC ---
def get_variance(image):
    height, width = image.shape[:2]
    target_width = 640
    scale = target_width / width
    dim = (target_width, int(height * scale))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_frame(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    noise_score = get_variance(frame)
    
    if noise_score < 300: return True, 0.1, f"Suspiciously smooth (Score: {int(noise_score)}). Likely AI."
    if len(faces) == 0: return True, 0.3, "No human face detected."
    return False, 0.98, f"Natural noise detected (Score: {int(noise_score)})."

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    try:
        image_frame = cv2.imread(temp_path)
        frames_to_analyze = [image_frame] if image_frame is not None else []
        
        if not frames_to_analyze:
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in [0, total_frames // 2, total_frames - 5]:
                if i < 0: continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret: frames_to_analyze.append(frame)
            cap.release()

        fake_votes = 0
        final_msg = ""
        for frame in frames_to_analyze:
            is_fake, score, msg = analyze_frame(frame)
            if is_fake: fake_votes += 1; final_msg = msg
            else: final_msg = msg

        if fake_votes > 0: return {"verdict": "FAKE", "score": 0.15, "message": final_msg}
        else: return {"verdict": "REAL", "score": 0.96, "message": final_msg}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
