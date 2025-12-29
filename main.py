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
import random
import string

app = FastAPI()

# --- TRICK: NEW DATABASE NAME ---
# Changing this name forces a fresh start without paying for Shell access!
DB_NAME = "verify_shield_v2.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create the NEW table structure
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (email TEXT PRIMARY KEY, 
                  name TEXT, 
                  password TEXT, 
                  verification_code TEXT,
                  is_verified INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

init_db() # Run immediately on startup

# --- MODELS ---
class UserSignup(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserVerification(BaseModel):
    email: str
    code: str

# --- HELPERS ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_code():
    return ''.join(random.choices(string.digits, k=6))

# --- USER ENDPOINTS ---

@app.post("/signup")
async def signup(user: UserSignup):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Check if email exists
        if c.execute("SELECT * FROM users WHERE email=?", (user.email,)).fetchone():
            return JSONResponse(content={"status": "error", "message": "Email already registered"}, status_code=400)
        
        v_code = generate_code()
        hashed_pass = hash_password(user.password)
        
        # Save user as Unverified
        c.execute("INSERT INTO users (email, name, password, verification_code, is_verified) VALUES (?, ?, ?, ?, 0)", 
                  (user.email, user.name, hashed_pass, v_code))
        conn.commit()

        # --- SIMULATE EMAIL (PRINT TO LOGS) ---
        print(f"\n[EMAIL SIMULATION] To: {user.email}")
        print(f"[EMAIL SIMULATION] Code: {v_code}\n")
        
        return {"status": "success", "message": "Verification code sent! Check server logs."}
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    finally:
        conn.close()

@app.post("/verify")
async def verify(data: UserVerification):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        user = c.execute("SELECT verification_code FROM users WHERE email=?", (data.email,)).fetchone()
        if not user:
            return JSONResponse(content={"status": "error", "message": "User not found"}, status_code=404)
        
        if user[0] == data.code:
            c.execute("UPDATE users SET is_verified=1, verification_code=NULL WHERE email=?", (data.email,))
            conn.commit()
            return {"status": "success", "message": "Account verified!"}
        else:
            return JSONResponse(content={"status": "error", "message": "Invalid code"}, status_code=400)
    finally:
        conn.close()

@app.post("/login")
async def login(user: UserLogin):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    row = c.execute("SELECT name, is_verified FROM users WHERE email=? AND password=?", 
                    (user.email, hash_password(user.password))).fetchone()
    conn.close()
    
    if row:
        name, is_verified = row
        if is_verified:
            return {"status": "success", "message": f"Welcome, {name}!", "username": name}
        else:
            return JSONResponse(content={"status": "error", "message": "Account not verified."}, status_code=401)
    else:
        return JSONResponse(content={"status": "error", "message": "Invalid credentials"}, status_code=401)

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
    
    if noise_score < 200: return True, 0.1, f"Suspiciously smooth (Score: {int(noise_score)}). Likely AI."
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
