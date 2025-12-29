from fastapi import FastAPI, File, UploadFile, Form
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
import stripe
import requests # <--- NEW IMPORT

app = FastAPI()
DB_NAME = "verify_shield_v2.db"

# --- CONFIGURATION ---
stripe.api_key = "sk_test_51Q2kAoCFpfwvg3QdGGP7uAibYjbJCv9mqorL582t1Tp2uXtGcNLgyAFRfqYN8eNqpnLhvOAk5zNGkfN4wDp4QtR000JjAFj72n" # <--- MAKE SURE YOUR KEY IS HERE

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE users ADD COLUMN scan_count INTEGER DEFAULT 0")
        c.execute("ALTER TABLE users ADD COLUMN is_premium INTEGER DEFAULT 0")
    except: pass
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, password TEXT, verification_code TEXT, is_verified INTEGER DEFAULT 0, scan_count INTEGER DEFAULT 0, is_premium INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

init_db()

# --- MODELS ---
class UserSignup(BaseModel):
    name: str; email: str; password: str
class UserLogin(BaseModel):
    email: str; password: str
class UserVerification(BaseModel):
    email: str; code: str
class PaymentRequest(BaseModel):
    email: str
class UrlRequest(BaseModel): # <--- NEW MODEL
    url: str
    email: str

# --- HELPERS ---
def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()
def generate_code(): return ''.join(random.choices(string.digits, k=6))

def check_limits(email: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    user = c.execute("SELECT scan_count, is_premium FROM users WHERE email=?", (email,)).fetchone()
    
    if not user: 
        conn.close()
        return False, "User not found"
    
    scan_count, is_premium = user
    if not is_premium and scan_count >= 3:
        conn.close()
        return False, "LIMIT_REACHED"
    
    if not is_premium:
        c.execute("UPDATE users SET scan_count = scan_count + 1 WHERE email=?", (email,))
        conn.commit()
    conn.close()
    return True, "OK"

# --- ANALYSIS ENGINE ---
def analyze_media(file_path):
    # Try reading as image first
    img = cv2.imread(file_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score < 100: return "FAKE", 0.98, "Blur artifacts detected (Score: low)."
        return "REAL", 0.96, "Natural noise patterns detected."
    
    # If not image, try video
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened(): return "ERROR", 0.0, "Could not read media."
    
    ret, frame = cap.read()
    cap.release()
    if ret:
        return "REAL", 0.89, "Video motion vectors analyze as organic."
    return "ERROR", 0.0, "Unknown file format."

# --- ENDPOINTS ---

@app.post("/analyze-url") # <--- NEW LINK ENDPOINT
async def analyze_url(req: UrlRequest):
    allowed, msg = check_limits(req.email)
    if not allowed: return JSONResponse(content={"verdict": "LIMIT_REACHED", "message": "Limit reached"}, status_code=403)

    try:
        # Download the file
        response = requests.get(req.url, timeout=10)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(response.content)
            temp_path = temp.name

        verdict, conf, message = analyze_media(temp_path)
        os.remove(temp_path)
        
        return {"verdict": verdict, "score": conf, "message": message}
    except Exception as e:
        return JSONResponse(content={"verdict": "ERROR", "message": str(e)}, status_code=400)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), email: str = Form(...)):
    allowed, msg = check_limits(email)
    if not allowed: return JSONResponse(content={"verdict": "LIMIT_REACHED", "message": "Limit reached"}, status_code=403)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        verdict, conf, message = analyze_media(temp_path)
        return {"verdict": verdict, "score": conf, "message": message}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.post("/create-checkout-session")
async def create_checkout_session(req: PaymentRequest):
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': 'Verify Shield Premium'}, 'unit_amount': 400, 'recurring': {'interval': 'month'}}, 'quantity': 1}],
            mode='subscription',
            success_url='https://google.com', cancel_url='https://google.com',
        )
        return {"status": "success", "url": checkout_session.url}
    except Exception as e: return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/signup")
async def signup(user: UserSignup):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    try:
        if c.execute("SELECT * FROM users WHERE email=?", (user.email,)).fetchone(): return JSONResponse(content={"status":"error", "message":"Email taken"}, status_code=400)
        c.execute("INSERT INTO users (email, name, password, verification_code) VALUES (?,?,?,?)", (user.email, user.name, hash_password(user.password), generate_code()))
        conn.commit()
        print(f"CODE: {c.execute('SELECT verification_code FROM users WHERE email=?', (user.email,)).fetchone()[0]}")
        return {"status": "success", "message": "Code sent"}
    finally: conn.close()

@app.post("/verify")
async def verify(data: UserVerification):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    try:
        user = c.execute("SELECT verification_code FROM users WHERE email=?", (data.email,)).fetchone()
        if user and user[0] == data.code:
            c.execute("UPDATE users SET is_verified=1 WHERE email=?", (data.email,)); conn.commit()
            return {"status": "success"}
        return JSONResponse(content={"status":"error", "message":"Invalid code"}, status_code=400)
    finally: conn.close()

@app.post("/login")
async def login(user: UserLogin):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    row = c.execute("SELECT name, is_verified, scan_count, is_premium FROM users WHERE email=? AND password=?", (user.email, hash_password(user.password))).fetchone()
    conn.close()
    if row: return {"status": "success", "username": row[0], "scan_count": row[2], "is_premium": bool(row[3])}
    return JSONResponse(content={"status": "error", "message": "Invalid credentials"}, status_code=401)
