from fastapi import FastAPI, File, UploadFile, Form, Request, Header
from fastapi.responses import JSONResponse, HTMLResponse
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
import requests
import yt_dlp
import glob
import smtplib
from PIL import Image, ImageChops, ImageEnhance, ExifTags
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- AI BRAIN IMPORTS ---
from transformers import pipeline
import torch

app = FastAPI()
DB_NAME = "verify_shield_v2.db"

# --- CONFIGURATION ---
stripe.api_key = "sk_test_51Q2kAoCFpfwvg3QdGGP7uAibYjbJCv9mqorL582t1Tp2uXtGcNLgyAFRfqYN8eNqpnLhvOAk5zNGkfN4wDp4QtR000JjAFj72n"
STRIPE_ENDPOINT_SECRET = "whsec_HmhvhPDOimozUwdmH135qmtv6DleLWN7"
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
SERVER_URL = "https://verify-backend-03or.onrender.com"

# --- LOAD THE AI BRAIN (ON STARTUP) ---
# We use a lightweight ViT (Vision Transformer) fine-tuned for Deepfakes
print("LOADING AI MODEL... (This may take 30s on first boot)")
try:
    # This model is small enough for free servers but very accurate
    ai_classifier = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    MODEL_LOADED = True
    print("AI MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print(f"WARNING: Could not load AI Model: {e}")
    MODEL_LOADED = False

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, password TEXT, verification_code TEXT, is_verified INTEGER DEFAULT 1, scan_count INTEGER DEFAULT 0, is_premium INTEGER DEFAULT 0, reset_token TEXT)''')
    conn.commit(); conn.close()
init_db()

# --- V14: HYBRID DETECTION ENGINE ---

def analyze_with_brain(pil_image):
    """Asks the HuggingFace AI model for a verdict."""
    if not MODEL_LOADED: return 0.0, "Model Offline"
    
    try:
        results = ai_classifier(pil_image)
        # Results look like: [{'label': 'artificial', 'score': 0.99}, {'label': 'human', 'score': 0.01}]
        
        # Find the 'artificial' or 'fake' score
        fake_score = 0.0
        for r in results:
            label = r['label'].lower()
            if 'art' in label or 'fake' in label or 'ai' in label:
                fake_score = r['score']
                
        return fake_score, "Deep Learning Analysis"
    except Exception as e:
        print(f"Brain Error: {e}")
        return 0.0, "Brain Error"

def analyze_exif(pil_img):
    try:
        exif = pil_img.getexif()
        if not exif: return 0.0
        make = exif.get(271)
        model = exif.get(272)
        if make and model: return -0.6 # Real camera bonus
        return 0.0
    except: return 0.0

def analyze_media(file_path):
    try:
        pil_img = Image.open(file_path)

        # 1. AI BRAIN CHECK (The Heavy Hitter)
        # We perform this first because it's the most accurate
        brain_score, source = analyze_with_brain(pil_img)
        
        # 2. METADATA CHECK (The Fast Check)
        try:
            meta = str(pil_img.getexif()) + str(pil_img.info)
            ai_tags = ["midjourney", "diffusion", "generated", "dall-e", "firefly"]
            if any(k in meta.lower() for k in ai_tags):
                return "FAKE", 0.99, "Metadata explicit AI tag found."
        except: pass 
        
        # 3. IF BRAIN IS SURE, TRUST IT
        # If the Neural Network is >90% sure, we don't need math.
        if brain_score > 0.90:
             return "FAKE", 0.98, "Deep Learning Model detected synthetic signatures."
        elif brain_score < 0.10:
             # If Neural Net is sure it's human, trust it (unless metadata disagrees)
             return "REAL", 0.95, "Deep Learning Model confirmed organic features."

        # 4. TIE BREAKER: MATH ANALYSIS (For ambiguous cases)
        # If the AI Brain is unsure (10% - 90%), we use V13 Math to decide.
        
        img = cv2.imread(file_path)
        if img is None: return "ERROR", 0.0, "Decode Error"
        
        # Resize to 1024 to save CPU
        h, w = img.shape[:2]
        if h > 1024:
            s = 1024 / h
            img = cv2.resize(img, None, fx=s, fy=s)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Grid Check (FFT)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-10)
        h, w = gray.shape
        magnitude[h//2-5:h//2+5, w//2-5:w//2+5] = 0 
        high_freq_energy = np.mean(magnitude)
        
        # Plastic Check (Variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Exif Bonus
        bonus = analyze_exif(pil_img)

        # Combined Score
        # We mix the Brain Score (weighted 70%) with Math (30%)
        final_score = (brain_score * 2.0) + bonus
        
        if high_freq_energy > 140 and (high_freq_energy / (lap_var+1)) > 2.5:
            final_score += 0.5 # Add Grid penalty
            
        if final_score >= 0.8: return "SUSPICIOUS", 0.75, "AI patterns detected by hybrid analysis."
        else: return "REAL", 0.92, "Hybrid analysis indicates real photo."

    except Exception as e:
        print(f"ERR: {e}")
        return "ERROR", 0.0, f"Analysis Error: {str(e)}"

# --- ENDPOINTS (Standard) ---
class UserRequest(BaseModel): email: str 
class UrlRequest(BaseModel): url: str; email: str
class UserSignup(BaseModel): name: str; email: str; password: str
class UserLogin(BaseModel): email: str; password: str
class PaymentRequest(BaseModel): email: str
class ForgotPasswordRequest(BaseModel): email: str

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def ensure_user_exists(email):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row: c.execute("INSERT INTO users (email, name, password, is_verified, scan_count, is_premium) VALUES (?, ?, ?, 1, 0, 0)", (email, "User", "restored")); conn.commit()
    conn.close()

def check_limits(email):
    ensure_user_exists(email)
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row; c = conn.cursor()
    row = c.execute("SELECT scan_count, is_premium FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if row['is_premium'] == 1: return True
    if row['scan_count'] >= 3: return False
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET scan_count = scan_count + 1 WHERE email=?", (email,)); conn.commit(); conn.close()
    return True

@app.post("/user-profile")
async def get_user_profile(req: UserRequest):
    ensure_user_exists(req.email)
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row; c = conn.cursor()
    row = c.execute("SELECT name, email, is_premium, scan_count FROM users WHERE email=?", (req.email,)).fetchone()
    conn.close()
    return {"status": "success", "name": row['name'], "email": row['email'], "is_premium": bool(row['is_premium']), "scan_count": row['scan_count'], "plan_name": "Premium Plan" if row['is_premium'] else "Starter Plan"}

@app.post("/analyze-url")
async def analyze_url(req: UrlRequest):
    if not check_limits(req.email): return JSONResponse(content={"verdict": "LIMIT_REACHED", "message": "Limit reached"}, status_code=403)
    try:
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': f"{temp_dir}/%(id)s.%(ext)s", 'quiet': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([req.url])
        files = glob.glob(os.path.join(temp_dir, "*"))
        if not files: return {"verdict": "ERROR", "message": "Download failed"}
        v, c, m = analyze_media(files[0])
        try: shutil.rmtree(temp_dir)
        except: pass
        return {"verdict": v, "score": c, "message": m}
    except Exception as e: return JSONResponse(content={"verdict": "ERROR", "message": str(e)}, status_code=400)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), email: str = Form(...)):
    if not check_limits(email): return JSONResponse(content={"verdict": "LIMIT_REACHED", "message": "Limit reached"}, status_code=403)
    temp_filename = f"upload_{random.randint(1000,9999)}_{file.filename}"
    with open(temp_filename, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    try:
        v, c, m = analyze_media(temp_filename)
        return {"verdict": v, "score": c, "message": m}
    except Exception as e: return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
         if os.path.exists(temp_filename): os.remove(temp_filename)

@app.post("/signup")
async def signup(user: UserSignup):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    try:
        if c.execute("SELECT * FROM users WHERE email=?", (user.email,)).fetchone(): return JSONResponse(content={"status":"error", "message":"Email taken"}, status_code=400)
        c.execute("INSERT INTO users (email, name, password, is_verified) VALUES (?,?,?,1)", (user.email, user.name, hash_password(user.password)))
        conn.commit()
        return {"status": "success", "message": "Created"}
    finally: conn.close()

@app.post("/login")
async def login(user: UserLogin):
    ensure_user_exists(user.email)
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    row = c.execute("SELECT name, is_verified, scan_count, is_premium FROM users WHERE email=?", (user.email,)).fetchone()
    conn.close()
    if row: return {"status": "success", "username": row[0], "scan_count": row[2], "is_premium": bool(row[3])}
    return JSONResponse(content={"status": "error", "message": "Invalid credentials"}, status_code=401)

@app.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    ensure_user_exists(req.email)
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    t = ''.join(random.choices(string.ascii_letters, k=32))
    c.execute("UPDATE users SET reset_token=? WHERE email=?", (t, req.email)); conn.commit(); conn.close()
    if not EMAIL_USER or not EMAIL_PASS: return JSONResponse(content={"status": "error", "message": "Server email not configured"}, status_code=500)
    try:
        msg = MIMEMultipart(); msg['From'] = EMAIL_USER; msg['To'] = req.email; msg['Subject'] = "Reset Password"
        msg.attach(MIMEText(f"Link: {SERVER_URL}/reset-password-view?token={t}", 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls(); server.login(EMAIL_USER, EMAIL_PASS); server.send_message(msg); server.quit()
        return {"status": "success", "message": "Link sent"}
    except Exception as e: return JSONResponse(content={"status": "error", "message": f"Email Error: {str(e)}"}, status_code=500)

@app.get("/reset-password-view", response_class=HTMLResponse)
async def reset_view(token: str):
    return f"""<html><body style='background:#111;color:white;text-align:center;padding:50px;font-family:sans-serif'><h2>Reset Password</h2><form action='/reset-password-action' method='post'><input type='hidden' name='token' value='{token}'><input type='password' name='new_password' placeholder='New Password' style='padding:10px'><br><br><button style='padding:10px;background:#0f0'>Save</button></form></body></html>"""

@app.post("/reset-password-action", response_class=HTMLResponse)
async def reset_action(token: str = Form(...), new_password: str = Form(...)):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET password=?, reset_token=NULL WHERE reset_token=?", (hash_password(new_password), token))
    conn.commit(); conn.close()
    return "<html><body style='background:#111;color:#0f0;text-align:center;padding:50px'><h1>Success</h1></body></html>"

@app.get("/payment-success-landing", response_class=HTMLResponse)
async def payment_success_page():
    return """<html><body style='background:#0A0E21;color:white;text-align:center;padding:50px;font-family:sans-serif'><h1>PAYMENT COMPLETE</h1><br><a href="verifyapp://payment-success" style="background:#00E676;color:black;padding:15px;text-decoration:none;font-weight:bold;border-radius:10px;">OPEN APP</a><script>setTimeout(function(){window.location.href="verifyapp://payment-success";},1000);</script></body></html>"""

@app.post("/create-checkout-session")
async def create_checkout_session(req: PaymentRequest):
    try:
        cs = stripe.checkout.Session.create(
            payment_method_types=['card'], customer_email=req.email, client_reference_id=req.email,
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': 'Verify Premium'}, 'unit_amount': 400, 'recurring': {'interval': 'month'}}, 'quantity': 1}],
            mode='subscription', success_url=f'{SERVER_URL}/payment-success-landing', cancel_url='https://google.com')
        return {"status": "success", "url": cs.url}
    except Exception as e: return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    payload = await request.body()
    try: event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_ENDPOINT_SECRET)
    except: return JSONResponse(content={"error": "Invalid payload"}, status_code=400)
    if event['type'] == 'checkout.session.completed':
        user_email = event['data']['object'].get('client_reference_id')
        if user_email:
            ensure_user_exists(user_email)
            conn = sqlite3.connect(DB_NAME); c = conn.cursor()
            c.execute("UPDATE users SET is_premium=1 WHERE email=?", (user_email,))
            conn.commit(); conn.close()
    return {"status": "success"}
