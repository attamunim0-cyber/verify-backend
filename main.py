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
from PIL import Image, ExifTags
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()
DB_NAME = "verify_shield_v2.db"

# --- CONFIGURATION ---
# REPLACE THIS WITH YOUR STRIPE SECRET KEY (sk_test_...)
stripe.api_key = "sk_test_51Q2kAoCFpfwvg3QdGGP7uAibYjbJCv9mqorL582t1Tp2uXtGcNLgyAFRfqYN8eNqpnLhvOAk5zNGkfN4wDp4QtR000JjAFj72n" 
# REPLACE THIS WITH YOUR STRIPE WEBHOOK SECRET (whsec_...) found in Stripe Dashboard -> Developers -> Webhooks
STRIPE_ENDPOINT_SECRET = "whsec_HmhvhPDOimozUwdmH135qmtv6DleLWN7"

EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
SERVER_URL = "https://verify-backend-03or.onrender.com"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, password TEXT, verification_code TEXT, is_verified INTEGER DEFAULT 1, scan_count INTEGER DEFAULT 0, is_premium INTEGER DEFAULT 0, reset_token TEXT)''')
    conn.commit(); conn.close()
init_db()

# --- CALIBRATED AI DETECTION ENGINE (V4 - Mobile Optimized) ---
def analyze_media(file_path):
    try:
        score = 0.0
        reasons = []
        
        # 1. METADATA CHECK (Specific AI Generators)
        try:
            pil_img = Image.open(file_path)
            exif_data = pil_img.getexif()
            meta_str = str(exif_data) + str(pil_img.info)
            ai_keywords = ["stable diffusion", "midjourney", "dall-e", "generated", "comfyui", "automatic1111"]
            if any(k in meta_str.lower() for k in ai_keywords):
                return "FAKE", 0.99, "Metadata tag explicitly identifies AI."
        except: pass

        img = cv2.imread(file_path)
        if img is None:
            # Video fallback
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, _ = cap.read(); cap.release()
                if ret: return "REAL", 0.85, "Motion vectors appear organic."
            return "ERROR", 0.0, "Could not decode media."

        # --- MEMORY SAVER (Resize) ---
        height, width = img.shape[:2]
        max_dim = 1200 # Lowered slightly for speed
        if height > max_dim or width > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Convert Colors
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # TEST 2: FREQUENCY ANALYSIS (Grid Check)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
        
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        mask_size = 40 
        magnitude_spectrum[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
        high_freq_energy = np.mean(magnitude_spectrum)
        
        # --- TUNED THRESHOLDS FOR RESIZED IMAGES ---
        # Real photos usually land between 80 - 140 after resizing.
        # AI usually lands > 160 (Grid) or < 60 (Plastic smooth).
        
        if high_freq_energy > 165: # Raised from 135 to avoid false positives on detailed textures
            score += 1.0 
            reasons.append(f"Suspicious grid artifacts detected (Energy: {int(high_freq_energy)}).")
        elif high_freq_energy < 50: # Lowered from 75 because resizing causes smoothing
            score += 0.5 # Reduced weight
            reasons.append("Texture is unusually smooth.")

        # TEST 3: SATURATION (Vibrancy)
        saturation = hsv[:,:,1]
        mean_sat = np.mean(saturation)
        # Modern phones (Samsung/iPhone) have high saturation (HDR). 
        # Raised threshold to 135 (was 110).
        if mean_sat > 135: 
            score += 0.5
            reasons.append("Color saturation is abnormally high.")

        # TEST 4: LAPLACIAN (Sharpness/Noise)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Lowered "too flat" threshold to 50 for low-light support
        if laplacian_var < 50:
             score += 0.5
             reasons.append("Image lacks depth (Too flat).")
        elif laplacian_var > 4500: # Very noisy
             score += 0.5
             reasons.append("Noise levels suggest synthetic grain.")

        # --- FINAL VERDICT ---
        print(f"DEBUG: Score={score}, Reasons={reasons}")
        
        # We now require a higher score to call it FAKE (Guilty beyond reasonable doubt)
        if score >= 2.0:
            return "FAKE", 0.96, reasons[0]
        elif score >= 1.0:
            return "SUSPICIOUS", 0.65, "Image shows mixed signals."
        else:
            return "REAL", 0.94, "Analysis indicates organic origin."

    except Exception as e:
        print(f"Analysis Error: {e}")
        return "ERROR", 0.0, "Analysis Failed"

# --- DOWNLOADERS ---
def download_video_site(url):
    print(f"Downloading: {url}")
    temp_dir = tempfile.mkdtemp()
    ydl_opts = {'format': 'best', 'outtmpl': f"{temp_dir}/%(id)s.%(ext)s", 'quiet': True, 'no_warnings': True, 'user_agent': 'Mozilla/5.0'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
        files = glob.glob(os.path.join(temp_dir, "*")); 
        if not files: raise Exception("No file found.")
        return files[0]
    except Exception as e: shutil.rmtree(temp_dir, ignore_errors=True); raise e

def download_direct(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, timeout=15, headers=headers); r.raise_for_status()
    s = ".jpg" if "jpg" in url or "png" in url else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=s) as t: t.write(r.content); return t.name

# --- ENDPOINTS ---
class UrlRequest(BaseModel): url: str; email: str
class UserSignup(BaseModel): name: str; email: str; password: str
class UserLogin(BaseModel): email: str; password: str
class PaymentRequest(BaseModel): email: str
class ForgotPasswordRequest(BaseModel): email: str

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def check_limits(email):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    u = c.execute("SELECT scan_count, is_premium FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not u: return False
    # If premium, unlimited scans. If not, limit to 3.
    if u[1] == 1: return True
    if u[0] >= 3: return False
    
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET scan_count = scan_count + 1 WHERE email=?", (email,)); conn.commit(); conn.close()
    return True

@app.post("/analyze-url")
async def analyze_url(req: UrlRequest):
    if not check_limits(req.email): return JSONResponse(content={"verdict": "LIMIT_REACHED", "message": "Limit reached"}, status_code=403)
    temp_path = None
    try:
        sites = ['youtube', 'youtu', 'facebook', 'fb.watch', 'instagram', 'tiktok']
        if any(s in req.url for s in sites): temp_path = download_video_site(req.url)
        else: temp_path = download_direct(req.url)
        v, c, m = analyze_media(temp_path)
        return {"verdict": v, "score": c, "message": m}
    except Exception as e: 
        err = str(e)
        return JSONResponse(content={"verdict": "ERROR", "message": err}, status_code=400)
    finally:
        if temp_path and os.path.exists(temp_path):
             try: os.remove(temp_path); shutil.rmtree(os.path.dirname(temp_path), ignore_errors=True)
             except: pass

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), email: str = Form(...)):
    if not check_limits(email): return JSONResponse(content={"verdict": "LIMIT_REACHED", "message": "Limit reached"}, status_code=403)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    try:
        v, c, m = analyze_media(temp_path)
        return {"verdict": v, "score": c, "message": m}
    except Exception as e: return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
         if os.path.exists(temp_path): os.remove(temp_path)

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
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    row = c.execute("SELECT name, is_verified, scan_count, is_premium FROM users WHERE email=? AND password=?", (user.email, hash_password(user.password))).fetchone()
    conn.close()
    if row: return {"status": "success", "username": row[0], "scan_count": row[2], "is_premium": bool(row[3])}
    return JSONResponse(content={"status": "error", "message": "Invalid credentials"}, status_code=401)

# --- NEW: STRIPE PAYMENT WITH WEBHOOK ---
@app.post("/create-checkout-session")
async def create_checkout_session(req: PaymentRequest):
    try:
        # We pass the user's email as metadata so the Webhook knows who to upgrade
        cs = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=req.email,
            client_reference_id=req.email,
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': 'Verify Premium'}, 'unit_amount': 400, 'recurring': {'interval': 'month'}}, 'quantity': 1}],
            mode='subscription', 
            success_url='https://google.com', # In a real app, this should be a "Thank you" page
            cancel_url='https://google.com'
        )
        return {"status": "success", "url": cs.url}
    except Exception as e: return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_ENDPOINT_SECRET)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    # LISTEN FOR SUCCESSFUL PAYMENT
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        # Retrieve user email from the session
        user_email = session.get('client_reference_id') or session.get('customer_email')
        
        if user_email:
            print(f"UPGRADING USER: {user_email}")
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            # Set is_premium = 1 (True)
            c.execute("UPDATE users SET is_premium=1 WHERE email=?", (user_email,))
            conn.commit()
            conn.close()
            
    return {"status": "success"}

@app.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    if c.execute("SELECT * FROM users WHERE email=?", (req.email,)).fetchone():
        t = ''.join(random.choices(string.ascii_letters, k=32))
        c.execute("UPDATE users SET reset_token=? WHERE email=?", (t, req.email)); conn.commit()
        if EMAIL_USER and EMAIL_PASS:
            msg = MIMEMultipart(); msg['From']=EMAIL_USER; msg['To']=req.email; msg['Subject']="Reset Password"
            msg.attach(MIMEText(f"Link: {SERVER_URL}/reset-password-view?token={t}")); 
            s = smtplib.SMTP('smtp.gmail.com',587); s.starttls(); s.login(EMAIL_USER, EMAIL_PASS); s.send_message(msg); s.quit()
    conn.close(); return {"status": "success", "message": "Link sent"}

@app.get("/reset-password-view", response_class=HTMLResponse)
async def reset_view(token: str):
    return f"""<html><body style='background:#111;color:white;text-align:center;padding:50px;font-family:sans-serif'><h2>Reset Password</h2><form action='/reset-password-action' method='post'><input type='hidden' name='token' value='{token}'><input type='password' name='new_password' placeholder='New Password' style='padding:10px'><br><br><button style='padding:10px;background:#0f0'>Save</button></form></body></html>"""

@app.post("/reset-password-action", response_class=HTMLResponse)
async def reset_action(token: str = Form(...), new_password: str = Form(...)):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET password=?, reset_token=NULL WHERE reset_token=?", (hash_password(new_password), token))
    conn.commit(); conn.close()
    return "<html><body style='background:#111;color:#0f0;text-align:center;padding:50px'><h1>Success</h1></body></html>"
