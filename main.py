from fastapi import FastAPI, File, UploadFile, Form, Request, Header, HTTPException
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
# 1. GET THIS FROM STRIPE DASHBOARD -> DEVELOPERS -> API KEYS
stripe.api_key = "sk_test_51Q2kAoCFpfwvg3QdGGP7uAibYjbJCv9mqorL582t1Tp2uXtGcNLgyAFRfqYN8eNqpnLhvOAk5zNGkfN4wDp4QtR000JjAFj72n" 

# 2. GET THIS FROM STRIPE DASHBOARD -> DEVELOPERS -> WEBHOOKS -> SIGNING SECRET
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

# --- V6: MULTI-FRAME VIDEO ANALYSIS & TUNED THRESHOLDS ---
def analyze_pixels(img_bgr, source_type="Image"):
    score = 0.0
    reasons = []

    # 1. RESIZE (Standardize input to prevent Server Crashes)
    height, width = img_bgr.shape[:2]
    max_dim = 1000 # Smaller size reduces compression noise false positives
    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 2. FREQUENCY (Grid Check)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
    
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30] = 0
    high_freq_energy = np.mean(magnitude_spectrum)
    
    # TUNING: Real videos have compression blocks that look like grids.
    # We RAISE the threshold for "Fake" to avoid catching real videos.
    grid_threshold = 175 if source_type == "Video Frame" else 165
    
    if high_freq_energy > grid_threshold: 
        score += 1.0 
        reasons.append(f"High-frequency grid artifacts ({int(high_freq_energy)}).")
    elif high_freq_energy < 40: # Lowered to avoid flagging dark real photos
        score += 0.5 
        reasons.append("Texture is unnaturally smooth.")

    # 3. SATURATION
    saturation = hsv[:,:,1]
    mean_sat = np.mean(saturation)
    if mean_sat > 140: # Relaxed for modern HDR cameras
        score += 0.5
        reasons.append("Saturation levels are synthetic.")

    # 4. LAPLACIAN (Blur/Noise)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 30: # Relaxed for low-light videos
            score += 0.5
            reasons.append("Lacks optical depth (Flat).")
    elif laplacian_var > 5000: 
            score += 0.5
            reasons.append("Noise pattern is synthetic.")

    return score, reasons

def analyze_media(file_path):
    try:
        # A. METADATA (Fast Fail)
        try:
            pil_img = Image.open(file_path)
            meta = str(pil_img.getexif()) + str(pil_img.info)
            if any(k in meta.lower() for k in ["midjourney", "diffusion", "generated"]):
                return "FAKE", 0.99, "Metadata identifies AI."
        except: pass

        # B. IMAGE HANDLING
        img = cv2.imread(file_path)
        if img is not None:
            score, reasons = analyze_pixels(img, "Image")
            if score >= 2.0: return "FAKE", 0.96, reasons[0]
            elif score >= 1.0: return "SUSPICIOUS", 0.65, "Mixed organic/digital signals."
            else: return "REAL", 0.94, "Consistent organic texture."

        # C. VIDEO HANDLING (Multi-Frame)
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened(): return "ERROR", 0.0, "Could not open media."

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3: 
             # Too short? Treat as Real or Error
             return "REAL", 0.80, "Video too short for analysis."

        # Analyze 3 frames: Start (10%), Middle (50%), End (90%)
        points = [total_frames * 0.1, total_frames * 0.5, total_frames * 0.9]
        frame_scores = []
        
        for p in points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(p))
            ret, frame = cap.read()
            if ret:
                s, _ = analyze_pixels(frame, "Video Frame")
                frame_scores.append(s)
        cap.release()

        if not frame_scores: return "ERROR", 0.0, "Could not read frames."

        # VERDICT LOGIC
        avg_score = sum(frame_scores) / len(frame_scores)
        
        # Real videos are messy. AI videos are consistent.
        # If ANY frame looks extremely fake (score > 2), flag it.
        if max(frame_scores) >= 2.5:
             return "FAKE", 0.95, "Synthetic artifacts found in keyframes."
        elif avg_score >= 1.5:
             return "FAKE", 0.90, "Consistently artificial texture."
        elif avg_score >= 0.8:
             # Relaxed: Real videos often land here due to compression
             return "SUSPICIOUS", 0.60, "Compression or editing detected."
        else:
             return "REAL", 0.92, "Organic motion and texture."

    except Exception as e:
        print(f"Error: {e}")
        return "ERROR", 0.0, "Analysis Failed"

# --- ENDPOINTS ---
class UserRequest(BaseModel): email: str # For Profile Fetch
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
    if u[1] == 1: return True
    if u[0] >= 3: return False
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET scan_count = scan_count + 1 WHERE email=?", (email,)); conn.commit(); conn.close()
    return True

# --- NEW: PROFILE ENDPOINT (FIXES THE 404 ERROR) ---
@app.post("/user-profile")
async def get_user_profile(req: UserRequest):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    row = c.execute("SELECT name, email, is_premium, scan_count FROM users WHERE email=?", (req.email,)).fetchone()
    conn.close()
    if row:
        return {
            "status": "success",
            "name": row[0],
            "email": row[1],
            "is_premium": bool(row[2]),
            "scan_count": row[3],
            "plan_name": "Premium Plan" if row[2] else "Free Starter Plan"
        }
    return JSONResponse(content={"status": "error", "message": "User not found"}, status_code=404)

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
        return JSONResponse(content={"verdict": "ERROR", "message": str(e)}, status_code=400)
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

@app.post("/create-checkout-session")
async def create_checkout_session(req: PaymentRequest):
    try:
        cs = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=req.email,
            client_reference_id=req.email,
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': 'Verify Premium'}, 'unit_amount': 400, 'recurring': {'interval': 'month'}}, 'quantity': 1}],
            mode='subscription', success_url='https://google.com', cancel_url='https://google.com')
        return {"status": "success", "url": cs.url}
    except Exception as e: return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_ENDPOINT_SECRET)
    except Exception as e: return JSONResponse(content={"error": str(e)}, status_code=400)

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_email = session.get('client_reference_id') or session.get('customer_email')
        if user_email:
            conn = sqlite3.connect(DB_NAME); c = conn.cursor()
            c.execute("UPDATE users SET is_premium=1 WHERE email=?", (user_email,))
            conn.commit(); conn.close()
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
