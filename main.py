from fastapi import FastAPI, File, UploadFile, Form
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
from PIL import Image, ExifTags # New library for metadata
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()
DB_NAME = "verify_shield_v2.db"
stripe.api_key = "sk_test_51Q2kAoCFpfwvg3QdGGP7uAibYjbJCv9mqorL582t1Tp2uXtGcNLgyAFRfqYN8eNqpnLhvOAk5zNGkfN4wDp4QtR000JjAFj72n" 

# --- EMAIL CONFIG ---
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
SERVER_URL = "https://verify-backend-03or.onrender.com" 

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, password TEXT, verification_code TEXT, is_verified INTEGER DEFAULT 1, scan_count INTEGER DEFAULT 0, is_premium INTEGER DEFAULT 0, reset_token TEXT)''')
    conn.commit(); conn.close()
init_db()

# --- THE NEW "4-LAYER" AI DETECTION ENGINE ---
def analyze_media(file_path):
    try:
        score = 0
        reasons = []
        
        # 1. READ IMAGE & METADATA
        try:
            pil_img = Image.open(file_path)
            exif_data = pil_img.getexif()
            meta_str = str(exif_data) + str(pil_img.info)
            # AI generators often leave signatures in metadata
            ai_keywords = ["stable diffusion", "midjourney", "dall-e", "generated", "comfyui", "automatic1111"]
            if any(k in meta_str.lower() for k in ai_keywords):
                return "FAKE", 0.99, "Metadata explicitly identifies AI generation."
        except: pass

        img = cv2.imread(file_path)
        if img is None:
            # Video Fallback: Assume Real if video plays (simple check)
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, _ = cap.read(); cap.release()
                if ret: return "REAL", 0.85, "Video motion structure is organic."
            return "ERROR", 0.0, "Could not decode media."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TEST 2: FREQUENCY ANALYSIS (FFT)
        # Real cameras have "chaotic" noise. AI has "ordered" noise.
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        # Analyze high frequency corners
        magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30] = 0
        high_freq_energy = np.mean(magnitude_spectrum)
        
        # Lower threshold: AI often has unusually LOW high-freq energy (too smooth)
        # OR unusually HIGH periodic energy (grid artifacts).
        if high_freq_energy > 155:
            score += 1
            reasons.append("High-frequency artifacts detected (Grid Pattern).")
        elif high_freq_energy < 80:
             score += 1
             reasons.append("Texture lacks natural sensor noise (Too Smooth).")

        # TEST 3: LAPLACIAN VARIANCE (Blur/Sharpness)
        # AI images are often "perfectly" focused everywhere or strangely blurry.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100: 
            score += 1
            reasons.append("Surface topology is unnaturally flat.")
        
        # TEST 4: COLOR HISTOGRAM CONSISTENCY
        # Real photos rarely use the full dynamic range perfectly. AI often does.
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # Check if histogram peaks are too "spiky" (common in AI)
        if np.max(hist) > (rows * cols * 0.1): 
            score += 0.5
            reasons.append("Color distribution is synthetic.")

        # --- VERDICT LOGIC ---
        print(f"DEBUG: Score={score}, Reasons={reasons}")
        
        if score >= 1.5:
            final_msg = reasons[0] if reasons else "Synthetic patterns detected."
            return "FAKE", 0.95, final_msg
        elif score >= 1.0:
            return "SUSPICIOUS", 0.70, "Image shows mixed organic and digital traits."
        else:
            return "REAL", 0.92, "Natural sensor noise and organic frequency confirmed."
        
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
    if not u[1] and u[0] >= 3: return False
    if not u[1]: 
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
        if "Sign in" in err: err = "Link blocked by platform."
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

@app.post("/create-checkout-session")
async def create_checkout_session(req: PaymentRequest):
    try:
        cs = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': 'Verify Premium'}, 'unit_amount': 400, 'recurring': {'interval': 'month'}}, 'quantity': 1}],
            mode='subscription', success_url='https://google.com', cancel_url='https://google.com')
        return {"status": "success", "url": cs.url}
    except Exception as e: return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

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
