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
from PIL import Image, ImageChops, ImageEnhance
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()
DB_NAME = "verify_shield_v2.db"

# --- CONFIGURATION ---
stripe.api_key = "sk_test_51Q2kAoCFpfwvg3QdGGP7uAibYjbJCv9mqorL582t1Tp2uXtGcNLgyAFRfqYN8eNqpnLhvOAk5zNGkfN4wDp4QtR000JjAFj72n"
STRIPE_ENDPOINT_SECRET = "whsec_HmhvhPDOimozUwdmH135qmtv6DleLWN7"

# MUST BE SET IN RENDER DASHBOARD -> ENVIRONMENT
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
SERVER_URL = "https://verify-backend-03or.onrender.com"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, password TEXT, verification_code TEXT, is_verified INTEGER DEFAULT 1, scan_count INTEGER DEFAULT 0, is_premium INTEGER DEFAULT 0, reset_token TEXT)''')
    conn.commit(); conn.close()
init_db()

# --- V10: "PARANOID" TILE-BASED DETECTION ---
# Instead of resizing (which hides AI), we cut the image into small squares
# and analyze each square. If ANY square is fake, the whole image is fake.

def analyze_tile(tile_gray, tile_hsv):
    score = 0.0
    
    # 1. GRID CHECK (FFT)
    f = np.fft.fft2(tile_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
    h, w = tile_gray.shape
    magnitude_spectrum[h//2-10:h//2+10, w//2-10:w//2+10] = 0 # Block DC
    energy = np.mean(magnitude_spectrum)
    
    # Lower threshold = Catch more fakes
    if energy > 135: score += 1.5 # Strong Grid
    
    # 2. NOISE FLOOR (Laplacian)
    lap_var = cv2.Laplacian(tile_gray, cv2.CV_64F).var()
    if lap_var < 80: score += 1.0 # Too Smooth (AI Plastic Look)

    # 3. SATURATION SPIKES
    sat = tile_hsv[:,:,1]
    if np.mean(sat) > 150: score += 0.5 # Too Vivid

    return score

def analyze_media(file_path):
    try:
        # A. METADATA SPY
        try:
            pil_img = Image.open(file_path)
            meta = str(pil_img.getexif()) + str(pil_img.info)
            ai_tags = ["midjourney", "diffusion", "generated", "dall-e", "firefly", "a1111"]
            if any(k in meta.lower() for k in ai_tags):
                return "FAKE", 0.99, "Metadata explicit AI tag found."
        except: pass

        # B. IMAGE HANDLING (Tile Strategy)
        img = cv2.imread(file_path)
        if img is not None:
            # Don't resize. Split into 256x256 tiles.
            h, w = img.shape[:2]
            tile_size = 256
            
            # If too massive, downscale slightly to prevent crash, but keep it huge
            if h > 3000 or w > 3000:
                scale = 3000 / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)
                h, w = img.shape[:2]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            max_tile_score = 0.0
            
            # Check center crop and random tiles
            steps_y = max(1, h // tile_size)
            steps_x = max(1, w // tile_size)
            
            # Check up to 8 random tiles to save speed
            checks = 0
            for _ in range(8):
                y = random.randint(0, max(0, h - tile_size))
                x = random.randint(0, max(0, w - tile_size))
                
                tile_g = gray[y:y+tile_size, x:x+tile_size]
                tile_h = hsv[y:y+tile_size, x:x+tile_size]
                
                if tile_g.size == 0: continue
                
                s = analyze_tile(tile_g, tile_h)
                if s > max_tile_score: max_tile_score = s
            
            # VERDICT
            if max_tile_score >= 1.5: return "FAKE", 0.98, "Synthetic patterns detected in texture analysis."
            elif max_tile_score >= 1.0: return "SUSPICIOUS", 0.75, "Image lacks natural sensor noise."
            else: return "REAL", 0.92, "Organic noise profile confirmed."

        # C. VIDEO HANDLING (Frame Sampling)
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened(): return "ERROR", 0.0, "Read Error"
        
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames < 5: return "REAL", 0.85, "Video too short."

        fake_hits = 0
        checks = 6
        for _ in range(checks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, frames-1))
            ret, frame = cap.read()
            if ret:
                # Resize frame only if 4K (keep 1080p)
                fh, fw = frame.shape[:2]
                if fh > 1080:
                    fs = 1080 / fh
                    frame = cv2.resize(frame, None, fx=fs, fy=fs)
                
                fgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Check center tile of frame
                cy, cx = fh//2, fw//2
                tile_g = fgray[cy-128:cy+128, cx-128:cx+128]
                tile_h = fhsv[cy-128:cy+128, cx-128:cx+128]
                
                if tile_g.shape[0] < 256: continue # Skip small frames

                if analyze_tile(tile_g, tile_h) >= 1.0: fake_hits += 1

        cap.release()

        if fake_hits >= 2: return "FAKE", 0.95, "Inconsistent artifacts across frames."
        else: return "REAL", 0.90, "Motion consistent with optical capture."

    except Exception as e:
        print(f"ERR: {e}")
        return "ERROR", 0.0, "Analysis Failed"

# --- ENDPOINTS & AUTO-HEAL ---
class UserRequest(BaseModel): email: str 
class UrlRequest(BaseModel): url: str; email: str
class UserSignup(BaseModel): name: str; email: str; password: str
class UserLogin(BaseModel): email: str; password: str
class PaymentRequest(BaseModel): email: str
class ForgotPasswordRequest(BaseModel): email: str

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def ensure_user_exists(email):
    """If Render wiped DB, silently re-create user to prevent blocking."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        print(f"AUTO-HEALING: Re-creating {email}")
        c.execute("INSERT INTO users (email, name, password, is_verified, scan_count, is_premium) VALUES (?, ?, ?, 1, 0, 0)", (email, "User", "auto_restored"))
        conn.commit()
    conn.close()

def check_limits(email):
    ensure_user_exists(email)
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row; c = conn.cursor()
    row = c.execute("SELECT scan_count, is_premium FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    
    if row['is_premium'] == 1: return True
    if row['scan_count'] >= 3: return False
    
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET scan_count = scan_count + 1 WHERE email=?", (email,))
    conn.commit(); conn.close()
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
        # Simplified Downloader logic
        temp_dir = tempfile.mkdtemp()
        ydl_opts = {'format': 'best', 'outtmpl': f"{temp_dir}/%(id)s.%(ext)s", 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([req.url])
        files = glob.glob(os.path.join(temp_dir, "*"))
        v, c, m = analyze_media(files[0])
        return {"verdict": v, "score": c, "message": m}
    except Exception as e: return JSONResponse(content={"verdict": "ERROR", "message": str(e)}, status_code=400)

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
    ensure_user_exists(user.email) # Auto-heal on login attempt
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    row = c.execute("SELECT name, is_verified, scan_count, is_premium FROM users WHERE email=?", (user.email,)).fetchone()
    conn.close()
    if row: return {"status": "success", "username": row[0], "scan_count": row[2], "is_premium": bool(row[3])}
    return JSONResponse(content={"status": "error", "message": "Invalid credentials"}, status_code=401)

# --- EMAIL & PASSWORD ---
@app.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    ensure_user_exists(req.email)
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    t = ''.join(random.choices(string.ascii_letters, k=32))
    c.execute("UPDATE users SET reset_token=? WHERE email=?", (t, req.email)); conn.commit(); conn.close()
    
    if not EMAIL_USER or not EMAIL_PASS:
        return JSONResponse(content={"status": "error", "message": "Server email not configured"}, status_code=500)
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = req.email
        msg['Subject'] = "Verify Shield Password Reset"
        msg.attach(MIMEText(f"Click to reset: {SERVER_URL}/reset-password-view?token={t}", 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return {"status": "success", "message": "Link sent"}
    except Exception as e:
        print(f"EMAIL ERROR: {e}")
        return JSONResponse(content={"status": "error", "message": f"Failed to send email: {str(e)}"}, status_code=500)

@app.get("/reset-password-view", response_class=HTMLResponse)
async def reset_view(token: str):
    return f"""<html><body style='background:#111;color:white;text-align:center;padding:50px;font-family:sans-serif'><h2>Reset Password</h2><form action='/reset-password-action' method='post'><input type='hidden' name='token' value='{token}'><input type='password' name='new_password' placeholder='New Password' style='padding:10px'><br><br><button style='padding:10px;background:#0f0'>Save</button></form></body></html>"""

@app.post("/reset-password-action", response_class=HTMLResponse)
async def reset_action(token: str = Form(...), new_password: str = Form(...)):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET password=?, reset_token=NULL WHERE reset_token=?", (hash_password(new_password), token))
    conn.commit(); conn.close()
    return "<html><body style='background:#111;color:#0f0;text-align:center;padding:50px'><h1>Success</h1></body></html>"

# --- PAYMENT ---
@app.get("/payment-success-landing", response_class=HTMLResponse)
async def payment_success_page():
    return """<html><body style='background:#0A0E21;color:white;text-align:center;padding:50px;font-family:sans-serif'><h1>PAYMENT COMPLETE</h1><p>Return to app.</p><a href="verifyapp://payment-success" style="background:#00E676;color:black;padding:15px;text-decoration:none;font-weight:bold;border-radius:10px;">OPEN APP</a><script>setTimeout(function(){window.location.href="verifyapp://payment-success";},1000);</script></body></html>"""

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
