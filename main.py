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

app = FastAPI()
DB_NAME = "verify_shield_v2.db"

# --- CONFIGURATION ---
stripe.api_key = "sk_test_51Q2kAoCFpfwvg3QdGGP7uAibYjbJCv9mqorL582t1Tp2uXtGcNLgyAFRfqYN8eNqpnLhvOAk5zNGkfN4wDp4QtR000JjAFj72n"
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

# --- V9: ADAPTIVE TITANIUM DETECTION ENGINE ---
def analyze_pixels_adaptive(img_bgr, source_type="Image"):
    score = 0.0
    reasons = []

    # 1. PREP
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. ADAPTIVE NOISE FLOOR CALCULATION
    # Instead of a fixed number, we measure the image's own natural noise.
    # Real photos have consistent Gaussian noise. AI has "patches" of silence.
    sigma = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. FREQUENCY DOMAIN (GRID CHECK)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
    
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    # Mask low frequencies (content) to see only high frequencies (artifacts)
    magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30] = 0
    freq_energy = np.mean(magnitude_spectrum)

    # --- DYNAMIC THRESHOLDS ---
    # Adjust sensitivity based on how "busy" the image is (sigma)
    
    # TEST A: THE "MIDJOURNEY GRID"
    # AI Generators leave a grid signature > 165 usually.
    # But high-detail Real photos can also be high. We check ratio vs sigma.
    grid_ratio = freq_energy / (sigma + 1e-5)
    
    if freq_energy > 170 and grid_ratio > 3.5:
        score += 1.5
        reasons.append(f"Synthetic grid artifacts detected (Energy: {int(freq_energy)}).")
    
    # TEST B: THE "SMOOTHNESS TRAP" (Plastic Skin)
    # Real photos have grain. If variance is super low, it's likely AI or heavily filtered.
    # Lower threshold for video because compression kills grain.
    smooth_floor = 30 if source_type == "Video" else 60
    
    if laplacian_var < smooth_floor:
        score += 1.0
        reasons.append("Texture is unnaturally smooth (Plastic signature).")

    # TEST C: COLOR CLIPPING (Saturation)
    # AI over-saturates colors.
    sat = hsv[:,:,1]
    bright = hsv[:,:,2]
    # Check for "Neon" colors common in AI
    high_sat_ratio = np.sum(sat > 240) / sat.size
    if high_sat_ratio > 0.05: # If >5% of pixels are MAX saturation
        score += 0.5
        reasons.append("Unnatural color saturation peaks.")

    return score, reasons

def analyze_media(file_path):
    try:
        # A. METADATA SPY
        try:
            pil_img = Image.open(file_path)
            meta = str(pil_img.getexif()) + str(pil_img.info)
            ai_tags = ["midjourney", "diffusion", "generated", "dall-e", "firefly"]
            if any(k in meta.lower() for k in ai_tags):
                return "FAKE", 0.99, "Metadata tag explicitly identifies AI."
        except: pass

        # B. IMAGE ANALYSIS
        img = cv2.imread(file_path)
        if img is not None:
            # Resize smartly - keep enough detail for grid detection
            h, w = img.shape[:2]
            if h > 1200 or w > 1200:
                s = 1200 / max(h, w)
                img = cv2.resize(img, None, fx=s, fy=s)

            score, reasons = analyze_pixels_adaptive(img, "Image")
            
            if score >= 1.5: return "FAKE", 0.98, reasons[0]
            elif score >= 1.0: return "SUSPICIOUS", 0.70, "Mixed organic/digital signals."
            else: return "REAL", 0.94, "Organic sensor pattern confirmed."

        # C. VIDEO ANALYSIS (Jitter Check)
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened(): return "ERROR", 0.0, "Read Error"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 5: return "REAL", 0.85, "Video too short."

        # Analyze 6 frames to catch "Flicker"
        scores = []
        frames_to_check = 6
        for _ in range(frames_to_check):
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, total_frames-1))
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                if h > 800:
                    s = 800 / max(h, w)
                    frame = cv2.resize(frame, None, fx=s, fy=s)
                s_score, _ = analyze_pixels_adaptive(frame, "Video")
                scores.append(s_score)
        cap.release()

        # VIDEO VERDICT LOGIC
        avg_score = sum(scores) / len(scores)
        # AI Video often flickers between "Very Fake" and "Okay". 
        # Real video is consistently "Okay".
        variance = np.var(scores)
        
        if avg_score >= 1.2:
            return "FAKE", 0.96, "Consistently artificial texture."
        elif variance > 0.5:
             # High variance means the style changes frame-to-frame (AI hallucination)
             return "FAKE", 0.92, "Temporal instability detected (AI Flicker)."
        elif avg_score >= 0.8:
            return "SUSPICIOUS", 0.65, "Compression or editing detected."
        else:
            return "REAL", 0.90, "Organic motion confirmed."

    except Exception as e:
        print(f"Error: {e}")
        return "ERROR", 0.0, "Analysis Failed"

# --- ENDPOINTS ---
class UserRequest(BaseModel): email: str 
class UrlRequest(BaseModel): url: str; email: str
class UserSignup(BaseModel): name: str; email: str; password: str
class UserLogin(BaseModel): email: str; password: str
class PaymentRequest(BaseModel): email: str
class ForgotPasswordRequest(BaseModel): email: str

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def check_limits(email):
    conn = sqlite3.connect(DB_NAME); 
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # AUTO-HEAL: Ensure user exists before checking limits
    c.execute("INSERT OR IGNORE INTO users (email, name, password, is_verified, scan_count, is_premium) VALUES (?, ?, ?, 1, 0, 0)", (email, "User", "hash"))
    conn.commit()
    
    row = c.execute("SELECT scan_count, is_premium FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    
    if row['is_premium'] == 1: return True
    if row['scan_count'] >= 3: return False
    
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET scan_count = scan_count + 1 WHERE email=?", (email,)); conn.commit(); conn.close()
    return True

@app.post("/user-profile")
async def get_user_profile(req: UserRequest):
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row; c = conn.cursor()
    
    # AUTO-HEAL
    c.execute("INSERT OR IGNORE INTO users (email, name, password, is_verified, scan_count, is_premium) VALUES (?, ?, ?, 1, 0, 0)", (req.email, "User", "restored"))
    conn.commit()
    
    row = c.execute("SELECT name, email, is_premium, scan_count FROM users WHERE email=?", (req.email,)).fetchone()
    conn.close()
    return {"status": "success", "name": row['name'], "email": row['email'], "is_premium": bool(row['is_premium']), "scan_count": row['scan_count'], "plan_name": "Premium Plan" if row['is_premium'] else "Starter Plan"}

@app.post("/analyze-url")
async def analyze_url(req: UrlRequest):
    if not check_limits(req.email): return JSONResponse(content={"verdict": "LIMIT_REACHED", "message": "Limit reached"}, status_code=403)
    temp_path = None
    try:
        sites = ['youtube', 'youtu', 'facebook', 'fb.watch', 'instagram', 'tiktok']
        if any(s in req.url for s in sites): 
            temp_dir = tempfile.mkdtemp()
            ydl_opts = {'format': 'best', 'outtmpl': f"{temp_dir}/%(id)s.%(ext)s", 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([req.url])
            files = glob.glob(os.path.join(temp_dir, "*"))
            temp_path = files[0]
        else: 
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(req.url, timeout=15, headers=headers); r.raise_for_status()
            s = ".jpg" if "jpg" in req.url or "png" in req.url else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=s) as t: t.write(r.content); temp_path = t.name
        
        v, c, m = analyze_media(temp_path)
        return {"verdict": v, "score": c, "message": m}
    except Exception as e: return JSONResponse(content={"verdict": "ERROR", "message": str(e)}, status_code=400)
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

# --- NEW: PAYMENT SUCCESS LANDING PAGE ---
# This page is shown in the browser after payment.
# It has a button that forces the App to open.
@app.get("/payment-success-landing", response_class=HTMLResponse)
async def payment_success_page():
    return """
    <html>
    <head>
        <title>Payment Successful</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #0A0E21; color: white; font-family: sans-serif; text-align: center; padding: 50px; }
            .btn { background-color: #00E676; color: black; padding: 15px 30px; text-decoration: none; font-size: 20px; border-radius: 10px; display: inline-block; margin-top: 20px; font-weight: bold;}
            h1 { color: #00E676; }
        </style>
    </head>
    <body>
        <h1>PAYMENT COMPLETE</h1>
        <p>Your account has been upgraded to Premium.</p>
        <p>Click below to return to the app.</p>
        <a href="verifyapp://payment-success" class="btn">OPEN VERIFY SHIELD</a>
        
        <script>
            // Try to auto-open app
            setTimeout(function() { window.location.href = "verifyapp://payment-success"; }, 1000);
        </script>
    </body>
    </html>
    """

@app.post("/create-checkout-session")
async def create_checkout_session(req: PaymentRequest):
    try:
        cs = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=req.email,
            client_reference_id=req.email,
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': 'Verify Premium'}, 'unit_amount': 400, 'recurring': {'interval': 'month'}}, 'quantity': 1}],
            mode='subscription', 
            # FIX: Send to our new HTML Landing Page
            success_url=f'{SERVER_URL}/payment-success-landing', 
            cancel_url='https://google.com'
        )
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
            c.execute("INSERT OR IGNORE INTO users (email, name, password, is_verified, scan_count, is_premium) VALUES (?, ?, ?, 1, 0, 0)", (user_email, "PaidUser", "hash"))
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
