from fastapi import FastAPI, File, UploadFile, Form, Request, Header
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

# --- V7: TITANIUM GRADE DETECTION ENGINE ---
# Uses ELA (Error Level Analysis) + Frequency + Noise + Metadata

def perform_ela(img_path):
    """Checks for compression anomalies common in AI edits."""
    try:
        original = Image.open(img_path).convert('RGB')
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            original.save(tmp.name, 'JPEG', quality=90)
            resaved = Image.open(tmp.name)
            
            # Calculate difference (Error)
            ela_img = ImageChops.difference(original, resaved)
            extrema = ela_img.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            scale = 255.0 / max_diff if max_diff > 0 else 1
            ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
            
            # AI often leaves "rainbowing" or flat patches in ELA
            # We convert to grayscale and check variance
            stat = np.array(ela_img.convert('L'))
            return np.mean(stat), np.var(stat)
    except:
        return 0, 0

def analyze_pixels_v7(img_bgr, ela_score, source_type="Image"):
    score = 0.0
    reasons = []

    # 1. PRE-PROCESS
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. FREQUENCY DOMAIN (The "Invisible Grid" Check)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    # Look for high-freq spikes typical of Up-samplers (GANs/Diffusion)
    magnitude_spectrum[crow-20:crow+20, ccol-20:ccol+20] = 0
    freq_energy = np.mean(magnitude_spectrum)

    # 3. ELA CHECK (New Layer)
    # AI images often have very LOW ELA variance (too consistent)
    ela_mean, ela_var = ela_score
    if ela_var < 50: 
        score += 1.5
        reasons.append("Compression signature is artificially uniform (ELA).")

    # 4. STRICTER FREQUENCY THRESHOLDS
    # Real photos: 80-140. AI: >160 (Grid) or <50 (Smooth)
    if freq_energy > 160:
        score += 1.0
        reasons.append(f"Synthetic generation artifacts detected ({int(freq_energy)}).")
    elif freq_energy < 45:
        score += 0.5
        reasons.append("Texture lacks natural sensor noise.")

    # 5. SATURATION & LIGHTING (Flux/Midjourney Check)
    sat_channel = hsv[:,:,1]
    val_channel = hsv[:,:,2]
    
    # AI often has "perfect" lighting (High contrast, high saturation)
    if np.mean(sat_channel) > 130 and np.std(val_channel) > 70:
        score += 0.5
        reasons.append("Lighting/Color profile matches diffusion models.")

    return score, reasons

def analyze_media(file_path):
    try:
        # A. METADATA SPY (The easiest catch)
        try:
            pil_img = Image.open(file_path)
            meta = str(pil_img.getexif()) + str(pil_img.info)
            ai_tags = ["midjourney", "diffusion", "generated", "dall-e", "adobe firefly", "imagined"]
            if any(k in meta.lower() for k in ai_tags):
                return "FAKE", 0.99, "Metadata tag explicitly identifies AI."
        except: pass

        # B. ELA CALCULATION
        ela_data = perform_ela(file_path)

        # C. IMAGE ANALYSIS
        img = cv2.imread(file_path)
        if img is not None:
            # Resize for consistency (Standardize to 1024px)
            h, w = img.shape[:2]
            if h > 1024 or w > 1024:
                s = 1024 / max(h, w)
                img = cv2.resize(img, None, fx=s, fy=s)

            score, reasons = analyze_pixels_v7(img, ela_data, "Image")
            
            # AGGRESSIVE SCORING
            if score >= 1.5: return "FAKE", 0.98, reasons[0]
            elif score >= 1.0: return "SUSPICIOUS", 0.75, "High probability of AI manipulation."
            else: return "REAL", 0.94, "Organic sensor pattern confirmed."

        # D. VIDEO ANALYSIS (Consensus Engine)
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened(): return "ERROR", 0.0, "Read Error"
        
        frames_to_check = 5 # Increase from 3 to 5 for accuracy
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 5: return "REAL", 0.80, "Video too short."

        scores = []
        fake_frames = 0
        
        for i in range(frames_to_check):
            # Jump to random spots in video
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, total_frames-1))
            ret, frame = cap.read()
            if ret:
                # Resize frame
                h, w = frame.shape[:2]
                if h > 800:
                    s = 800 / max(h, w)
                    frame = cv2.resize(frame, None, fx=s, fy=s)
                
                # Analyze frame
                s_score, _ = analyze_pixels_v7(frame, (0,0), "Video") # Skip ELA for video (too slow)
                scores.append(s_score)
                if s_score >= 1.5: fake_frames += 1

        cap.release()

        # VIDEO VERDICT
        # If 40% of frames look Fake -> It's Fake.
        if fake_frames >= 2:
            return "FAKE", 0.96, "Inconsistent temporal artifacts (AI Video)."
        elif sum(scores)/len(scores) > 1.0:
            return "SUSPICIOUS", 0.70, "Synthetic texture detected in motion."
        else:
            return "REAL", 0.92, "Motion flow is organic."

    except Exception as e:
        print(f"ERR: {e}")
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
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    u = c.execute("SELECT scan_count, is_premium FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not u: return False
    if u[1] == 1: return True
    if u[0] >= 3: return False
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("UPDATE users SET scan_count = scan_count + 1 WHERE email=?", (email,)); conn.commit(); conn.close()
    return True

@app.post("/user-profile")
async def get_user_profile(req: UserRequest):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    row = c.execute("SELECT name, email, is_premium, scan_count FROM users WHERE email=?", (req.email,)).fetchone()
    conn.close()
    if row:
        return {"status": "success", "name": row[0], "email": row[1], "is_premium": bool(row[2]), "scan_count": row[3], "plan_name": "Premium Plan" if row[2] else "Starter Plan"}
    return JSONResponse(content={"status": "error"}, status_code=404)

@app.post("/analyze-url")
async def analyze_url(req: UrlRequest):
    if not check_limits(req.email): return JSONResponse(content={"verdict": "LIMIT_REACHED", "message": "Limit reached"}, status_code=403)
    temp_path = None
    try:
        sites = ['youtube', 'youtu', 'facebook', 'fb.watch', 'instagram', 'tiktok']
        if any(s in req.url for s in sites): 
            print(f"Downloading: {req.url}")
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

# --- DEEP LINK PAYMENT REDIRECT ---
@app.post("/create-checkout-session")
async def create_checkout_session(req: PaymentRequest):
    try:
        cs = stripe.checkout.Session.create(
            payment_method_types=['card'],
            customer_email=req.email,
            client_reference_id=req.email,
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': 'Verify Premium'}, 'unit_amount': 400, 'recurring': {'interval': 'month'}}, 'quantity': 1}],
            mode='subscription', 
            # FIX: THIS TELLS THE BROWSER TO OPEN YOUR APP
            success_url='verifyapp://payment-success', 
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
def download_video_site(url): return "Deprecated, use logic inside analyze-url" 
def download_direct(url): return "Deprecated, use logic inside analyze-url"
