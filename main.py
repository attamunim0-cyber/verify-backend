from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI()
DB_NAME = "verify_shield_v2.db"
stripe.api_key = "sk_test_51Q2kAoCFpfwvg3QdGGP7uAibYjbJCv9mqorL582t1Tp2uXtGcNLgyAFRfqYN8eNqpnLhvOAk5zNGkfN4wDp4QtR000JjAFj72n" # <--- ENSURE KEY IS HERE

# --- EMAIL CREDENTIALS ---
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
# The URL of your server (used for the reset link)
SERVER_URL = "https://verify-backend-03or.onrender.com" 

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Add reset_token column if not exists
        c.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
    except: pass
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, password TEXT, verification_code TEXT, is_verified INTEGER DEFAULT 1, scan_count INTEGER DEFAULT 0, is_premium INTEGER DEFAULT 0, reset_token TEXT)''')
    conn.commit()
    conn.close()
init_db()

# --- MODELS ---
class UserSignup(BaseModel): name: str; email: str; password: str
class UserLogin(BaseModel): email: str; password: str
class PaymentRequest(BaseModel): email: str
class UrlRequest(BaseModel): url: str; email: str
class ForgotPasswordRequest(BaseModel): email: str

# --- HELPERS ---
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def send_reset_email(to_email, token):
    if not EMAIL_USER or not EMAIL_PASS:
        print("⚠️ Email credentials missing in Render Environment.")
        return False
    
    reset_link = f"{SERVER_URL}/reset-password-view?token={token}"
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = "Reset Your Password - Verify Shield"
    
    body = f"""
    You requested a password reset.
    
    Click the link below to set a new password:
    {reset_link}
    
    If you did not request this, please ignore this email.
    """
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email Failed: {e}")
        return False

# --- NEW: PASSWORD RESET ENDPOINTS ---

@app.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    user = c.execute("SELECT * FROM users WHERE email=?", (req.email,)).fetchone()
    if not user:
        conn.close()
        # Pretend it worked for security (don't reveal if email exists)
        return {"status": "success", "message": "If email exists, link sent."}
    
    # Generate unique token
    token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    c.execute("UPDATE users SET reset_token=? WHERE email=?", (token, req.email))
    conn.commit()
    conn.close()
    
    send_reset_email(req.email, token)
    return {"status": "success", "message": "Reset link sent to your email."}

@app.get("/reset-password-view", response_class=HTMLResponse)
async def reset_password_view(token: str):
    # Simple HTML Form served directly by Python
    html_content = f"""
    <html>
        <head>
            <title>Reset Password</title>
            <style>
                body {{ font-family: sans-serif; background: #0A0E21; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; }}
                .card {{ background: #1D1E33; padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #00E676; }}
                input {{ padding: 10px; border-radius: 5px; border: none; width: 100%; margin-bottom: 15px; }}
                button {{ background: #00E676; padding: 10px 20px; border: none; border-radius: 5px; font-weight: bold; cursor: pointer; width: 100%; }}
                h2 {{ color: #00E676; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h2>Verify Shield</h2>
                <p>Enter your new password below.</p>
                <form action="/reset-password-action" method="post">
                    <input type="hidden" name="token" value="{token}">
                    <input type="password" name="new_password" placeholder="New Password" required>
                    <button type="submit">Update Password</button>
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/reset-password-action", response_class=HTMLResponse)
async def reset_password_action(token: str = Form(...), new_password: str = Form(...)):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    # Find user by token
    user = c.execute("SELECT email FROM users WHERE reset_token=?", (token,)).fetchone()
    
    if not user:
        conn.close()
        return HTMLResponse("<h1>Error: Invalid or expired link.</h1>")
    
    # Update Password and clear token
    new_hash = hash_password(new_password)
    c.execute("UPDATE users SET password=?, reset_token=NULL WHERE email=?", (new_hash, user[0]))
    conn.commit()
    conn.close()
    
    return HTMLResponse("""
        <body style='background:#0A0E21; color:white; text-align:center; padding-top:50px; font-family:sans-serif;'>
            <h1 style='color:#00E676'>Success!</h1>
            <p>Your password has been reset.</p>
            <p>You can now return to the app and login.</p>
        </body>
    """)

# --- UPDATED SIGNUP (AUTO VERIFY) ---
@app.post("/signup")
async def signup(user: UserSignup):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    try:
        if c.execute("SELECT * FROM users WHERE email=?", (user.email,)).fetchone(): 
            return JSONResponse(content={"status":"error", "message":"Email taken"}, status_code=400)
        
        # NOTE: is_verified is set to 1 (True) automatically now
        c.execute("INSERT INTO users (email, name, password, is_verified) VALUES (?,?,?, 1)", 
                  (user.email, user.name, hash_password(user.password)))
        conn.commit()
        
        return {"status": "success", "message": "Account created. Please Login."}
    finally: conn.close()

# --- OTHER STANDARD ENDPOINTS ---
def check_limits(email):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    user = c.execute("SELECT scan_count, is_premium FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not user: return False
    if not user[1] and user[0] >= 3: return False
    if not user[1]:
        conn = sqlite3.connect(DB_NAME); c = conn.cursor()
        c.execute("UPDATE users SET scan_count = scan_count + 1 WHERE email=?", (email,)); conn.commit(); conn.close()
    return True

@app.post("/login")
async def login(user: UserLogin):
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    row = c.execute("SELECT name, is_verified, scan_count, is_premium FROM users WHERE email=? AND password=?", (user.email, hash_password(user.password))).fetchone()
    conn.close()
    if row: return {"status": "success", "username": row[0], "scan_count": row[2], "is_premium": bool(row[3])}
    return JSONResponse(content={"status": "error", "message": "Invalid credentials"}, status_code=401)

# (KEEP THE DOWNLOADER AND ANALYZE FUNCTIONS FROM PREVIOUS STEPS HERE...)
# To save space, I assume you keep the 'download_video_site', 'download_direct', 'analyze_media' 
# and the '/analyze', '/analyze-url' logic exactly as I gave you in the last step.
# If you need the full file again, let me know.
