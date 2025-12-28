from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests
import boto3
import os
import shutil

app = FastAPI()

# LOAD KEYS (Render will provide these secretly)
SIGHTENGINE_USER = os.environ.get("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_SECRET")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

# CONNECT TO AWS
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name='ap-southeast-2' # Adjusted to your screenshot (Sydney)
)

@app.get("/")
def home():
    return {"status": "Verify Shield is Online"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # 1. SAVE FILE LOCALLY (Temporary)
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. UPLOAD TO AWS S3
        s3_client.upload_file(temp_filename, S3_BUCKET_NAME, file.filename)

        # 3. GENERATE SECURE LINK (Valid for 5 mins)
        file_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': file.filename},
            ExpiresIn=300
        )

        # 4. ASK SIGHTENGINE TO SCAN
        params = {
            'models': 'deepfake',
            'api_user': SIGHTENGINE_USER,
            'api_secret': SIGHTENGINE_SECRET,
            'url': file_url
        }
        response = requests.get('https://api.sightengine.com/1.0/check.json', params=params)
        data = response.json()

        # 5. CLEAN UP (Delete local temp file)
        os.remove(temp_filename)

        # 6. RETURN RESULT
        if data['status'] == 'success':
            fake_score = data['type']['deepfake']
            return {
                "status": "success",
                "score": fake_score, 
                "verdict": "FAKE" if fake_score > 0.5 else "REAL" 
            }
        else:
            return {"status": "error", "message": "AI Engine failed"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
