from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import os
from typing import List, Optional
from io import BytesIO
from PIL import Image
import cv2
import tensorflow as tf

app = FastAPI(title="Face Emotion Detection API")

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.h5")
CLASS_NAMES_ENV = os.environ.get("CLASS_NAMES")  # comma separated
CLASS_NAMES_FILE = "models/class_names.txt"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load class names
def load_class_names() -> Optional[List[str]]:
    if CLASS_NAMES_ENV:
        return [c.strip() for c in CLASS_NAMES_ENV.split(",") if c.strip()]
    if os.path.exists(CLASS_NAMES_FILE):
        with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return None

CLASS_NAMES = load_class_names()

# Load model (lazy, to surface helpful error on startup)
MODEL = None
def load_model(path=MODEL_PATH):
    global MODEL
    if MODEL is not None:
        return MODEL
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Please place your Keras model there.")
    MODEL = tf.keras.models.load_model(path)
    return MODEL

# Preprocess function - adjust if your model expects different input
def preprocess_image(img: Image.Image) -> np.ndarray:
    # Convert to grayscale, detect face, crop to face, resize to 48x48
    # If face detection fails, use the whole image.
    # Normalize to [0,1] and add batch/channel dims (1,48,48,1)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) > 0:
        # choose the largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face = gray[y : y + h, x : x + w]
    else:
        face = gray
    # Resize to 48x48 (common for emotion models). If your model uses color or different size, change here.
    face_resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    face_normalized = face_resized.astype("float32") / 255.0
    # Add channel and batch dimension
    arr = np.expand_dims(face_normalized, axis=-1)  # (48,48,1)
    arr = np.expand_dims(arr, axis=0)  # (1,48,48,1)
    return arr

@app.on_event("startup")
def startup_event():
    # Try loading model early so startup fails fast if missing
    try:
        _ = load_model()
        app.state.model_loaded = True
    except Exception as e:
        # Keep server running but mark model as not loaded and log message
        app.state.model_loaded = False
        app.state.model_error = str(e)
        print(f"Warning: model did not load on startup: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file (jpg/png) and returns predicted emotion label and probabilities.
    """
    if not app.state.model_loaded:
        raise HTTPException(status_code=500, detail={"error": "Model not loaded", "message": app.state.model_error})
    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    try:
        x = preprocess_image(img)  # shape (1,H,W,C)
        model = load_model()
        preds = model.predict(x)
        # preds shape can be (1, num_classes) or other; we handle common case
        if isinstance(preds, list):
            # models that return multiple outputs -> take first
            preds = preds[0]
        preds = np.asarray(preds).squeeze()
        # Normalize if outputs are logits: attempt softmax if not already sum=1
        if preds.ndim == 0:
            # single value output
            result = {"predictions": float(preds)}
            return JSONResponse(result)
        # If negative values present or sum not ~1, apply softmax
        if np.any(preds < 0) or not np.isclose(np.sum(preds), 1.0, atol=1e-3):
            exp = np.exp(preds - np.max(preds))
            probs = exp / np.sum(exp)
        else:
            probs = preds / np.sum(preds)
        probs_list = [float(p) for p in probs]
        top_idx = int(np.argmax(probs))
        label = None
        if CLASS_NAMES and top_idx < len(CLASS_NAMES):
            label = CLASS_NAMES[top_idx]
        result = {"predictions": probs_list, "label": label}
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok", "model_loaded": bool(app.state.model_loaded)}
