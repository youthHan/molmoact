# server.py

import base64
import io
import math
import argparse
from typing import List, Optional

import numpy as np
import torch
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import uvicorn

# --- Helper Functions for Model ---

def crop_and_resize_pil(img: Image.Image, crop_scale: float) -> Image.Image:
    """Center-crop a PIL image and resize back to original size."""
    w, h = img.size
    rel = math.sqrt(crop_scale)
    cw, ch = int(w * rel), int(h * rel)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BILINEAR)

def center_crop_image(img: Image.Image) -> Image.Image:
    """Center-crop image to a fixed 0.9 area scale."""
    return crop_and_resize_pil(img, 0.9)

# --- FastAPI App Setup ---

app = FastAPI(title="Robot Action Generation Server")

# --- Global Model Loading ---
# This part runs only once when the server starts.
# You must edit the CHECKPOINT variable to point to your model checkpoint.
parser = argparse.ArgumentParser(description="Run the action generation server.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
parser.add_argument("--worker_id", type=str, required=True, help="Path to the model checkpoint.")
args = parser.parse_args()

CHECKPOINT = args.checkpoint
model = None
processor = None

@app.on_event("startup")
def load_model():
    global model, processor
    print(f"Loading model and processor from: {CHECKPOINT}...")
    try:
        processor = AutoProcessor.from_pretrained(
            CHECKPOINT, trust_remote_code=True, torch_dtype="bfloat16", 
            device_map="auto", padding_side="left"
        )
        model = AutoModelForImageTextToText.from_pretrained(
            CHECKPOINT, trust_remote_code=True, torch_dtype="bfloat16", device_map="auto"
        )
        print("✅ Model and processor loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Server will start, but requests will fail. Please check the checkpoint path and dependencies.")

# --- Data Models for API ---

class ActionRequest(BaseModel):
    img_b64: str
    wrist_img_b64: str
    language_instruction: str
    unnorm_key: str

class ActionResponse(BaseModel):
    action: Optional[List[List[float]]]
    annotated_image_b64: str
    trace: Optional[List[List[int]]]
    error: Optional[str] = None


# --- Core Logic (The original 'step' function, adapted for the server) ---

def step_server(img, wrist_img, language_instruction, unnorm_key):
    """
    Runs the model to generate actions based on observations.
    """
    image = Image.fromarray(img)
    wrist = Image.fromarray(wrist_img)
    image = center_crop_image(image)
    wrist = center_crop_image(wrist)
    imgs = [image, wrist]

    prompt = (
        f"The task is {language_instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {language_instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for the first image? "
        "Second, what is the trajectory of the end effector in the first image? "
        "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
        "along with other images from different camera views as additional information, "
        "what is the action that the robot should take?"
    )
        
    text = processor.apply_chat_template(
        [{"role": "user", "content": [dict(type="text", text=prompt)]}], 
        tokenize=False, add_generation_prompt=True
    )
        
    inputs = processor(images=[imgs], text=text, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Generated text: {generated_text}")

    trace = model.parse_trace(generated_text)
    action = model.parse_action(generated_text, unnorm_key=unnorm_key)

    if action is None or (isinstance(action, (list, np.ndarray)) and len(action) == 0):
        raise ValueError("parse_action produced no action.")
        
    annotated = np.array(img.copy())
    return action, annotated, trace


# --- API Endpoint ---

@app.post("/generate_action", response_model=ActionResponse)
async def generate_action(request: ActionRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check server logs.")

    try:
        # Decode base64 images to numpy arrays (RGB)
        img_bytes = base64.b64decode(request.img_b64)
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        wrist_img_bytes = base64.b64decode(request.wrist_img_b64)
        wrist_img_np = cv2.imdecode(np.frombuffer(wrist_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        wrist_img_rgb = cv2.cvtColor(wrist_img_np, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {e}")

    try:
        action, annotated_image_np, trace = step_server(
            img_rgb, wrist_img_rgb, request.language_instruction, request.unnorm_key
        )

        # Encode annotated image back to base64
        annotated_bgr = cv2.cvtColor(annotated_image_np, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', annotated_bgr)
        annotated_img_b64 = base64.b64encode(buffer).decode('utf-8')

        action_list = action.tolist() if isinstance(action, np.ndarray) else action
        trace_list = trace.tolist() if isinstance(trace, np.ndarray) else trace

        return ActionResponse(
            action=action_list,
            annotated_image_b64=annotated_img_b64,
            trace=trace_list[0] if len(trace_list) >= 1 and isinstance(trace_list[0], list) and isinstance(trace_list[0][0], list)  else trace_list
        )
    except Exception as e:
        print(f"Error during model inference: {e}")
        # Return the original image on failure to prevent client from crashing
        return ActionResponse(
            action=None,
            annotated_image_b64=request.img_b64,
            trace=None,
            error=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000+int(args.worker_id))