from fastapi import FastAPI, HTTPException, UploadFile, File
import io
import torch
import torchvision.models.convnext
from torchvision import transforms
from PIL import Image
import json
import re
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

ENTROPY_THRESHOLD = 2.0

# Extracting Class Names of the Model
# Plant Model
with open(BASE_DIR / "data" / "plant_class_names.json", "r") as f:
    plant_class_names = json.load(f)
plant_class_names = [
    re.sub(r"[^a-zA-Z]+", " ", plt).title() for plt in plant_class_names
]

torch.serialization.add_safe_globals([torchvision.models.convnext.ConvNeXt])

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

plant_model = torch.load(
    BASE_DIR / "backend" / "models" / "plant_pre_trained_model.pth",
    map_location=device,
    weights_only=False,
)
plant_model.eval()


def run_inference_mode(model, image_bytes, class_names):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transformed_img = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        preds_logits = model(transformed_img)
        probs = F.softmax(preds_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)

        if entropy.item() > ENTROPY_THRESHOLD:
            return {
                "status": "Uncertain",
                "prediction": "High uncertainty detected. This image does not match known classes.",
                "entropy": entropy.item(),
            }
        pred_label = preds_logits.argmax(dim=1)
        return {
            "status": "Success",
            "prediction": class_names[pred_label],
            "entropy": entropy.item(),
        }


@app.get("/")
def http_check():
    return {"message": "The server is working"}


@app.post("/predict/plant")
async def predict_plant(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return run_inference_mode(plant_model, image_bytes, plant_class_names)


@app.get("/predict/logs")
def get_prediction_logs():
    log_file = BASE_DIR / "data" / "prediction_logs.json"

    if not log_file.exists() or log_file.stat().st_size == 0:
        return []

    with open(log_file, "r", encoding="utf-8") as fp:
        logs = json.load(fp)

    return logs


@app.post("/predict/logs")
def predict_logs(data: Dict[str, Any]):
    try:
        data_dir = BASE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        log_file = data_dir / "prediction_logs.json"

        logs = []

        if log_file.exists() and log_file.stat().st_size > 0:
            try:
                with open(log_file, "r", encoding="utf-8") as fp:
                    content = json.load(fp)
                    if isinstance(content, list):
                        logs = content
                    elif content is not None:
                        logs = [content]
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading log file, resetting logs: {e}")
                logs = []

        logs.append(data)

        with open(log_file, "w", encoding="utf-8") as fp:
            json.dump(logs, fp, indent=4)

        return data

    except Exception as e:
        print(f"CRITICAL BACKEND ERROR: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
