import io
import base64
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = r"C:\Users\khali\Desktop\face_auth_app\resnet50_finetuned_faces.pth"
AUTHORIZED_FILE = r"C:\Users\khali\Desktop\face_auth_app\dataset.csv"
NUM_CLASSES = 105

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# -----------------------------
# Load CSV and create mapping
# -----------------------------
df = pd.read_csv(AUTHORIZED_FILE)
df.columns = [c.strip().lower() for c in df.columns]  # normalize column names

# Use iterrows() to map model class index -> name & status
class_to_info = {i: {'name': row['name'], 'status': row['status'].lower()}
                 for i, row in df.iterrows()}

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

@app.route("/authenticate", methods=["POST"])
def authenticate():
    try:
        if "image" in request.files:
            file = request.files["image"]
            img = Image.open(file).convert("RGB")
        else:
            data = request.get_json()
            if not data or "image" not in data:
                return jsonify({"status": "error", "message": "No image provided"})
            img_bytes = base64.b64decode(data["image"])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            pred_idx = pred.item()

        info = class_to_info.get(pred_idx)
        if info:
            name = info['name']
            status = info['status']
        else:
            name = f"Person_{pred_idx}"
            status = "unknown"

        return jsonify({
            "status": "success",
            "prediction": name,
            "authorized": status
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
