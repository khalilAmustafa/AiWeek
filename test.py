import base64
import requests

with open("test.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post("http://localhost:5000/authenticate", json={"image": img_base64})
print(response.json())