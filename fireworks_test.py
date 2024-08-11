import base64
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Firefly API details
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path):
    image_base64 = encode_image(image_path)

    payload = {
        "model": "accounts/fireworks/models/firellava-13b",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Can you describe this image?",
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            }],
        }]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No description found.")
    else:
        return f"Error: {response.status_code} - {response.text}"
