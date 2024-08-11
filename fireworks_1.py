import os
import base64
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Firefly API details
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path,recipe_name):
    image_base64 = encode_image(image_path)

    prompt = f"""
           You are given an image of the user's food.
           They say it's a {recipe_name}.
           Figure out what kind of food is in the image (and its nutrition facts) and adjust the user's nutrition plan to include the food items in the image.
           Respond back to the user using <message></message> tags briefly explaining to them how you changed their nutrition plan based on the image they uploaded.
           Figure out what type of ingredients are in the image, provide description, and also provide the updated nutrition plan.
        """.format(image_base64=image_base64)

    payload = {
        "model": "accounts/fireworks/models/firellava-13b",
        "temperature": 0.3,
        "max_tokens": 8512,
        "messages": [{
            "role": "user",
            "content": prompt
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

if __name__ == "__main__":
    test_image_path = "your_image.jpg"
    test_recipe_name = "your_recipe_name"
    explanation = get_image_description(test_image_path, test_recipe_name)
    print(explanation)
