firectl import model llava-yi-34b
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

if FIREWORKS_API_KEY:
    print("API key successfully read from .env file.")
else:
    print("Failed to read API key.")

# Example usage of the API key
print(f"The API key is: {FIREWORKS_API_KEY}")


#create deployment
firectl create deployment llava-yi-34b --wait
