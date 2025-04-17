from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Debug log to verify if the API key is loaded
print(f"GROQ_API_KEY: {GROQ_API_KEY}")