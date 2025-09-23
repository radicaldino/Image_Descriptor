import os
from dotenv import load_dotenv
load_dotenv()

PORT = int(os.getenv("PORT",))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


# openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
