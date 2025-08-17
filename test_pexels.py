import os 
import requests 
from dotenv import load_dotenv 
 
load_dotenv() 
 
api_key = os.getenv("PEXELS_API_KEY") 
print(f"API Key: {api_key}") 
