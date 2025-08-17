# Load environment variables from .env file 
try: 
    from dotenv import load_dotenv 
    load_dotenv() 
    print("? .env file loaded successfully") 
except Exception as e: 
    print(f"?? Failed to load .env: {e}") 
