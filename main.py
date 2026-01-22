"""Main entry point for the AI QA Validator API."""

import uvicorn
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

def validate_startup():
    """Validate required environment variables on startup."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key or gemini_api_key == "your_api_key_here":
        print("ERROR: GEMINI_API_KEY environment variable is not configured.", file=sys.stderr)
        print("Please set GEMINI_API_KEY in your .env file or environment.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Validate configuration before starting server
    validate_startup()
    
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
