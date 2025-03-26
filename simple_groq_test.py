"""
Simple script to validate basic Groq API connectivity.
"""

import os
from dotenv import load_dotenv
import requests


def test_groq_api_direct():
    """Test Groq API connectivity directly using requests."""
    print("\n===== Testing Groq API Connectivity =====")

    # Load environment variables
    load_dotenv()

    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment variables!")
        return False

    print(f"✓ GROQ_API_KEY found (starts with: {api_key[:4]}...)")

    # Test Groq API with a simple request
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",  # Updated to new model
            "messages": [{"role": "user", "content": "Hello! Say hi in 5 words or less."}],
            "temperature": 0.7,
            "max_tokens": 20,
        }

        print("Sending request to Groq API...")
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {}).get("content", "No content")
            print(f"✓ Response: {message}")
            print("✓ Groq API connection successful!")
            return True
        else:
            print(f"❌ Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error connecting to Groq API: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting simple Groq API validation test...")
    result = test_groq_api_direct()

    print("\n===== TEST RESULTS =====")
    if result:
        print("✅ Groq API test PASSED!")
        print("The Groq API is working properly.")
    else:
        print("❌ Groq API test FAILED!")
        print("- Check your API key, network connection, and Groq service status.")
        print("- Visit https://status.groq.com/ to verify the Groq service is operational.")
