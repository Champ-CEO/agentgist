"""
Test script to validate Groq API connectivity and model configurations.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agentgist.groq_strategies import OptimizedGroqChat
from agentgist.config import LLAMA_3_3, DEEPSEEK_R1
from langchain_core.messages import HumanMessage


def test_groq_api():
    """Test standard Groq API connectivity."""
    print("\n===== Testing Standard Groq API Connectivity =====")

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment variables!")
        return False

    print(f"✓ GROQ_API_KEY found (starts with: {os.getenv('GROQ_API_KEY')[:4]}...)")

    # Test basic Groq model
    try:
        model_name = LLAMA_3_3.name
        print(f"Testing model: {model_name}")

        chat = ChatGroq(model=model_name, temperature=0.1)
        response = chat.invoke([HumanMessage(content="Say hello in 5 words or less.")])

        print(f"✓ Response: {response.content}")
        print("✓ Basic Groq API test successful!")
        return True
    except Exception as e:
        print(f"❌ Error with standard Groq API: {str(e)}")
        return False


def test_optimized_groq():
    """Test our Optimized Groq implementation."""
    print("\n===== Testing Optimized Groq Implementation =====")

    try:
        # Test both models
        models = [
            ("General purpose", LLAMA_3_3.name, LLAMA_3_3.temperature),
            ("Report writer", DEEPSEEK_R1.name, DEEPSEEK_R1.temperature),
        ]

        for purpose, model_name, temp in models:
            print(f"\nTesting {purpose} model: {model_name}")

            chat = OptimizedGroqChat(model=model_name, temperature=temp, optimize_tokens=True)

            response = chat.invoke(
                [
                    HumanMessage(
                        content="Summarize the key features of AgentGist in three sentences."
                    )
                ]
            )

            print(f"✓ Response: {response.content}")

        print("\n✓ Optimized Groq implementation test successful!")
        return True
    except Exception as e:
        print(f"❌ Error with optimized Groq implementation: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting Groq API validation tests...")
    load_dotenv()  # Load environment variables from .env file

    standard_test = test_groq_api()
    optimized_test = test_optimized_groq() if standard_test else False

    print("\n===== TEST RESULTS =====")
    if standard_test and optimized_test:
        print("✅ All Groq API tests PASSED!")
        print("The Groq API is working properly and both models are accessible.")
    else:
        print("❌ Some tests FAILED!")
        if not standard_test:
            print(
                "- Basic Groq API connectivity issues. Check your API key and network connection."
            )
        if standard_test and not optimized_test:
            print("- Optimized Groq implementation issues. Check the implementation code.")
