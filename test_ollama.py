#!/usr/bin/env python3
"""
Test Ollama communication and identify issues.
"""

import os
import sys
import time

# Set environment variables
os.environ["PEXELS_API_KEY"] = "SkEG6SqXRKE6OzoUVqlATFTn9hC8jmrf7TRimoA9D6wt8ME9ZCirpscf"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "llama3:8b"

def test_ollama_import():
    """Test if ollama module can be imported."""
    print("🔍 Testing Ollama import...")
    try:
        import ollama
        print("✅ Ollama module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Ollama import failed: {e}")
        return False

def test_ollama_connection():
    """Test basic Ollama connection."""
    print("\n🔍 Testing Ollama connection...")
    try:
        import ollama
        
        # Test basic connection
        print("Testing connection to Ollama server...")
        response = ollama.list()
        print(f"✅ Ollama connection successful")
        print(f"Available models: {[model['name'] for model in response['models']]}")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_ollama_chat():
    """Test Ollama chat functionality."""
    print("\n🔍 Testing Ollama chat...")
    try:
        import ollama
        
        # Simple test prompt
        test_prompt = "Hello, please respond with just 'OK'"
        print(f"Sending test prompt: {test_prompt}")
        
        start_time = time.time()
        response = ollama.chat(
            model="llama3:8b",
            messages=[{"role": "user", "content": test_prompt}]
        )
        end_time = time.time()
        
        response_content = response.get("message", {}).get("content", "")
        print(f"✅ Chat successful in {end_time - start_time:.2f}s")
        print(f"Response: {response_content[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Ollama chat failed: {e}")
        return False

def test_llm_handler():
    """Test the LLM handler with Ollama."""
    print("\n🔍 Testing LLM Handler...")
    try:
        from improved_llm_handler import ImprovedLLMHandler
        
        print("Creating LLM handler...")
        handler = ImprovedLLMHandler("llama3:8b")
        
        print("Testing simple prompt...")
        test_prompt = "Generate a simple JSON response with this exact format: {\"status\": \"ok\", \"message\": \"test successful\"}"
        
        start_time = time.time()
        response = handler._get_ollama_response(test_prompt)
        end_time = time.time()
        
        if response:
            print(f"✅ LLM Handler test successful in {end_time - start_time:.2f}s")
            print(f"Response: {response}")
            return True
        else:
            print("❌ LLM Handler returned None")
            return False
            
    except Exception as e:
        print(f"❌ LLM Handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Ollama tests."""
    print("🚀 Ollama Integration Test")
    print("=" * 50)
    
    tests = [
        ("Ollama Import", test_ollama_import),
        ("Ollama Connection", test_ollama_connection),
        ("Ollama Chat", test_ollama_chat),
        ("LLM Handler", test_llm_handler)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ollama integration is working.")
        return True
    else:
        print("⚠️ Some tests failed. Ollama integration needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
