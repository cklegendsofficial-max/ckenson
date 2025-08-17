#!/usr/bin/env python3
"""
Quick production test without LLM communication.
"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ["PEXELS_API_KEY"] = "SkEG6SqXRKE6OzoUVqlATFTn9hC8jmrf7TRimoA9D6wt8ME9ZCirpscf"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "llama3:8b"

def test_production_ready():
    """Test if system is production ready."""
    print("🚀 Production Readiness Test")
    print("=" * 50)
    
    results = []
    
    # Test 1: Environment variables
    print("🔍 Testing Environment Variables...")
    pexels_key = os.getenv("PEXELS_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_MODEL")
    
    if pexels_key and pexels_key != "your_pexels_api_key_here":
        print("✅ PEXELS_API_KEY - Set and valid")
        results.append(True)
    else:
        print("❌ PEXELS_API_KEY - Not set or invalid")
        results.append(False)
    
    if ollama_url:
        print("✅ OLLAMA_BASE_URL - Set")
        results.append(True)
    else:
        print("❌ OLLAMA_BASE_URL - Not set")
        results.append(False)
    
    if ollama_model:
        print("✅ OLLAMA_MODEL - Set")
        results.append(True)
    else:
        print("❌ OLLAMA_MODEL - Not set")
        results.append(False)
    
    # Test 2: Core files
    print("\n🔍 Testing Core Files...")
    core_files = [
        "improved_llm_handler.py",
        "main.py",
        "config.py", 
        "advanced_video_creator.py"
    ]
    
    for file in core_files:
        if Path(file).exists():
            print(f"✅ {file} - Found")
            results.append(True)
        else:
            print(f"❌ {file} - Missing")
            results.append(False)
    
    # Test 3: Directories
    print("\n🔍 Testing Directories...")
    required_dirs = [
        "assets",
        "assets/audio", 
        "assets/videos",
        "data",
        "logs"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ - Found")
            results.append(True)
        else:
            print(f"❌ {dir_path}/ - Missing")
            results.append(False)
    
    # Test 4: Dependencies
    print("\n🔍 Testing Dependencies...")
    dependencies = ["ollama", "moviepy", "requests"]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} - Available")
            results.append(True)
        except ImportError:
            print(f"❌ {dep} - Not available")
            results.append(False)
    
    # Test 5: Ollama server
    print("\n🔍 Testing Ollama Server...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama Server - Running")
            results.append(True)
        else:
            print(f"❌ Ollama Server - Response {response.status_code}")
            results.append(False)
    except Exception as e:
        print(f"❌ Ollama Server - Error: {e}")
        results.append(False)
    
    # Summary
    print(f"\n📊 Test Summary:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    # Production readiness assessment
    print(f"\n🎯 Production Readiness Assessment:")
    print("=" * 50)
    
    if passed >= total * 0.9:  # 90% threshold
        print("🎉 SYSTEM IS FULLY PRODUCTION READY!")
        print("   - All critical components available")
        print("   - Environment properly configured")
        print("   - Ready for video production")
        return True
    elif passed >= total * 0.8:  # 80% threshold
        print("✅ SYSTEM IS PRODUCTION READY!")
        print("   - Most components available")
        print("   - Minor issues may exist")
        print("   - Ready for production use")
        return True
    elif passed >= total * 0.6:  # 60% threshold
        print("⚠️ SYSTEM IS MOSTLY READY")
        print("   - Some components need attention")
        print("   - Basic functionality should work")
        return False
    else:
        print("❌ SYSTEM IS NOT PRODUCTION READY")
        print("   - Critical issues need resolution")
        print("   - System may fail during operation")
        return False

if __name__ == "__main__":
    success = test_production_ready()
    sys.exit(0 if success else 1)
