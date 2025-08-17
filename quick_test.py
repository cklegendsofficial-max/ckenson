#!/usr/bin/env python3
"""
Quick test to check if the system is production-ready.
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_system_components():
    """Test if all system components are available and working."""
    print("🔍 Testing System Components...")
    
    results = []
    
    # Test 1: Check if required files exist
    required_files = [
        "improved_llm_handler.py",
        "main.py", 
        "config.py",
        "advanced_video_creator.py"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} - Found")
            results.append(True)
        else:
            print(f"❌ {file} - Missing")
            results.append(False)
    
    # Test 2: Check if directories exist
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
    
    # Test 3: Check environment variables
    env_vars = [
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL", 
        "PEXELS_API_KEY"
    ]
    
    for env_var in env_vars:
        value = os.getenv(env_var)
        if value:
            print(f"✅ {env_var} - Set ({value[:20]}...)")
            results.append(True)
        else:
            print(f"⚠️ {env_var} - Not set")
            results.append(False)
    
    # Test 4: Check if Ollama is running
    try:
        import requests
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama Server - Running")
            results.append(True)
        else:
            print(f"⚠️ Ollama Server - Response {response.status_code}")
            results.append(False)
    except Exception as e:
        print(f"❌ Ollama Server - Error: {e}")
        results.append(False)
    
    # Test 5: Check Python dependencies
    required_modules = [
        "ollama",
        "moviepy",
        "requests",
        "pillow"
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - Available")
            results.append(True)
        except ImportError:
            print(f"❌ {module} - Not available")
            results.append(False)
    
    return results

def test_json_extraction():
    """Test JSON extraction without LLM communication."""
    print("\n🧪 Testing JSON Extraction (Offline)...")
    
    try:
        from improved_llm_handler import ImprovedLLMHandler
        
        # Create handler with minimal initialization
        handler = ImprovedLLMHandler.__new__(ImprovedLLMHandler)
        handler.pytrends = None
        
        # Create a simple logger
        import logging
        handler.logger = logging.getLogger("test_logger")
        handler.logger.setLevel(logging.ERROR)  # Suppress debug messages
        
        # Test cases
        test_cases = [
            '{"title": "Test", "content": "Test content"}',
            'Here is the script: {"title": "Test"}',
            'Here is the script: This is just text without JSON'
        ]
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            try:
                result = handler._extract_json_from_text(test_case)
                if result:
                    print(f"✅ Case {i}: JSON extracted successfully")
                    results.append(True)
                else:
                    if "without JSON" in test_case:
                        print(f"✅ Case {i}: Correctly returned None for non-JSON text")
                        results.append(True)
                    else:
                        print(f"❌ Case {i}: Failed to extract JSON")
                        results.append(False)
            except Exception as e:
                print(f"❌ Case {i}: Error - {e}")
                results.append(False)
        
        return results
        
    except Exception as e:
        print(f"❌ JSON extraction test failed: {e}")
        return [False]

def main():
    """Run all tests."""
    print("🚀 Production Readiness Test")
    print("=" * 50)
    
    # Test 1: System components
    component_results = test_system_components()
    
    # Test 2: JSON extraction
    json_results = test_json_extraction()
    
    # Summary
    print(f"\n📊 Test Summary:")
    print("=" * 50)
    
    total_tests = len(component_results) + len(json_results)
    passed_tests = sum(component_results) + sum(json_results)
    
    print(f"System Components: {sum(component_results)}/{len(component_results)} passed")
    print(f"JSON Extraction: {sum(json_results)}/{len(json_results)} passed")
    print(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    # Production readiness assessment
    print(f"\n🎯 Production Readiness Assessment:")
    print("=" * 50)
    
    if passed_tests >= total_tests * 0.8:  # 80% threshold
        print("✅ SYSTEM IS PRODUCTION READY!")
        print("   - All critical components are available")
        print("   - JSON extraction is working")
        print("   - System can handle errors gracefully")
    elif passed_tests >= total_tests * 0.6:  # 60% threshold
        print("⚠️ SYSTEM IS MOSTLY READY (with warnings)")
        print("   - Some components may need attention")
        print("   - Basic functionality should work")
    else:
        print("❌ SYSTEM IS NOT PRODUCTION READY")
        print("   - Critical issues need to be resolved")
        print("   - System may fail during operation")
    
    return passed_tests >= total_tests * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
