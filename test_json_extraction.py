#!/usr/bin/env python3
"""
Test script for the improved JSON extraction functionality.
"""

import sys
import os

# Add the current directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from improved_llm_handler import ImprovedLLMHandler

def test_fallback_mechanism():
    """Test the fallback response mechanism."""
    print("\n🔄 Testing Fallback Response Mechanism...")
    
    handler = ImprovedLLMHandler()
    
    # Test cases that should trigger fallback
    test_cases = [
        # Case 1: Viral ideas prompt
        ("viral video idea", 'Here is the script: This is just text without JSON'),
        
        # Case 2: Script writing prompt  
        ("script", 'Here is the script: This is just text without JSON'),
        
        # Case 3: Generic prompt
        ("generic", 'Here is the script: This is just text without JSON'),
    ]
    
    results = []
    
    for i, (prompt_type, raw_text) in enumerate(test_cases, 1):
        print(f"\n📝 Fallback test case {i}:")
        print(f"Prompt type: {prompt_type}")
        print(f"Raw text: {raw_text[:50]}...")
        
        try:
            fallback_response = handler._create_fallback_response(prompt_type, raw_text)
            if fallback_response:
                print(f"✅ Fallback response created successfully")
                print(f"   Response type: {type(fallback_response)}")
                print(f"   Keys: {list(fallback_response.keys())}")
                
                # Check if it's a valid fallback response
                if "fallback_response" in fallback_response:
                    print(f"   ✅ Marked as fallback response")
                    results.append(("✅", f"Case {i}: Fallback created"))
                else:
                    print(f"   ⚠️ Not marked as fallback response")
                    results.append(("⚠️", f"Case {i}: Response created but not marked"))
            else:
                print("❌ No fallback response created")
                results.append(("❌", f"Case {i}: No fallback"))
                
        except Exception as e:
            print(f"❌ Error creating fallback: {e}")
            results.append(("❌", f"Case {i}: Exception - {e}"))
    
    # Summary
    print(f"\n📊 Fallback Test Summary:")
    print("=" * 50)
    for status, message in results:
        print(f"{status} {message}")
    
    success_count = sum(1 for status, _ in results if status in ["✅", "⚠️"])
    total_count = len(results)
    
    print(f"\n🎯 Overall: {success_count}/{total_count} fallback tests passed")
    
    if success_count == total_count:
        print("🎉 All fallback tests passed!")
        return True
    else:
        print("⚠️ Some fallback tests failed.")
        return False

def test_json_extraction():
    """Test the improved JSON extraction methods."""
    print("🧪 Testing Improved JSON Extraction...")
    
    handler = ImprovedLLMHandler()
    
    # Test cases that were failing before
    test_cases = [
        # Case 1: Response with prefix (exactly like the error in logs)
        'Here is the script: {"video_title": "The Mysterious Disappearance of the Mary Celeste", "target_duration_minutes": 15, "script": [{"sentence": "Test sentence", "visual_query": "test", "timing_seconds": 0, "engagement_hook": "test"}]}',
        
        # Case 2: Response with prefix and newlines
        '''Here is the script for the 15-20 minute video on "The Mysterious Disappearance of the Mary Celeste":

{"video_title": "The Mysterious Disappearance of the Mary Celeste", "target_duration_minutes": 15, "script": [{"sentence": "Test sentence", "visual_query": "test", "timing_seconds": 0, "engagement_hook": "test"}]}''',
        
        # Case 3: Malformed JSON that needs fixing
        'Here is the script: {"video_title": "Test", content: "Test content", "script": [{"sentence": "Test", visual_query: "test", timing_seconds: 0}]}',
        
        # Case 4: No JSON (should return None)
        'Here is the script: This is just text without any JSON content',
        
        # Case 5: Valid JSON without prefix
        '{"title": "Test", "content": "Test content"}',
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test case {i}:")
        print(f"Input: {test_case[:80]}...")
        
        try:
            result = handler._extract_json_from_text(test_case)
            if result:
                print(f"✅ Successfully extracted JSON ({len(result)} chars)")
                print(f"   Preview: {result[:100]}...")
                
                # Try to parse the extracted JSON
                import json
                try:
                    parsed = json.loads(result)
                    print(f"   ✅ JSON is valid and parseable")
                    results.append(("✅", f"Case {i}: Success"))
                except json.JSONDecodeError as e:
                    print(f"   ❌ Extracted text is not valid JSON: {e}")
                    results.append(("❌", f"Case {i}: Invalid JSON after extraction"))
                    
            else:
                print("❌ No JSON extracted")
                results.append(("❌", f"Case {i}: No extraction"))
                
        except Exception as e:
            print(f"❌ Error during extraction: {e}")
            results.append(("❌", f"Case {i}: Exception - {e}"))
    
    # Summary
    print(f"\n📊 Test Summary:")
    print("=" * 50)
    for status, message in results:
        print(f"{status} {message}")
    
    success_count = sum(1 for status, _ in results if status == "✅")
    total_count = len(results)
    
    print(f"\n🎯 Overall: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("🎉 All tests passed! JSON extraction is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. JSON extraction needs more work.")
        return False

if __name__ == "__main__":
    print("🚀 Starting Comprehensive JSON Extraction Tests...")
    
    # Test 1: JSON extraction
    extraction_success = test_json_extraction()
    
    # Test 2: Fallback mechanism
    fallback_success = test_fallback_mechanism()
    
    # Overall result
    print(f"\n🎯 FINAL RESULTS:")
    print("=" * 50)
    print(f"JSON Extraction: {'✅ PASSED' if extraction_success else '❌ FAILED'}")
    print(f"Fallback Mechanism: {'✅ PASSED' if fallback_success else '❌ FAILED'}")
    
    overall_success = extraction_success and fallback_success
    if overall_success:
        print("🎉 All tests passed! The system is ready for production.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    sys.exit(0 if overall_success else 1)
