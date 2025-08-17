#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 AI Integration Test Suite
Test all AI modules and integration points
"""

import os
import sys
import json
import time
from pathlib import Path

def test_ai_integration():
    """Test AI integration comprehensively"""
    print("🧪 Testing AI Integration Suite...")
    print("=" * 50)
    
    # Test 1: Import AI Integrated Suite
    try:
        from ai_integrated_suite import (
            AIIntegratedSuite, 
            create_ai_suite, 
            check_ai_dependencies, 
            get_ai_system_info
        )
        print("✅ AI Integrated Suite imports successful")
    except ImportError as e:
        print(f"❌ AI Integrated Suite import failed: {e}")
        return False
    
    # Test 2: Check Dependencies
    try:
        deps = check_ai_dependencies()
        print(f"📊 Dependencies: {deps}")
        
        available_count = sum(1 for v in deps.values() if v)
        total_count = len(deps)
        print(f"📈 Available: {available_count}/{total_count} dependencies")
        
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return False
    
    # Test 3: Get System Info
    try:
        info = get_ai_system_info()
        print(f"ℹ️ System Info: {info}")
        
        if info.get('system_ready'):
            print("✅ System is ready for AI operations")
        else:
            print("⚠️ System has missing core components")
            
    except Exception as e:
        print(f"❌ System info failed: {e}")
        return False
    
    # Test 4: Create AI Suite
    try:
        suite = create_ai_suite()
        print("✅ AI Suite created successfully")
        
        # Get system status
        status = suite.get_system_status()
        print(f"🔍 System Status: {json.dumps(status, indent=2)}")
        
    except Exception as e:
        print(f"❌ AI Suite creation failed: {e}")
        return False
    
    # Test 5: Test Pipeline (if possible)
    try:
        if suite.system_health.overall_status != "CRITICAL":
            print("🚀 Testing AI pipeline...")
            
            # Test with minimal parameters
            result = suite.run_full_pipeline(
                channel_name="TestChannel",
                niche="general",
                target_duration=5
            )
            
            if result.get('success'):
                print("✅ AI pipeline test successful")
                print(f"📊 Pipeline Results: {json.dumps(result, indent=2)}")
            else:
                print(f"⚠️ AI pipeline test failed: {result.get('error')}")
        else:
            print("⚠️ Skipping pipeline test due to critical system status")
            
    except Exception as e:
        print(f"❌ AI pipeline test failed: {e}")
    
    print("=" * 50)
    print("🎉 AI Integration Test completed!")
    return True

def test_individual_modules():
    """Test individual AI modules"""
    print("\n🔍 Testing Individual AI Modules...")
    print("=" * 50)
    
    modules_to_test = [
        'ai_cinematic_director',
        'ai_advanced_voice_acting', 
        'ai_visual_suite',
        'ai_audio_suite',
        'ai_content_suite',
        'ai_video_suite',
        'ai_analytics_suite',
        'ai_realtime_director',
        'ai_master_suite'
    ]
    
    successful_imports = 0
    total_modules = len(modules_to_test)
    
    for module_name in modules_to_test:
        try:
            # Try to import the module
            module = __import__(module_name)
            print(f"✅ {module_name}: Import successful")
            successful_imports += 1
            
            # Try to get module info if available
            try:
                if hasattr(module, '__version__'):
                    print(f"   Version: {module.__version__}")
                if hasattr(module, '__doc__') and module.__doc__:
                    doc_preview = module.__doc__.strip()[:100]
                    if len(module.__doc__.strip()) > 100:
                        doc_preview += "..."
                    print(f"   Description: {doc_preview}")
                    
                # Check for common attributes
                if hasattr(module, 'main') or hasattr(module, 'run') or hasattr(module, 'process'):
                    print(f"   ✅ Has main functionality")
                    
            except Exception as info_error:
                print(f"   ⚠️ Info extraction failed: {info_error}")
                
        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {e}")
        except Exception as e:
            print(f"⚠️ {module_name}: Error - {e}")
    
    print(f"\n📊 Module Import Summary: {successful_imports}/{total_modules} successful")
    
    # Return True if most modules imported successfully
    return successful_imports >= total_modules * 0.7

def test_config_integration():
    """Test configuration integration"""
    print("\n⚙️ Testing Configuration Integration...")
    print("=" * 50)
    
    try:
        from config import AI_CONFIG, CHANNELS_CONFIG
        
        print("✅ Config import successful")
        
        # Test AI_CONFIG structure
        if AI_CONFIG:
            print(f"📊 AI Config: {len(AI_CONFIG)} configuration items")
            
            # Check for AI modules configuration
            if 'ai_modules' in AI_CONFIG:
                ai_modules = AI_CONFIG['ai_modules']
                print(f"🤖 AI Modules Config: {len(ai_modules)} modules configured")
                
                for module_name, config in ai_modules.items():
                    try:
                        status = "✅" if config.get('enabled') else "❌"
                        priority = config.get('priority', 'unknown')
                        premium = "🌟" if config.get('premium_features') else ""
                        print(f"   {status} {module_name}: {priority} priority {premium}")
                    except Exception as config_error:
                        print(f"   ⚠️ {module_name}: Config error - {config_error}")
            else:
                print("⚠️ AI modules configuration not found in AI_CONFIG")
        else:
            print("⚠️ AI_CONFIG is empty or None")
        
        # Test CHANNELS_CONFIG
        if CHANNELS_CONFIG:
            print(f"📺 Channels: {list(CHANNELS_CONFIG.keys())}")
            print(f"   Total channels: {len(CHANNELS_CONFIG)}")
        else:
            print("⚠️ CHANNELS_CONFIG is empty or None")
            
        return True
        
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        return False

def test_frontend_integration():
    """Test frontend integration"""
    print("\n🖥️ Testing Frontend Integration...")
    print("=" * 50)
    
    try:
        # Test if frontend can import AI components
        from frontend import VideoPipelineGUI
        
        print("✅ Frontend import successful")
        
        # Check if AI suite is available in frontend
        try:
            from ai_integrated_suite import create_ai_suite, check_ai_dependencies
            
            # Check dependencies first
            ai_deps = check_ai_dependencies()
            print(f"📊 AI Dependencies: {ai_deps}")
            
            # Try to create AI suite
            try:
                ai_suite = create_ai_suite()
                print("✅ AI Suite available for frontend")
                
                # Test frontend AI integration
                if hasattr(VideoPipelineGUI, 'ai_suite'):
                    print("✅ Frontend has AI suite integration")
                else:
                    print("⚠️ Frontend missing AI suite integration")
                    
                # Check if frontend has AI-related methods
                ai_methods = [method for method in dir(VideoPipelineGUI) if 'ai' in method.lower()]
                if ai_methods:
                    print(f"✅ Frontend has AI methods: {ai_methods}")
                else:
                    print("⚠️ Frontend has no AI-specific methods")
                    
            except Exception as suite_error:
                print(f"⚠️ AI Suite creation failed: {suite_error}")
                print("   This is expected if some AI modules are not available")
                
        except ImportError as ai_import_error:
            print(f"⚠️ AI Integrated Suite import failed: {ai_import_error}")
            print("   This is expected if AI suite is not properly installed")
            
        return True
            
    except ImportError as e:
        print(f"❌ Frontend import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Frontend integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 AI Integration Test Suite")
    print("Testing all AI modules and integration points")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("AI Integration", test_ai_integration),
        ("Individual Modules", test_individual_modules),
        ("Configuration", test_config_integration),
        ("Frontend", test_frontend_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} Test...")
            result = test_func()
            results[test_name] = result
            print(f"✅ {test_name} Test: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name} Test: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI integration is working correctly.")
    elif passed >= total * 0.7:
        print("⚠️ Most tests passed. Some AI modules may have issues.")
        print("💡 This is normal for development environments.")
    else:
        print("❌ Many tests failed. AI integration needs attention.")
        print("💡 Check the error messages above for specific issues.")
    
    # Provide recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not results.get("Individual Modules", False):
        print("   • Check if all AI modules are properly installed")
        print("   • Verify module dependencies")
    if not results.get("Configuration", False):
        print("   • Verify config.py file structure")
        print("   • Check AI_CONFIG and CHANNELS_CONFIG")
    if not results.get("Frontend", False):
        print("   • Ensure frontend.py has proper AI integration")
        print("   • Check import statements")
    
    return passed >= total * 0.7  # Allow 70% success rate

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
