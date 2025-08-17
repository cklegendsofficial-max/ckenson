#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST MEMORY FIXES - Test script for memory management fixes
Tests the implemented memory management solutions
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_manager():
    """Test Memory Manager functionality"""
    print("🧪 Testing Memory Manager...")
    
    try:
        from core_engine.memory_manager import create_memory_manager
        
        # Create memory manager
        manager = create_memory_manager({
            'gpu_memory_limit_gb': 2.0,
            'cpu_fallback_threshold': 1.0,
            'auto_memory_management': True
        })
        
        if manager:
            print("✅ Memory Manager created successfully")
            
            # Test memory status
            status = manager.get_memory_status()
            print(f"📊 Memory Status: GPU={status.gpu_available}, "
                  f"GPU Memory={status.total_gpu_memory_gb:.1f}GB, "
                  f"Free={status.free_gpu_memory_gb:.1f}GB")
            
            # Test CPU usage decision
            should_use_cpu = manager.should_use_cpu(1.0)
            print(f"💡 Should use CPU: {should_use_cpu}")
            
            # Test memory optimization
            optimization = manager.optimize_memory()
            print(f"⚡ Memory optimization: {optimization}")
            
            return True
        else:
            print("❌ Memory Manager creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Memory Manager test failed: {e}")
        return False

def test_ai_cinematic_director():
    """Test AI Cinematic Director memory fixes"""
    print("\n🎬 Testing AI Cinematic Director...")
    
    try:
        from ai_cinematic_director import CinematicAIDirector
        
        # Create director instance
        director = CinematicAIDirector()
        print("✅ Cinematic Director created successfully")
        
        # Test capabilities
        capabilities = director.get_capabilities()
        print(f"🎯 Capabilities: {capabilities}")
        
        # Test story creation
        story = director.create_story_structure("TestChannel", "motivation", 5.0)
        print(f"📖 Story created: {story.get('success', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cinematic Director test failed: {e}")
        return False

def test_ai_integrated_suite():
    """Test AI Integrated Suite memory fixes"""
    print("\n🚀 Testing AI Integrated Suite...")
    
    try:
        from ai_integrated_suite import create_ai_suite
        
        # Create AI suite
        suite = create_ai_suite({
            'memory_config': {
                'gpu_memory_limit_gb': 2.0,
                'cpu_fallback_threshold': 1.0,
                'sequential_initialization': True
            }
        })
        
        if suite:
            print("✅ AI Suite created successfully")
            
            # Test system status
            status = suite.get_system_status()
            print(f"📊 System Status: {status['overall_status']}")
            print(f"🔧 Available Modules: {status['system_health']['available_modules']}/{status['system_health']['total_modules']}")
            
            # Test memory status
            memory_status = suite.get_memory_status()
            print(f"🧠 Memory Status: {memory_status}")
            
            # Test CPU usage decision
            should_use_cpu = suite.should_use_cpu(1.0)
            print(f"💡 Should use CPU: {should_use_cpu}")
            
            # Test memory clearing
            clear_result = suite.clear_memory()
            print(f"🧹 Memory clearing: {clear_result}")
            
            return True
        else:
            print("❌ AI Suite creation failed")
            return False
            
    except Exception as e:
        print(f"❌ AI Suite test failed: {e}")
        return False

def test_cinematic_video_creation():
    """Test cinematic video creation functionality"""
    print("\n🎬 Testing Cinematic Video Creation...")
    
    try:
        from content_pipeline.advanced_video_creator import AdvancedVideoCreator
        
        # Create video creator
        creator = AdvancedVideoCreator()
        print("✅ Video Creator created successfully")
        
        # Check if cinematic video method exists
        if hasattr(creator, 'create_cinematic_video'):
            print("✅ create_cinematic_video method available")
            
            # Test method signature
            import inspect
            sig = inspect.signature(creator.create_cinematic_video)
            print(f"📝 Method signature: {sig}")
            
            return True
        else:
            print("❌ create_cinematic_video method not available")
            return False
            
    except Exception as e:
        print(f"❌ Cinematic video creation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 STARTING MEMORY FIXES TEST SUITE")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Memory Manager", test_memory_manager()))
    test_results.append(("AI Cinematic Director", test_ai_cinematic_director()))
    test_results.append(("AI Integrated Suite", test_ai_integrated_suite()))
    test_results.append(("Cinematic Video Creation", test_cinematic_video_creation()))
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Memory fixes are working correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)

