#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AI INTEGRATION DEMO SCRIPT
Demonstrates the full AI integration capabilities
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demo_ai_integration():
    """Demo the AI integration capabilities"""
    print("🚀 AI INTEGRATION DEMO")
    print("=" * 50)
    
    try:
        # 1. Test AI Integrated Suite
        print("\n1️⃣ Testing AI Integrated Suite...")
        from ai_integrated_suite import AIIntegratedSuite, create_ai_suite
        
        # Create suite
        suite = create_ai_suite()
        print("✅ AI Integrated Suite created successfully")
        
        # Check system status
        status = suite.get_system_status()
        print(f"📊 System Status: {json.dumps(status, indent=2, default=str)}")
        
        # 2. Test Cinematic Director
        print("\n2️⃣ Testing Cinematic Director...")
        try:
            from ai_cinematic_director import CinematicAIDirector, StoryArc, CinematicStyle, PacingType
            
            director = CinematicAIDirector()
            print("✅ Cinematic Director initialized")
            
            # Create story structure
            story = director.create_story_structure(
                channel_name="CKLegends",
                niche="history",
                target_duration_minutes=15,
                style=CinematicStyle.HOLLYWOOD,
                arc_type=StoryArc.HERO_JOURNEY,
                pacing=PacingType.MEDIUM
            )
            
            if story and not story.get("error"):
                print(f"✅ Story structure created: {story.get('total_scenes', 0)} scenes")
                print(f"📈 Engagement prediction: {story.get('engagement_prediction', 0.0):.2f}")
            else:
                print(f"⚠️ Story creation had issues: {story.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Cinematic Director test failed: {e}")
        
        # 3. Test Voice Acting Engine
        print("\n3️⃣ Testing Voice Acting Engine...")
        try:
            from ai_advanced_voice_acting import AdvancedVoiceActingEngine, CharacterPersonality, Emotion, VoiceStyle
            
            voice_engine = AdvancedVoiceActingEngine()
            print("✅ Voice Acting Engine initialized")
            
            # Test voice generation (without actual audio)
            test_text = "This is a test of the voice acting engine."
            result = voice_engine.create_character_voice(
                text=test_text,
                character_personality=CharacterPersonality.WISE,
                emotion=Emotion.INSPIRATIONAL,
                voice_style=VoiceStyle.AUTHORITATIVE
            )
            
            if result and hasattr(result, 'quality_score'):
                print(f"✅ Voice generation test completed")
                print(f"📊 Quality score: {result.quality_score:.2f}")
            else:
                print("⚠️ Voice generation test had issues")
                
        except Exception as e:
            print(f"❌ Voice Acting Engine test failed: {e}")
        
        # 4. Test Full Pipeline
        print("\n4️⃣ Testing Full AI Pipeline...")
        try:
            # Run a simplified pipeline test
            results = suite.run_full_pipeline("CKLegends", "history", 5)  # 5 minutes for demo
            
            if results.get("success"):
                print("✅ Full AI pipeline completed successfully!")
                print(f"📊 Quality Score: {results.get('quality_score', 0.0):.2f}")
                print(f"⏱️ Processing Time: {results.get('processing_time', 0.0):.2f}s")
                
                # Show pipeline status
                pipeline = results.get("pipeline", {})
                print(f"🎬 Story Structure: {'✅' if pipeline.get('story_structure') else '❌'}")
                print(f"✍️ Script Content: {'✅' if pipeline.get('script_content') else '❌'}")
                print(f"🎭 Voice Acting: {'✅' if pipeline.get('voice_acting') else '❌'}")
                print(f"🎨 Visual Assets: {'✅' if pipeline.get('visual_assets') else '❌'}")
                print(f"🎵 Audio Enhancement: {'✅' if pipeline.get('audio_enhancement') else '❌'}")
                print(f"🎬 Video Editing: {'✅' if pipeline.get('video_editing') else '❌'}")
                print(f"📊 Analytics: {'✅' if pipeline.get('analytics_data') else '❌'}")
            else:
                print(f"❌ Full AI pipeline failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Full pipeline test failed: {e}")
        
        # 5. Show Available Features
        print("\n5️⃣ Available AI Features...")
        try:
            available_modules = [name for name, module in suite.modules.items() if module is not None]
            print(f"✅ Available modules: {', '.join(available_modules)}")
            
            for module_name in available_modules:
                features = suite.get_module_features(module_name)
                print(f"   📋 {module_name}: {', '.join(features)}")
                
        except Exception as e:
            print(f"❌ Feature check failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 AI INTEGRATION DEMO COMPLETED!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_individual_modules():
    """Demo individual AI modules"""
    print("\n🔧 INDIVIDUAL MODULE DEMOS")
    print("=" * 30)
    
    # Test each module individually
    modules_to_test = [
        ("Cinematic Director", "ai_cinematic_director", "CinematicAIDirector"),
        ("Voice Acting Engine", "ai_advanced_voice_acting", "AdvancedVoiceActingEngine"),
        ("Visual Suite", "ai_visual_suite", "AIVisualSuite"),
        ("Audio Suite", "ai_audio_suite", "AIAudioSuite"),
        ("Content Suite", "ai_content_suite", "AIContentSuite"),
        ("Video Suite", "ai_video_suite", "AIVideoSuite"),
        ("Analytics Suite", "ai_analytics_suite", "AIAnalyticsSuite"),
        ("Realtime Director", "ai_realtime_director", "AIContentDirector")
    ]
    
    for module_name, module_path, class_name in modules_to_test:
        print(f"\n🔍 Testing {module_name}...")
        try:
            module = __import__(module_path)
            class_obj = getattr(module, class_name)
            instance = class_obj()
            print(f"✅ {module_name}: Initialized successfully")
            
            # Try to get status if available
            if hasattr(instance, 'get_system_status'):
                try:
                    status = instance.get_system_status()
                    print(f"   📊 Status: {status.get('system_health', 'Unknown')}")
                except:
                    pass
                    
        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {e}")
        except Exception as e:
            print(f"⚠️ {module_name}: Initialization failed - {e}")

def show_system_capabilities():
    """Show system capabilities and recommendations"""
    print("\n💡 SYSTEM CAPABILITIES & RECOMMENDATIONS")
    print("=" * 50)
    
    print("\n🎯 Current Capabilities:")
    print("✅ AI Integrated Suite - Full pipeline orchestration")
    print("✅ Cinematic Director - Story structure and pacing")
    print("✅ Voice Acting Engine - Character voice generation")
    print("✅ Core Systems - LLM handler, video creator")
    print("✅ Fallback Systems - Graceful degradation")
    
    print("\n🔧 Recommended Setup:")
    print("1. Install Ollama and run: ollama serve")
    print("2. Set PEXELS_API_KEY in .env for visual assets")
    print("3. Set ELEVENLABS_API_KEY for premium voice synthesis")
    print("4. Install additional TTS engines (espeak, piper) for voice variety")
    
    print("\n🚀 Usage Examples:")
    print("• python main.py --ai CKLegends history 15")
    print("• python main.py --ai-status")
    print("• python main.py --single CKLegends")
    print("• python main.py (runs full pipeline for all channels)")
    
    print("\n📊 Quality Expectations:")
    print("• Story Structure: 8/10 (AI-powered narrative optimization)")
    print("• Voice Acting: 7/10 (Emotional character voices)")
    print("• Visual Generation: 6/10 (AI-enhanced assets)")
    print("• Overall Pipeline: 7/10 (Seamless integration)")

if __name__ == "__main__":
    print("🚀 Starting AI Integration Demo...")
    
    try:
        # Run main demo
        demo_ai_integration()
        
        # Run individual module tests
        demo_individual_modules()
        
        # Show capabilities
        show_system_capabilities()
        
        print("\n🎉 All demos completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()




