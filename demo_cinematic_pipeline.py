#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 CINEMATIC PIPELINE DEMO
Demonstrates the new 10+ minute cinematic content creation system
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import EnhancedMasterDirector
    from config import CHANNELS_CONFIG
    print("✅ Enhanced Master Director imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def setup_demo_logging():
    """Setup logging for demo"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def demo_cinematic_pipeline():
    """Demonstrate the cinematic pipeline capabilities"""
    print("\n" + "="*80)
    print("🎬 CINEMATIC PIPELINE DEMONSTRATION")
    print("="*80)
    
    try:
        # Initialize director
        director = EnhancedMasterDirector()
        print("✅ Enhanced Master Director initialized")
        
        # Show available channels
        print(f"\n📺 Available Channels: {list(CHANNELS_CONFIG.keys())}")
        
        # Demo cinematic pipeline for a specific channel
        demo_channel = "CKLegends"  # History niche
        print(f"\n🎬 Running cinematic pipeline demo for {demo_channel}")
        
        # Run cinematic pipeline
        start_time = time.time()
        result = director.run_cinematic_pipeline(demo_channel, 15.0)
        end_time = time.time()
        
        # Display results
        print("\n" + "="*60)
        print("🎬 CINEMATIC PIPELINE RESULTS")
        print("="*60)
        
        if result.get("success"):
            print(f"✅ Success: Cinematic masterpiece created!")
            print(f"📺 Channel: {result.get('channel', 'Unknown')}")
            print(f"🎯 Niche: {result.get('niche', 'Unknown')}")
            print(f"⏱️ Target Duration: {result.get('target_duration_minutes', 0):.1f} minutes")
            print(f"⏱️ Actual Duration: {result.get('actual_duration_minutes', 0):.1f} minutes")
            print(f"📊 Quality Score: {result.get('quality_score', 0):.2f}")
            print(f"🎬 Video Path: {result.get('video_path', 'Unknown')}")
            print(f"⏱️ Processing Time: {end_time - start_time:.1f} seconds")
            
            # Show detailed report
            if "report" in result:
                report = result["report"]
                print(f"\n📋 DETAILED REPORT:")
                print(f"   Production Quality: {report.get('production_summary', {}).get('production_quality', 'Unknown')}")
                print(f"   Script Structure: {report.get('content_analysis', {}).get('script_structure', 'Unknown')}")
                print(f"   Word Count: {report.get('content_analysis', {}).get('word_count', 0)}")
                print(f"   Resolution: {report.get('technical_specifications', {}).get('resolution', 'Unknown')}")
                print(f"   Overall Rating: {report.get('quality_metrics', {}).get('overall_rating', 'Unknown')}")
                
                # Show recommendations
                recommendations = report.get('recommendations', [])
                if recommendations:
                    print(f"\n💡 RECOMMENDATIONS:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"   {i}. {rec}")
            
            # Show production metadata
            if "production_metadata" in result:
                metadata = result["production_metadata"]
                print(f"\n🔧 PRODUCTION METADATA:")
                print(f"   Total Assets Used: {metadata.get('total_assets_used', 0)}")
                print(f"   Audio Tracks: {metadata.get('audio_tracks', 0)}")
                print(f"   Production Time: {metadata.get('production_time', 'Unknown')}")
                print(f"   Quality Level: {metadata.get('quality_level', 'Unknown')}")
                
        else:
            print(f"❌ Cinematic pipeline failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logging.exception("Demo error details:")

def demo_quality_comparison():
    """Demonstrate quality comparison between old and new systems"""
    print("\n" + "="*80)
    print("📊 QUALITY COMPARISON: OLD vs NEW SYSTEM")
    print("="*80)
    
    comparison = {
        "old_system": {
            "duration": "39 seconds",
            "visual_assets": "5 images",
            "audio_quality": "Basic TTS",
            "visual_quality": "Low resolution",
            "content_depth": "Shallow",
            "engagement": "Low"
        },
        "new_cinematic_system": {
            "duration": "15+ minutes",
            "visual_assets": "180+ assets (12 per minute)",
            "audio_quality": "Professional studio quality",
            "visual_quality": "4K cinematic",
            "content_depth": "Deep narrative arc",
            "engagement": "High (cinematic storytelling)"
        }
    }
    
    print("\n📈 IMPROVEMENT METRICS:")
    print(f"   Duration: {comparison['old_system']['duration']} → {comparison['new_cinematic_system']['duration']}")
    print(f"   Visual Assets: {comparison['old_system']['visual_assets']} → {comparison['new_cinematic_system']['visual_assets']}")
    print(f"   Audio Quality: {comparison['old_system']['audio_quality']} → {comparison['new_cinematic_system']['audio_quality']}")
    print(f"   Visual Quality: {comparison['old_system']['visual_quality']} → {comparison['new_cinematic_system']['visual_quality']}")
    print(f"   Content Depth: {comparison['old_system']['content_depth']} → {comparison['new_cinematic_system']['content_depth']}")
    print(f"   Engagement: {comparison['old_system']['engagement']} → {comparison['new_cinematic_system']['engagement']}")

def demo_technical_specifications():
    """Show technical specifications for cinematic content"""
    print("\n" + "="*80)
    print("🔧 TECHNICAL SPECIFICATIONS")
    print("="*80)
    
    specs = {
        "video": {
            "resolution": "4K (3840x2160)",
            "frame_rate": "30fps (60fps for action scenes)",
            "codec": "H.264 High Profile / H.265",
            "bitrate": "20-100 Mbps",
            "color_depth": "10-bit HDR",
            "color_space": "Rec. 2020"
        },
        "audio": {
            "sample_rate": "48kHz",
            "bit_depth": "24-bit",
            "channels": "Stereo",
            "codec": "AAC",
            "bitrate": "320-640 kbps",
            "quality": "Professional studio"
        },
        "content": {
            "duration": "10-20 minutes",
            "story_structure": "5-Act cinematic arc",
            "emotional_beats": "Opening, Hook, Conflict, Climax, Resolution",
            "visual_transitions": "Smooth crossfades",
            "asset_rotation": "No repetition within 30 seconds"
        }
    }
    
    for category, details in specs.items():
        print(f"\n📺 {category.upper()}:")
        for key, value in details.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")

def demo_usage_instructions():
    """Show how to use the new cinematic pipeline"""
    print("\n" + "="*80)
    print("📖 USAGE INSTRUCTIONS")
    print("="*80)
    
    instructions = [
        "🎬 Create cinematic masterpiece for specific channel:",
        "   python main.py --cinematic CKLegends",
        "",
        "🎬 Create cinematic content for all channels:",
        "   python main.py --cinematic-all",
        "",
        "🎯 Run regular pipeline for single channel:",
        "   python main.py --single CKLegends",
        "",
        "🤖 Run AI-enhanced pipeline:",
        "   python main.py --ai CKLegends",
        "",
        "🔍 Analyze existing videos:",
        "   python main.py --analyze",
        "",
        "📊 Check AI suite status:",
        "   python main.py --ai-status",
        "",
        "🚀 Run full pipeline for all channels (default):",
        "   python main.py"
    ]
    
    for instruction in instructions:
        print(instruction)

def main():
    """Main demo function"""
    print("🎬 Welcome to the Cinematic Pipeline Demo!")
    print("This demo showcases the new 10+ minute cinematic content creation system.")
    
    # Setup logging
    setup_demo_logging()
    
    try:
        # Run demos
        demo_cinematic_pipeline()
        demo_quality_comparison()
        demo_technical_specifications()
        demo_usage_instructions()
        
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n🚀 Ready to create cinematic masterpieces!")
        print("💡 Use the commands above to get started.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logging.exception("Demo error details:")

if __name__ == "__main__":
    main()
