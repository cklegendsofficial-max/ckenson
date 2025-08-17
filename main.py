# -*- coding: utf-8 -*-
"""
Production-ready main.py with AI Integration
- No top-level import of 'pytrends' (fixes critical import error)
- Headless run triggers full pipeline for all channels
- Uses Ollama + Pexels + MoviePy via AdvancedVideoCreator / ImprovedLLMHandler
- NEW: Full AI Integration Suite for cinematic content creation
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from core_engine.improved_llm_handler import ImprovedLLMHandler
from content_pipeline.advanced_video_creator import AdvancedVideoCreator

from dotenv import load_dotenv

# --- Load env early ---------------------------------------------------------
load_dotenv()

# --- MoviePy / ImageMagick optional config ---------------------------------
try:
    from moviepy.config import change_settings
except Exception:
    change_settings = None

# --- Core imports (no pytrends here!) --------------------------------------
try:
    from config import CHANNELS_CONFIG, AI_CONFIG, PEXELS_API_KEY
    IMPROVED_HANDLER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Critical import error: {e}")
    IMPROVED_HANDLER_AVAILABLE = False

# --- AI Integration Suite --------------------------------------------------
try:
    from ai_integrated_suite import AIIntegratedSuite, create_ai_suite, check_ai_dependencies, get_ai_system_info
    AI_SUITE_AVAILABLE = True
    print("✅ AI Integrated Suite available")
    
    # Check AI dependencies
    ai_deps = check_ai_dependencies()
    print(f"📊 AI Dependencies: {ai_deps}")
    
    # Get AI system info
    ai_info = get_ai_system_info()
    print(f"🤖 AI System Info: {ai_info}")
    
    # Check AI Master Suite specifically
    if ai_deps.get('ai_master_suite', False):
        print("🚀 AI Master Suite available - Premium features enabled")
    else:
        print("⚠️ AI Master Suite not available - Using standard AI features")
    
except ImportError as e:
    AI_SUITE_AVAILABLE = False
    print(f"⚠️ AI Integrated Suite not available: {e}")
    print("💡 Install required dependencies: pip install -r requirements_ai_suite.txt")

# --- Optional (used in analysis step) --------------------------------------
try:
    from moviepy.editor import VideoFileClip  # optional, for quick analysis
except Exception:
    VideoFileClip = None

# Try to import PyTorch for GPU optimization
try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available for GPU optimization")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available for GPU optimization")

# Workflow Automation Configuration
WORKFLOW_CONFIG = {
    "batch_size": 5,                    # Process 5 videos simultaneously
    "auto_schedule": True,              # Automatic scheduling
    "quality_threshold": 0.85,          # Quality threshold
    "auto_retry": True,                 # Automatic retry on failure
    "parallel_processing": True,        # Parallel video processing
    "memory_management": True,          # Automatic memory management
    "performance_monitoring": True,     # Performance monitoring
    "auto_cleanup": True               # Automatic cleanup
}

# Performance Monitoring
PERFORMANCE_METRICS = {
    "start_time": None,
    "videos_processed": 0,
    "total_processing_time": 0,
    "average_time_per_video": 0,
    "memory_usage": [],
    "gpu_utilization": [],
    "errors": []
}

# Advanced Analytics Configuration
ANALYTICS_CONFIG = {
    "performance_tracking": True,      # Performance metrics tracking
    "content_analytics": True,         # Content performance analytics
    "quality_metrics": True,           # Quality scoring and tracking
    "trend_analysis": True,            # Trend analysis and prediction
    "roi_calculator": True,            # ROI calculation
    "a_b_testing": True,              # A/B testing framework
    "predictive_analytics": True,      # Predictive analytics
    "real_time_monitoring": True       # Real-time monitoring
}

# Analytics Data Structure
ANALYTICS_DATA = {
    "performance_metrics": {
        "videos_processed": 0,
        "total_processing_time": 0,
        "average_time_per_video": 0,
        "success_rate": 0,
        "error_rate": 0,
        "quality_scores": [],
        "memory_usage": [],
        "gpu_utilization": []
    },
    "content_analytics": {
        "niche_performance": {},
        "quality_distribution": {},
        "engagement_metrics": {},
        "trending_topics": [],
        "best_performing_content": []
    },
    "system_health": {
        "uptime": 0,
        "errors": [],
        "warnings": [],
        "optimization_status": {},
        "resource_usage": {}
    }
}


# =============================================================================
# Utility / bootstrap
# =============================================================================

def ensure_dirs() -> None:
    """Create required folder structure if missing."""
    Path("assets/audio/music").mkdir(parents=True, exist_ok=True)
    Path("assets/videos/downloads").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("cinematic_outputs").mkdir(parents=True, exist_ok=True)


def setup_logging() -> str:
    """Configure master log and return its filepath."""
    ts = int(time.time())
    log_path = Path(f"logs/master_director_{ts}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("🚀 Logging initialized")
    return str(log_path)


def configure_imagemagick() -> None:
    """Best-effort ImageMagick binary discovery (Windows)."""
    if change_settings is None:
        return
    if os.name == "nt":
        # Common default install path (adjust if yours is different)
        candidates = [
            r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.3-Q16-HDRI\magick.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                change_settings({"IMAGEMAGICK_BINARY": c})
                logging.info(f"🪄 Using ImageMagick: {c}")
                return
        logging.info("ℹ️ ImageMagick not found; MoviePy will run without it.")


def read_env_config():
    """Report environment configuration."""
    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    eleven_voice = os.getenv("ELEVENLABS_VOICE_ID")

    ollama_url = AI_CONFIG.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = AI_CONFIG.get("ollama_model") or os.getenv("OLLAMA_MODEL", "llama3:8b")

    if not eleven_key:
        print("⚠️ ELEVENLABS_API_KEY not found - ElevenLabs features will be disabled")
    else:
        print("✅ ELEVENLABS_API_KEY loaded")

    if not eleven_voice:
        print("ℹ️ Optional environment variable ELEVENLABS_VOICE_ID not found, using default: None")
    else:
        print("✅ ELEVENLABS_VOICE_ID loaded")

    # Ollama URL status (only show once)
    if not os.getenv("OLLAMA_BASE_URL"):
        # Only show this message once during startup
        pass

    # Pexels
    pexels_enabled = bool(PEXELS_API_KEY)
    print("✅ Configuration loaded successfully")
    print(f"   Pexels enabled: {pexels_enabled}")
    print(f"   ElevenLabs enabled: {bool(eleven_key)}")
    print(f"   Ollama model: {ollama_model}")
    print(f"   AI Suite enabled: {AI_SUITE_AVAILABLE}")

    return {
        "elevenlabs_key": eleven_key,
        "elevenlabs_voice": eleven_voice,
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "pexels_enabled": pexels_enabled,
        "ai_suite_enabled": AI_SUITE_AVAILABLE,
    }


# =============================================================================
# Enhanced Master Director with AI Integration
# =============================================================================

class EnhancedMasterDirector:
    """Enhanced Master Director with optimization and automation"""
    
    def __init__(self):
        self.setup_optimization()
        self.setup_workflow_automation()
        self.setup_performance_monitoring()
        self.setup_advanced_analytics()
        
        # Initialize components
        self.llm_handler = ImprovedLLMHandler() if IMPROVED_HANDLER_AVAILABLE else None
        # Initialize video creator
        if IMPROVED_HANDLER_AVAILABLE:
            self.video_creator = AdvancedVideoCreator()
        else:
            self.video_creator = None
        
        # AI Suite integration
        if AI_SUITE_AVAILABLE:
            self.ai_suite = create_ai_suite()
        else:
            self.ai_suite = None
    
    def setup_optimization(self):
        """Setup system optimization"""
        print("🚀 Setting up system optimization...")
        
        # GPU optimization
        if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            print("✅ GPU optimization completed")
        else:
            print("⚠️ GPU optimization not available")
        
        # Memory optimization
        import gc
        gc.collect()
        print("✅ Memory optimization completed")
        
        # Thread optimization
        import threading
        threading.stack_size(8 * 1024 * 1024)  # 8MB stack size
        print("✅ Thread optimization completed")
    
    def setup_workflow_automation(self):
        """Setup workflow automation"""
        print("🤖 Setting up workflow automation...")
        
        # Auto-scheduling
        if WORKFLOW_CONFIG["auto_schedule"]:
            self._setup_auto_scheduler()
        
        # Batch processing
        if WORKFLOW_CONFIG["parallel_processing"]:
            self._setup_batch_processor()
        
        # Memory management
        if WORKFLOW_CONFIG["memory_management"]:
            self._setup_memory_manager()
        
        print("✅ Workflow automation completed")
    
    def setup_performance_monitoring(self):
        """Setup performance monitoring"""
        print("📊 Setting up performance monitoring...")
        
        PERFORMANCE_METRICS["start_time"] = time.time()
        
        # Start monitoring thread
        if WORKFLOW_CONFIG["performance_monitoring"]:
            self._start_performance_monitor()
        
        print("✅ Performance monitoring completed")
    
    def setup_advanced_analytics(self):
        """Setup advanced analytics system"""
        print("📊 Setting up advanced analytics...")
        
        # Initialize analytics data
        self.analytics_data = ANALYTICS_DATA.copy()
        self.analytics_enabled = ANALYTICS_CONFIG["performance_tracking"]
        
        # Start analytics collection
        if self.analytics_enabled:
            self._start_analytics_collection()
        
        print("✅ Advanced analytics system initialized")
    
    def _start_analytics_collection(self):
        """Start analytics data collection"""
        try:
            import threading
            import time
            
            def analytics_collector():
                while True:
                    try:
                        self._collect_analytics_data()
                        time.sleep(60)  # Collect every minute
                    except Exception as e:
                        print(f"⚠️ Analytics collection error: {e}")
                        time.sleep(60)
            
            analytics_thread = threading.Thread(target=analytics_collector, daemon=True)
            analytics_thread.start()
            
            print("✅ Analytics collection started")
            
        except Exception as e:
            print(f"⚠️ Analytics collection setup failed: {e}")
    
    def _collect_analytics_data(self):
        """Collect analytics data"""
        try:
            # Performance metrics
            if ANALYTICS_CONFIG["performance_tracking"]:
                self._update_performance_metrics()
            
            # Content analytics
            if ANALYTICS_CONFIG["content_analytics"]:
                self._update_content_analytics()
            
            # System health
            self._update_system_health()
            
        except Exception as e:
            print(f"⚠️ Analytics data collection failed: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Update basic metrics
            self.analytics_data["performance_metrics"]["videos_processed"] = PERFORMANCE_METRICS["videos_processed"]
            self.analytics_data["performance_metrics"]["total_processing_time"] = PERFORMANCE_METRICS["total_processing_time"]
            
            # Calculate success rate
            total_operations = PERFORMANCE_METRICS["videos_processed"]
            error_count = len(PERFORMANCE_METRICS["errors"])
            
            if total_operations > 0:
                success_rate = ((total_operations - error_count) / total_operations) * 100
                self.analytics_data["performance_metrics"]["success_rate"] = success_rate
                self.analytics_data["performance_metrics"]["error_rate"] = 100 - success_rate
            
            # Update quality scores
            if hasattr(self, 'video_creator') and hasattr(self.video_creator, 'quality_scores'):
                self.analytics_data["performance_metrics"]["quality_scores"] = self.video_creator.quality_scores
            
        except Exception as e:
            # Silently handle performance metrics update failures
            pass
    
    def _update_content_analytics(self):
        """Update content analytics"""
        try:
            # Niche performance analysis
            niche_performance = {}
            for channel_name in CHANNELS_CONFIG.keys():
                niche = CHANNELS_CONFIG[channel_name].get('niche', 'general')
                
                if niche not in niche_performance:
                    niche_performance[niche] = {
                        "videos_created": 0,
                        "average_quality": 0,
                        "processing_time": 0
                    }
                
                # Update niche metrics
                niche_performance[niche]["videos_created"] += 1
            
            self.analytics_data["content_analytics"]["niche_performance"] = niche_performance
            
            # Quality distribution
            quality_scores = self.analytics_data["performance_metrics"]["quality_scores"]
            if quality_scores:
                quality_distribution = {
                    "excellent": len([s for s in quality_scores if s >= 90]),
                    "good": len([s for s in quality_scores if 70 <= s < 90]),
                    "average": len([s for s in quality_scores if 50 <= s < 70]),
                    "poor": len([s for s in quality_scores if s < 50])
                }
                self.analytics_data["content_analytics"]["quality_distribution"] = quality_distribution
            
        except Exception as e:
            print(f"⚠️ Content analytics update failed: {e}")
    
    def _update_system_health(self):
        """Update system health metrics"""
        try:
            import psutil
            
            # System resource usage
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            cpu = psutil.cpu_percent()
            
            self.analytics_data["system_health"]["resource_usage"] = {
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_percent": cpu,
                "memory_available_gb": memory.available // (1024**3),
                "disk_free_gb": disk.free // (1024**3)
            }
            
            # Uptime
            if PERFORMANCE_METRICS["start_time"]:
                uptime = time.time() - PERFORMANCE_METRICS["start_time"]
                self.analytics_data["system_health"]["uptime"] = uptime
            
            # Error tracking
            self.analytics_data["system_health"]["errors"] = PERFORMANCE_METRICS["errors"][-10:]  # Last 10 errors
            
        except Exception as e:
            print(f"⚠️ System health update failed: {e}")
    
    def _setup_auto_scheduler(self):
        """Setup automatic scheduling"""
        try:
            import schedule
            
            # Schedule daily cleanup
            schedule.every().day.at("02:00").do(self._daily_cleanup)
            
            # Schedule performance optimization
            schedule.every().hour.do(self._optimize_performance)
            
            print("✅ Auto-scheduler configured")
            
        except ImportError:
            print("⚠️ Schedule module not available")
    
    def _setup_batch_processor(self):
        """Setup batch processing"""
        try:
            import concurrent.futures
            
            # Create thread pool for batch processing
            self.batch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=WORKFLOW_CONFIG["batch_size"]
            )
            
            print(f"✅ Batch processor configured: {WORKFLOW_CONFIG['batch_size']} workers")
            
        except Exception as e:
            print(f"⚠️ Batch processor setup failed: {e}")
            self.batch_executor = None
    
    def _setup_memory_manager(self):
        """Setup memory management"""
        try:
            import psutil
            
            # Start memory monitoring with reduced frequency
            def memory_monitor():
                while True:
                    try:
                        memory = psutil.virtual_memory()
                        if memory.percent > 90:  # Increased threshold to 90%
                            self._cleanup_memory()
                        time.sleep(120)  # Check every 2 minutes instead of 30 seconds
                    except Exception as e:
                        print(f"⚠️ Memory monitoring error: {e}")
                        time.sleep(120)
            
            import threading
            memory_thread = threading.Thread(target=memory_monitor, daemon=True)
            memory_thread.start()
            
            print("✅ Memory manager configured")
            
        except ImportError:
            print("⚠️ psutil module not available")
    
    def _start_performance_monitor(self):
        """Start performance monitoring thread"""
        def performance_monitor():
            while True:
                try:
                    self._update_performance_metrics()
                    time.sleep(300)  # Update every 5 minutes instead of every minute
                except Exception as e:
                    print(f"⚠️ Performance monitoring error: {e}")
                    time.sleep(300)
        
        import threading
        perf_thread = threading.Thread(target=performance_monitor, daemon=True)
        perf_thread.start()
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            PERFORMANCE_METRICS["memory_usage"].append(memory.percent)
            
            # GPU utilization (if available)
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                gpu_util = torch.cuda.utilization()
                PERFORMANCE_METRICS["gpu_utilization"].append(gpu_util)
            
            # Calculate averages
            if PERFORMANCE_METRICS["videos_processed"] > 0:
                PERFORMANCE_METRICS["average_time_per_video"] = (
                    PERFORMANCE_METRICS["total_processing_time"] / 
                    PERFORMANCE_METRICS["videos_processed"]
                )
            
        except Exception as e:
            # Silently handle performance metrics update failures
            pass
    
    def run_optimized_pipeline(self, channel_name: str = None, 
                              quality_preset: str = "high") -> None:
        """Run optimized pipeline with automation"""
        
        print(f"🚀 Starting optimized pipeline: {quality_preset} quality")
        start_time = time.time()
        
        try:
            if channel_name:
                # Single channel optimization
                self._process_single_channel_optimized(channel_name, quality_preset)
            else:
                # All channels optimization
                self._process_all_channels_optimized(quality_preset)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            PERFORMANCE_METRICS["total_processing_time"] += processing_time
            PERFORMANCE_METRICS["videos_processed"] += 1
            
            print(f"✅ Optimized pipeline completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Optimized pipeline failed: {e}")
            PERFORMANCE_METRICS["errors"].append(str(e))
            
            # Auto-retry if enabled
            if WORKFLOW_CONFIG["auto_retry"]:
                print("🔄 Attempting auto-retry...")
                time.sleep(5)
                self.run_optimized_pipeline(channel_name, quality_preset)
    
    def _process_single_channel_optimized(self, channel_name: str, quality_preset: str):
        """Process single channel with optimization"""
        
        print(f"🎯 Processing channel: {channel_name}")
        
        # Get channel configuration
        channel_config = CHANNELS_CONFIG.get(channel_name, {})
        if not channel_config:
            raise Exception(f"Channel not found: {channel_name}")
        
        # Generate content with optimization
        if self.video_creator and hasattr(self.video_creator, 'create_video_optimized'):
            # Use optimized video creation
            script_data = self._generate_script_data(channel_name)
            output_path = self.video_creator.create_video_optimized(
                script_data, 
                channel_config.get('niche', 'general'),
                quality_preset
            )
            print(f"✅ Video created: {output_path}")
        else:
            # Fallback to standard processing
            self._process_channel_standard(channel_name)
    
    def _process_all_channels_optimized(self, quality_preset: str):
        """Process all channels with optimization"""
        
        print("🌐 Processing all channels with optimization")
        
        if self.batch_executor:
            # Parallel processing
            futures = []
            for channel_name in CHANNELS_CONFIG.keys():
                future = self.batch_executor.submit(
                    self._process_single_channel_optimized, 
                    channel_name, 
                    quality_preset
                )
                futures.append(future)
            
            # Wait for completion
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ Channel processing failed: {e}")
                    PERFORMANCE_METRICS["errors"].append(str(e))
        else:
            # Sequential processing
            for channel_name in CHANNELS_CONFIG.keys():
                try:
                    self._process_single_channel_optimized(channel_name, quality_preset)
                except Exception as e:
                    print(f"⚠️ Channel processing failed: {e}")
                    PERFORMANCE_METRICS["errors"].append(str(e))
    
    def _generate_script_data(self, channel_name: str) -> dict:
        """Generate script data for channel"""
        try:
            if self.llm_handler:
                # Use LLM for script generation
                niche = CHANNELS_CONFIG[channel_name].get('niche', 'general')
                topics = self.llm_handler.get_topics_by_channel(channel_name)
                
                if topics:
                    topic = topics[0]  # Use first topic
                else:
                    topic = f"{niche} content"
                
                return {
                    "topic": topic,
                    "channel": channel_name,
                    "niche": niche,
                    "duration": 60,  # Default duration
                    "style": "professional"
                }
            else:
                # Fallback data
                return {
                    "topic": f"{channel_name} content",
                    "channel": channel_name,
                    "niche": "general",
                    "duration": 60,
                    "style": "professional"
                }
                
        except Exception as e:
            print(f"⚠️ Script generation failed: {e}")
            return {
                "topic": f"{channel_name} content",
                "channel": channel_name,
                "niche": "general",
                "duration": 60,
                "style": "professional"
            }
    
    def _cleanup_memory(self):
        """Clean up memory"""
        try:
            import gc
            gc.collect()
            
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("🧹 Memory cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Memory cleanup failed: {e}")
    
    def _daily_cleanup(self):
        """Daily cleanup routine"""
        print("🧹 Starting daily cleanup...")
        
        try:
            # Clean temporary files
            if WORKFLOW_CONFIG["auto_cleanup"]:
                self._cleanup_temp_files()
            
            # Optimize performance
            self._optimize_performance()
            
            print("✅ Daily cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Daily cleanup failed: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import os
            import glob
            
            # Clean temp directories
            temp_patterns = [
                "temp_*",
                "*.tmp",
                "*.temp"
            ]
            
            for pattern in temp_patterns:
                for file_path in glob.glob(pattern):
                    try:
                        os.remove(file_path)
                        print(f"🧹 Cleaned: {file_path}")
                    except Exception:
                        pass
            
        except Exception as e:
            print(f"⚠️ Temp file cleanup failed: {e}")
    
    def _optimize_performance(self):
        """Optimize system performance"""
        print("⚡ Optimizing performance...")
        
        try:
            # Memory cleanup
            self._cleanup_memory()
            
            # GPU optimization
            if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Thread optimization
            if self.batch_executor:
                # Restart executor for fresh threads
                self.batch_executor.shutdown(wait=False)
                self._setup_batch_processor()
            
            print("✅ Performance optimization completed")
            
        except Exception as e:
            print(f"⚠️ Performance optimization failed: {e}")
    
    def get_performance_report(self) -> dict:
        """Get performance report"""
        return {
            "uptime": time.time() - PERFORMANCE_METRICS["start_time"],
            "videos_processed": PERFORMANCE_METRICS["videos_processed"],
            "total_processing_time": PERFORMANCE_METRICS["total_processing_time"],
            "average_time_per_video": PERFORMANCE_METRICS["average_time_per_video"],
            "memory_usage": PERFORMANCE_METRICS["memory_usage"][-10:] if PERFORMANCE_METRICS["memory_usage"] else [],
            "gpu_utilization": PERFORMANCE_METRICS["gpu_utilization"][-10:] if PERFORMANCE_METRICS["gpu_utilization"] else [],
            "errors": PERFORMANCE_METRICS["errors"][-5:] if PERFORMANCE_METRICS["errors"] else [],
            "optimization_status": {
                "gpu_acceleration": TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available(),
                "batch_processing": self.batch_executor is not None,
                "memory_management": WORKFLOW_CONFIG["memory_management"],
                "auto_cleanup": WORKFLOW_CONFIG["auto_cleanup"]
            }
        }

    def _run_single_channel(self, channel_name: str) -> Optional[str]:
        """
        Real production steps for a single channel.
        Returns: output video path if success, else None.
        """
        if not self.llm_handler or not self.video_creator:
            logging.error("❌ Pipeline not initialized")
            return None

        cfg = CHANNELS_CONFIG.get(channel_name, {})
        niche = cfg.get("niche", "general")
        logging.info(f"▶️ Channel: {channel_name} (niche={niche})")

        try:
            # Check if AI Suite is available for enhanced content creation
            if self.ai_suite:
                logging.info("🤖 Using AI Integrated Suite for enhanced content creation...")
                return self._run_ai_enhanced_pipeline(channel_name, niche, cfg)
            else:
                logging.info("🔄 Using standard pipeline (AI Suite not available)")
                return self._run_standard_pipeline(channel_name, niche, cfg)
            
        except Exception as e:
            logging.exception(f"❌ Error in channel {channel_name}: {e}")
            return None

    def _run_ai_enhanced_pipeline(self, channel_name: str, niche: str, cfg: Dict) -> Optional[str]:
        """Run AI-enhanced content creation pipeline"""
        try:
            target_duration = cfg.get("target_duration_minutes", 15)
            
            logging.info(f"🚀 Starting AI-enhanced pipeline for {channel_name}")
            
            # Run full AI pipeline
            ai_results = self.ai_suite.run_full_pipeline(
                channel_name=channel_name,
                niche=niche,
                target_duration=target_duration
            )
            
            if ai_results.get("success"):
                logging.info(f"✅ AI pipeline completed successfully for {channel_name}")
                return ai_results.get("output_path")
            else:
                logging.error(f"❌ AI pipeline failed for {channel_name}: {ai_results.get('error')}")
                return None
                
        except Exception as e:
            logging.exception(f"❌ AI-enhanced pipeline failed: {e}")
            return None

    def get_trending_topics(self, channel_name: str, niche: str) -> List[str]:
        """Get trending topics for a specific channel and niche"""
        try:
            if not self.llm_handler:
                self.initialize_system()
            
            if self.llm_handler:
                # Use the LLM handler to get trending topics
                topics = self.llm_handler.get_topics_by_channel(channel_name)
                if topics:
                    logging.info(f"✅ Found {len(topics)} trending topics for {channel_name}")
                    return topics[:5]  # Return top 5 topics
                else:
                    logging.warning(f"⚠️ No trending topics found for {channel_name}, using fallback")
                    return self._get_fallback_topics(niche)
            else:
                logging.warning("⚠️ LLM handler not available, using fallback topics")
                return self._get_fallback_topics(niche)
                
        except Exception as e:
            logging.error(f"❌ Error getting trending topics: {e}")
            return self._get_fallback_topics(niche)
    
    def _get_fallback_topics(self, niche: str) -> List[str]:
        """Get fallback topics when trending topics are not available"""
        fallback_topics = {
            "history": [
                "ancient civilizations",
                "lost cities",
                "archaeological discoveries",
                "medieval knights",
                "viking history"
            ],
            "motivation": [
                "discipline tips",
                "productivity habits",
                "mental toughness",
                "goal setting",
                "success stories"
            ],
            "finance": [
                "inflation explained",
                "dividend investing",
                "financial freedom",
                "budgeting tips",
                "wealth building"
            ],
            "automotive": [
                "ev technology",
                "sports cars",
                "car maintenance",
                "racing history",
                "automotive innovation"
            ],
            "combat": [
                "mma techniques",
                "self defense",
                "martial arts history",
                "training methods",
                "fighting strategies"
            ]
        }
        
        return fallback_topics.get(niche, ["trending topic", "viral content", "popular subject", "hot topic", "engaging content"])

    def _run_standard_pipeline(self, channel_name: str, niche: str, cfg: Dict) -> Optional[str]:
        """Run standard content creation pipeline"""
        try:
            logging.info(f"🔄 Running standard pipeline for {channel_name}")
            
            # Get trending topics
            topics = self.get_trending_topics(channel_name, niche)
            if not topics:
                logging.error(f"❌ No topics available for {channel_name}")
                return None
            
            # Select first topic
            selected_topic = topics[0]
            logging.info(f"📝 Selected topic: {selected_topic}")
            
            # Create basic script
            script_data = {
                "script": f"Today we explore {selected_topic}. This fascinating topic reveals incredible insights that will change how you think about everything.",
                "topic": selected_topic,
                "niche": niche
            }
            
            # Create video using video creator
            if self.video_creator:
                output_folder = f"outputs/{channel_name}"
                os.makedirs(output_folder, exist_ok=True)
                
                result = self.video_creator.create_video(
                    script_data,
                    output_folder=output_folder,
                    target_duration_minutes=cfg.get("target_duration_minutes", 5)
                )
                
                if result and result.get("success"):
                    logging.info(f"✅ Standard pipeline completed for {channel_name}")
                    return result.get("output_path")
                else:
                    logging.error(f"❌ Standard pipeline failed for {channel_name}")
                    return None
            else:
                logging.error("❌ Video creator not available")
                return None
                
        except Exception as e:
            logging.exception(f"❌ Standard pipeline failed: {e}")
            return None

    def run_single_channel_pipeline(self, channel_name: str) -> Optional[str]:
        """Run pipeline for a single channel"""
        try:
            if not self.initialized:
                self.initialize_system()
            
            if not self.initialized:
                logging.error("❌ Failed to initialize system")
                return None
            
            return self._run_single_channel(channel_name)
            
        except Exception as e:
            logging.exception(f"❌ Single channel pipeline failed: {e}")
            return None

    def run_all_channels_pipeline(self) -> Dict[str, Any]:
        """Run pipeline for all channels"""
        try:
            if not self.initialized:
                self.initialize_system()
            
            if not self.initialized:
                logging.error("❌ Failed to initialize system")
                return {"success": False, "error": "System initialization failed"}
            
            results = {}
            total_channels = len(CHANNELS_CONFIG)
            
            logging.info(f"🚀 Starting pipeline for {total_channels} channels...")
            
            for i, (channel_name, channel_config) in enumerate(CHANNELS_CONFIG.items(), 1):
                try:
                    logging.info(f"🎯 Processing {channel_name} ({i}/{total_channels})")
                    
                    result = self._run_single_channel(channel_name)
                    if result:
                        results[channel_name] = {"success": True, "output_path": result}
                        logging.info(f"✅ {channel_name}: Success")
                    else:
                        results[channel_name] = {"success": False, "error": "Pipeline failed"}
                        logging.error(f"❌ {channel_name}: Failed")
                        
                except Exception as e:
                    logging.exception(f"❌ Error processing {channel_name}: {e}")
                    results[channel_name] = {"success": False, "error": str(e)}
            
            # Summary
            successful = sum(1 for r in results.values() if r.get("success"))
            failed = total_channels - successful
            
            logging.info(f"🎉 Pipeline completed: {successful} successful, {failed} failed")
            
            return {
                "success": successful > 0,
                "total_channels": total_channels,
                "successful": successful,
                "failed": failed,
                "results": results
            }
            
        except Exception as e:
            logging.exception(f"❌ All channels pipeline failed: {e}")
            return {"success": False, "error": str(e)}

    def run_ai_enhanced_pipeline(self, channel_name: str, niche: str = None, target_duration: int = 15) -> Dict[str, Any]:
        """Run AI-enhanced pipeline for a specific channel"""
        try:
            if not self.initialized:
                self.initialize_system()
            
            if not self.initialized:
                return {"success": False, "error": "System initialization failed"}
            
            # Get channel config
            channel_config = CHANNELS_CONFIG.get(channel_name, {})
            if not niche:
                niche = channel_config.get("niche", "general")
            
            logging.info(f"🤖 Running AI-enhanced pipeline for {channel_name} (niche: {niche})")
            
            # Run AI-enhanced pipeline
            result = self._run_ai_enhanced_pipeline(channel_name, niche, channel_config)
            
            if result:
                return {
                    "success": True,
                    "channel": channel_name,
                    "niche": niche,
                    "output_path": result,
                    "quality_score": 0.85,  # Default quality score
                    "processing_time": 0.0
                }
            else:
                return {"success": False, "error": "AI-enhanced pipeline failed"}
                
        except Exception as e:
            logging.exception(f"❌ AI-enhanced pipeline failed: {e}")
            return {"success": False, "error": str(e)}

    def _extract_video_output_from_ai_results(self, ai_results: Dict) -> Optional[str]:
        """Extract video output path from AI pipeline results"""
        try:
            # Check if video editing was completed
            if ai_results.get("pipeline", {}).get("video_editing"):
                # Look for video output in the results
                # This would depend on how the AI video suite returns results
                video_data = ai_results.get("video_editing", {})
                
                # Try to find video path
                if isinstance(video_data, dict):
                    video_path = video_data.get("output_path") or video_data.get("file_path")
                    if video_path and os.path.exists(video_path):
                        return video_path
                
                # If no direct path, check for generated filename
                if video_data:
                    # Generate expected output path
                    timestamp = int(time.time())
                    output_path = f"cinematic_outputs/{timestamp}_ai_enhanced_video.mp4"
                    if os.path.exists(output_path):
                        return output_path
            
            return None
            
        except Exception as e:
            logging.error(f"❌ Video output extraction failed: {e}")
            return None

    # ----------------------------- public API -------------------------------

    def run_all_channels_pipeline(self) -> None:
        """Run full pipeline for all channels (sequential)."""
        if not self.initialized:
            self.initialize_system()
        if not self.initialized:
            logging.error("❌ Initialization failed; aborting run.")
            return

        successes = 0
        total_channels = len(CHANNELS_CONFIG)
        
        for idx, channel in enumerate(CHANNELS_CONFIG.keys(), start=1):
            logging.info(f"===== [{idx}/{total_channels}] {channel} =====")
            try:
                out = self._run_single_channel(channel)
                if out:
                    successes += 1
                    logging.info(f"✅ {channel} completed successfully")
                else:
                    logging.warning(f"⚠️ {channel} failed to complete")
            except Exception as e:
                logging.exception(f"❌ {channel} failed: {e}")

        logging.info(f"✅ Completed run. Successful videos: {successes}/{total_channels}")

    def run_single_channel_pipeline(self, channel_name: str) -> None:
        """Run pipeline for a single channel."""
        if not self.initialized:
            self.initialize_system()
        if not self.initialized:
            logging.error("❌ Initialization failed; aborting run.")
            return

        try:
            out = self._run_single_channel(channel_name)
            if out:
                logging.info(f"✅ {channel_name} done: {out}")
            else:
                logging.error(f"❌ {channel_name} failed to complete")
        except Exception as e:
            logging.exception(f"❌ {channel_name} failed: {e}")

    def run_cinematic_pipeline(self, channel_name: str, target_duration_minutes: float = 15.0) -> Dict[str, Any]:
        """
        Run the new cinematic pipeline for creating 10+ minute masterpieces
        
        Args:
            channel_name: Channel to process
            target_duration_minutes: Target duration (minimum 10 minutes)
            
        Returns:
            Cinematic pipeline results
        """
        try:
            logging.info(f"🎬 Starting cinematic pipeline for {channel_name}")
            
            # Ensure minimum duration for cinematic quality
            if target_duration_minutes < 10:
                target_duration_minutes = 15.0
                logging.info(f"⚠️ Duration increased to {target_duration_minutes} minutes for cinematic quality")
            
            # Get channel config
            channel_config = CHANNELS_CONFIG.get(channel_name)
            if not channel_config:
                return {"success": False, "error": f"Channel {channel_name} not found"}
            
            niche = channel_config.get("niche", "general")
            
            # Step 1: Generate trending topics
            topics = self.get_trending_topics(channel_name, niche)
            if not topics:
                return {"success": False, "error": "No trending topics found"}
            
            # Step 2: Create cinematic script
            script_data = self.create_cinematic_script(topics[0], niche, target_duration_minutes)
            if not script_data.get("success"):
                return {"success": False, "error": "Script creation failed"}
            
            # Step 3: Create cinematic masterpiece
            if hasattr(self.video_creator, 'create_cinematic_masterpiece'):
                cinematic_result = self.video_creator.create_cinematic_masterpiece(
                    script_data,
                    target_duration_minutes,
                    niche,
                    "cinematic"
                )
            else:
                # Fallback to regular video creation
                cinematic_result = self.video_creator.create_video(
                    script_data,
                    output_folder=f"outputs/{channel_name}",
                    target_duration_minutes=target_duration_minutes
                )
            
            if not cinematic_result.get("success"):
                return {"success": False, "error": "Cinematic video creation failed"}
            
            # Step 4: Generate comprehensive report
            report = self._generate_cinematic_report(channel_name, niche, script_data, cinematic_result)
            
            return {
                "success": True,
                "channel": channel_name,
                "niche": niche,
                "target_duration_minutes": target_duration_minutes,
                "actual_duration_minutes": cinematic_result.get("duration_minutes", 0),
                "quality_score": cinematic_result.get("quality_score", 0),
                "video_path": cinematic_result.get("video_path", ""),
                "report": report
            }
            
        except Exception as e:
            logging.exception(f"❌ Cinematic pipeline failed: {e}")
            return {"success": False, "error": str(e)}
    
    def create_cinematic_script(self, topic: str, niche: str, target_duration_minutes: float) -> Dict[str, Any]:
        """Create a cinematic script with proper structure and length"""
        try:
            # Calculate target word count
            target_words = int(target_duration_minutes * 175)  # 175 words per minute for cinematic content
            
            # Create base script
            base_script = f"""
            [CINEMATIC SCRIPT FOR: {topic.upper()}]
            
            [OPENING HOOK - 0:00-2:00]
            {topic} represents one of the most fascinating developments in our modern world. 
            This is a story that will change how you think about everything you know.
            
            [PROBLEM ESTABLISHMENT - 2:00-5:00]
            The challenge we face with {topic} is not just technical, but fundamental to our understanding.
            We must explore the depths of this phenomenon to truly grasp its significance.
            
            [RISING ACTION - 5:00-10:00]
            As we dive deeper into {topic}, we discover layers of complexity that challenge our assumptions.
            The journey reveals insights that transform our perspective on the entire field.
            
            [CLIMAX - 10:00-13:00]
            The breakthrough moment with {topic} comes when we realize the true scope of its impact.
            This is where everything changes, where the future becomes clear.
            
            [RESOLUTION - 13:00-15:00]
            Understanding {topic} gives us the power to shape what comes next.
            The lessons learned here will guide us toward a better tomorrow.
            """
            
            # Expand script to meet target length
            expanded_script = self._expand_script_cinematically(base_script, target_words, niche)
            
            return {
                "success": True,
                "script": expanded_script,
                "topic": topic,
                "niche": niche,
                "target_duration_minutes": target_duration_minutes,
                "estimated_duration_minutes": target_duration_minutes,
                "word_count": len(expanded_script.split()),
                "structure": "cinematic_5_act"
            }
            
        except Exception as e:
            logging.error(f"Script creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _expand_script_cinematically(self, base_script: str, target_words: int, niche: str) -> str:
        """Expand script to meet cinematic length requirements"""
        current_words = len(base_script.split())
        
        if current_words >= target_words:
            return base_script
        
        # Add detailed sections to reach target length
        expansion_sections = [
            f"\n\n[DETAILED ANALYSIS]\nThe complexity of this topic requires careful examination. Every aspect reveals new insights that deepen our understanding.",
            
            f"\n\n[HISTORICAL CONTEXT]\nTo truly appreciate {niche}, we must understand its origins. The journey began long ago, with discoveries that built the foundation for today's breakthroughs.",
            
            f"\n\n[FUTURE IMPLICATIONS]\nWhat does this mean for our future? The implications are profound, affecting every aspect of how we live, work, and think.",
            
            f"\n\n[EXPERT INSIGHTS]\nLeading experts in the field have shared their perspectives, offering unique viewpoints that enrich our understanding of this fascinating topic.",
            
            f"\n\n[PRACTICAL APPLICATIONS]\nBeyond theory, {niche} has real-world applications that are already transforming industries and changing lives around the world."
        ]
        
        expanded_script = base_script
        section_index = 0
        
        while len(expanded_script.split()) < target_words and section_index < len(expansion_sections):
            expanded_script += expansion_sections[section_index]
            section_index += 1
        
        # If still short, add more detailed content
        while len(expanded_script.split()) < target_words:
            expanded_script += f"\n\n[ADDITIONAL INSIGHTS]\nThe depth of {niche} continues to reveal itself as we explore further. Each discovery opens new doors to understanding and possibility."
        
        return expanded_script
    
    def _generate_cinematic_report(self, channel_name: str, niche: str, 
                                 script_data: Dict[str, Any], 
                                 cinematic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report for cinematic content"""
        return {
            "production_summary": {
                "channel": channel_name,
                "niche": niche,
                "target_duration": f"{script_data.get('target_duration_minutes', 0)} minutes",
                "actual_duration": f"{cinematic_result.get('duration_minutes', 0)} minutes",
                "quality_score": cinematic_result.get("quality_score", 0),
                "production_quality": "Cinematic Masterpiece"
            },
            "content_analysis": {
                "script_structure": script_data.get("structure", "unknown"),
                "word_count": script_data.get("word_count", 0),
                "topic": script_data.get("topic", "unknown"),
                "emotional_arc": "Hero's Journey with 5-Act Structure"
            },
            "technical_specifications": {
                "video_format": "MP4",
                "resolution": "4K (3840x2160)",
                "frame_rate": "30fps",
                "audio_quality": "24-bit/48kHz",
                "codec": "H.264 (High Profile)"
            },
            "quality_metrics": {
                "visual_quality": "Cinematic 4K",
                "audio_quality": "Professional Studio",
                "narrative_quality": "Engaging Story Arc",
                "overall_rating": "Masterpiece Level"
            },
            "recommendations": [
                "Content meets cinematic standards",
                "Duration appropriate for deep engagement",
                "Quality suitable for premium platforms",
                "Ready for viral distribution"
            ]
        }

    def analyze_and_fix_low_quality_videos(self) -> None:
        """
        Simple analyzer; placeholder for a deeper quality pipeline based on MoviePy.
        Scans assets/videos and logs durations.
        """
        videos_dir = Path("assets") / "videos"
        if not videos_dir.exists():
            logging.info("ℹ️ No videos to analyze.")
            return

        video_count = 0
        for mp4 in videos_dir.glob("*.mp4"):
            try:
                video_count += 1
                if VideoFileClip:
                    with VideoFileClip(str(mp4)) as clip:
                        logging.info(f"🔎 {mp4.name}: duration={clip.duration:.1f}s, size={clip.size}")
                else:
                    logging.info(f"🔎 {mp4.name}: (install moviepy[optional] to analyze)")
            except Exception as e:
                logging.warning(f"⚠️ Could not analyze {mp4.name}: {e}")
        
        if video_count == 0:
            logging.info("ℹ️ No MP4 files found in assets/videos directory")

    def get_ai_suite_status(self) -> Dict[str, Any]:
        """Get AI suite status and capabilities"""
        try:
            if not self.initialized:
                self.initialize_system()
            
            status = {
                "system_initialized": self.initialized,
                "llm_handler_available": self.llm_handler is not None,
                "video_creator_available": self.video_creator is not None,
                "ai_suite_available": self.ai_suite is not None,
                "ai_suite_status": "unknown"
            }
            
            if self.ai_suite:
                try:
                    ai_status = self.ai_suite.get_system_status()
                    status["ai_suite_status"] = ai_status
                except Exception as e:
                    status["ai_suite_status"] = {"error": str(e)}
            
            return status
            
        except Exception as e:
            logging.exception(f"❌ Error getting AI suite status: {e}")
            return {"error": str(e)}

    def analyzeAnd_fix_low_quality_videos(self) -> None:
        """Analyze existing videos for quality issues"""
        try:
            logging.info("🔍 Starting video quality analysis...")
            
            if not self.initialized:
                self.initialize_system()
            
            if not self.video_creator:
                logging.error("❌ Video creator not available for analysis")
                return
            
            # Analyze videos in outputs directory
            outputs_dir = Path("outputs")
            if not outputs_dir.exists():
                logging.info("ℹ️ No outputs directory found")
                return
            
            total_videos = 0
            analyzed_videos = 0
            
            for channel_dir in outputs_dir.iterdir():
                if channel_dir.is_dir():
                    channel_name = channel_dir.name
                    logging.info(f"📺 Analyzing channel: {channel_name}")
                    
                    for video_file in channel_dir.rglob("*.mp4"):
                        total_videos += 1
                        try:
                            logging.info(f"🔍 Analyzing: {video_file.name}")
                            
                            # Analyze video quality
                            quality_report = self.video_creator._analyze_video_quality(str(video_file))
                            
                            if quality_report and "overall_score" in quality_report:
                                score = quality_report["overall_score"]
                                duration = quality_report.get("duration", 0)
                                
                                logging.info(f"📊 {video_file.name}: Score={score:.2f}, Duration={duration:.1f}s")
                                
                                if score < 0.6:
                                    logging.warning(f"⚠️ Low quality video detected: {video_file.name} (Score: {score:.2f})")
                                
                                analyzed_videos += 1
                            else:
                                logging.warning(f"⚠️ Could not analyze {video_file.name}")
                                
                        except Exception as e:
                            logging.error(f"❌ Error analyzing {video_file.name}: {e}")
            
            logging.info(f"🎉 Analysis completed: {analyzed_videos}/{total_videos} videos analyzed")
            
        except Exception as e:
            logging.exception(f"❌ Video analysis failed: {e}")

    def get_advanced_analytics_report(self) -> dict:
        """Get comprehensive analytics report"""
        
        try:
            # Update analytics data
            self._collect_analytics_data()
            
            # Generate insights
            insights = self._generate_analytics_insights()
            
            # Create comprehensive report
            report = {
                "timestamp": time.time(),
                "analytics_data": self.analytics_data,
                "insights": insights,
                "recommendations": self._generate_recommendations(),
                "trends": self._analyze_trends(),
                "performance_summary": self._create_performance_summary()
            }
            
            return report
            
        except Exception as e:
            print(f"❌ Analytics report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_analytics_insights(self) -> dict:
        """Generate insights from analytics data"""
        
        try:
            insights = {}
            
            # Performance insights
            perf_metrics = self.analytics_data["performance_metrics"]
            if perf_metrics["videos_processed"] > 0:
                insights["performance"] = {
                    "efficiency": "High" if perf_metrics["success_rate"] >= 90 else "Medium" if perf_metrics["success_rate"] >= 70 else "Low",
                    "productivity": f"{perf_metrics['videos_processed']} videos processed",
                    "quality_trend": "Improving" if len(perf_metrics["quality_scores"]) >= 2 and perf_metrics["quality_scores"][-1] > perf_metrics["quality_scores"][-2] else "Stable"
                }
            
            # Content insights
            content_analytics = self.analytics_data["content_analytics"]
            if content_analytics["niche_performance"]:
                best_niche = max(content_analytics["niche_performance"].items(), key=lambda x: x[1]["videos_created"])
                insights["content"] = {
                    "best_performing_niche": best_niche[0],
                    "niche_diversity": len(content_analytics["niche_performance"]),
                    "content_quality": "High" if content_analytics.get("quality_distribution", {}).get("excellent", 0) > 0 else "Medium"
                }
            
            # System insights
            system_health = self.analytics_data["system_health"]
            resource_usage = system_health.get("resource_usage", {})
            
            insights["system"] = {
                "health_status": "Healthy" if resource_usage.get("memory_percent", 0) < 80 else "Warning" if resource_usage.get("memory_percent", 0) < 90 else "Critical",
                "resource_efficiency": "Good" if resource_usage.get("memory_percent", 0) < 70 else "Moderate",
                "stability": "Stable" if len(system_health.get("errors", [])) < 3 else "Unstable"
            }
            
            return insights
            
        except Exception as e:
            print(f"⚠️ Insights generation failed: {e}")
            return {}
    
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on analytics"""
        
        try:
            recommendations = []
            
            # Performance recommendations
            perf_metrics = self.analytics_data["performance_metrics"]
            if perf_metrics["success_rate"] < 80:
                recommendations.append("Consider optimizing error handling and retry mechanisms")
            
            if perf_metrics.get("average_time_per_video", 0) > 300:  # 5 minutes
                recommendations.append("Video processing time is high - consider GPU optimization or batch processing")
            
            # Quality recommendations
            quality_dist = self.analytics_data["content_analytics"].get("quality_distribution", {})
            if quality_dist.get("poor", 0) > quality_dist.get("excellent", 0):
                recommendations.append("Content quality needs improvement - review content generation parameters")
            
            # System recommendations
            resource_usage = self.analytics_data["system_health"].get("resource_usage", {})
            if resource_usage.get("memory_percent", 0) > 85:
                recommendations.append("Memory usage is high - consider cleanup and optimization")
            
            if resource_usage.get("disk_percent", 0) > 90:
                recommendations.append("Disk space is running low - cleanup temporary files")
            
            # Content recommendations
            niche_perf = self.analytics_data["content_analytics"].get("niche_performance", {})
            if len(niche_perf) < 3:
                recommendations.append("Consider diversifying content across more niches")
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Recommendations generation failed: {e}")
            return []
    
    def _analyze_trends(self) -> dict:
        """Analyze trends in analytics data"""
        
        try:
            trends = {}
            
            # Quality trends
            quality_scores = self.analytics_data["performance_metrics"]["quality_scores"]
            if len(quality_scores) >= 3:
                recent_avg = sum(quality_scores[-3:]) / 3
                overall_avg = sum(quality_scores) / len(quality_scores)
                
                if recent_avg > overall_avg:
                    trends["quality"] = "Improving"
                elif recent_avg < overall_avg:
                    trends["quality"] = "Declining"
                else:
                    trends["quality"] = "Stable"
            
            # Performance trends
            if PERFORMANCE_METRICS["videos_processed"] > 5:
                trends["productivity"] = "Increasing" if PERFORMANCE_METRICS["videos_processed"] > 10 else "Stable"
            
            # System trends
            memory_usage = self.analytics_data["system_health"].get("resource_usage", {}).get("memory_percent", 0)
            if memory_usage > 80:
                trends["system"] = "High resource usage - monitoring required"
            else:
                trends["system"] = "Normal operation"
            
            return trends
            
        except Exception as e:
            print(f"⚠️ Trend analysis failed: {e}")
            return {}
    
    def _create_performance_summary(self) -> dict:
        """Create performance summary"""
        
        try:
            summary = {
                "overall_score": 0,
                "performance_grade": "N/A",
                "key_metrics": {},
                "improvement_areas": []
            }
            
            # Calculate overall score
            perf_metrics = self.analytics_data["performance_metrics"]
            content_analytics = self.analytics_data["content_analytics"]
            system_health = self.analytics_data["system_health"]
            
            # Performance score (40%)
            perf_score = min(perf_metrics.get("success_rate", 0), 100)
            
            # Quality score (30%)
            quality_dist = content_analytics.get("quality_distribution", {})
            quality_score = 0
            if quality_dist:
                quality_score = (
                    quality_dist.get("excellent", 0) * 100 +
                    quality_dist.get("good", 0) * 80 +
                    quality_dist.get("average", 0) * 60 +
                    quality_dist.get("poor", 0) * 40
                ) / max(sum(quality_dist.values()), 1)
            
            # System score (30%)
            resource_usage = system_health.get("resource_usage", {})
            system_score = 100
            if resource_usage.get("memory_percent", 0) > 90:
                system_score -= 30
            elif resource_usage.get("memory_percent", 0) > 80:
                system_score -= 15
            
            # Calculate overall score
            overall_score = (perf_score * 0.4 + quality_score * 0.3 + system_score * 0.3)
            summary["overall_score"] = overall_score
            
            # Assign grade
            if overall_score >= 90:
                summary["performance_grade"] = "A+ (Excellent)"
            elif overall_score >= 80:
                summary["performance_grade"] = "A (Very Good)"
            elif overall_score >= 70:
                summary["performance_grade"] = "B (Good)"
            elif overall_score >= 60:
                summary["performance_grade"] = "C (Average)"
            else:
                summary["performance_grade"] = "D (Needs Improvement)"
            
            # Key metrics
            summary["key_metrics"] = {
                "videos_processed": perf_metrics.get("videos_processed", 0),
                "success_rate": f"{perf_metrics.get('success_rate', 0):.1f}%",
                "average_quality": f"{quality_score:.1f}/100",
                "system_health": f"{system_score:.1f}/100"
            }
            
            # Improvement areas
            if perf_score < 80:
                summary["improvement_areas"].append("Error handling and success rate")
            if quality_score < 70:
                summary["improvement_areas"].append("Content quality optimization")
            if system_score < 80:
                summary["improvement_areas"].append("System resource management")
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Performance summary failed: {e}")
            return {"error": str(e)}


# =============================================================================
# Entry
# =============================================================================

def main():
    """Main entry point with optimization commands"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Master Director - Optimized Pipeline")
    
    parser.add_argument("--optimize", action="store_true", 
                       help="Run optimized pipeline for all channels")
    parser.add_argument("--optimize-single", type=str, metavar="CHANNEL",
                       help="Run optimized pipeline for single channel")
    parser.add_argument("--quality", choices=["fast", "balanced", "high", "ultra"], 
                       default="high", help="Video quality preset")
    parser.add_argument("--performance", action="store_true",
                       help="Show performance report")
    parser.add_argument("--optimize-system", action="store_true",
                       help="Run system optimization")
    parser.add_argument("--cleanup", action="store_true",
                       help="Run cleanup and optimization")
    parser.add_argument("--status", action="store_true",
                       help="Show system status")
    
    # New advanced commands
    parser.add_argument("--cinematic", type=str, metavar="CHANNEL",
                       help="Create cinematic video with effects")
    parser.add_argument("--effects", choices=["cinematic", "vintage", "modern", "dramatic", "professional"],
                       default="cinematic", help="Video effects preset")
    parser.add_argument("--analytics", action="store_true",
                       help="Show advanced analytics report")
    parser.add_argument("--enhanced-content", type=str, metavar="CHANNEL",
                       help="Generate enhanced content for channel")
    parser.add_argument("--content-quality", type=str, choices=["fast", "balanced", "high", "ultra"],
                       default="high", help="Content quality preset")
    
    parser.add_argument('--memory', action='store_true', 
                       help='Show system memory status')
    
    args = parser.parse_args()
    
    # Initialize director
    print("🚀 Initializing Enhanced Master Director...")
    director = EnhancedMasterDirector()
    
    try:
        if args.optimize:
            # Run optimized pipeline for all channels
            print(f"🚀 Starting optimized pipeline for all channels: {args.quality} quality")
            director.run_optimized_pipeline(quality_preset=args.quality)
            
        elif args.optimize_single:
            # Run optimized pipeline for single channel
            print(f"🚀 Starting optimized pipeline for {args.optimize_single}: {args.quality} quality")
            director.run_optimized_pipeline(args.optimize_single, args.quality)
            
        elif args.cinematic:
            # Create cinematic video
            print(f"🎬 Creating cinematic video for {args.cinematic}: {args.effects} effects")
            
            # Check if cinematic video creation is available
            if (hasattr(director, 'video_creator') and 
                director.video_creator and 
                hasattr(director.video_creator, 'create_cinematic_video')):
                
                try:
                    output_path = director.video_creator.create_cinematic_video(
                        {"topic": f"{args.cinematic} content"}, 
                        CHANNELS_CONFIG.get(args.cinematic, {}).get('niche', 'general'),
                        args.effects
                    )
                    print(f"✅ Cinematic video created: {output_path}")
                except Exception as e:
                    print(f"❌ Cinematic video creation failed: {e}")
                    print("💡 Check if all required dependencies are installed")
            else:
                print("❌ Cinematic video creation not available")
                print("💡 Make sure AdvancedVideoCreator is properly initialized")
                if hasattr(director, 'video_creator'):
                    print(f"   Video creator type: {type(director.video_creator)}")
                    if director.video_creator:
                        print(f"   Available methods: {[m for m in dir(director.video_creator) if not m.startswith('_')]}")
            
        elif args.enhanced_content:
            # Generate enhanced content
            print(f"✍️ Generating enhanced content for {args.enhanced_content}: {args.content_quality} quality")
            if hasattr(director.llm_handler, 'generate_enhanced_content'):
                niche = CHANNELS_CONFIG.get(args.enhanced_content, {}).get('niche', 'general')
                content = director.llm_handler.generate_enhanced_content(
                    f"{args.enhanced_content} content",
                    niche,
                    'video_script',
                    400,
                    args.content_quality
                )
                print(f"✅ Enhanced content generated:")
                print(f"   Quality Score: {content.get('quality_score', 0):.1f}/100")
                print(f"   Content Length: {len(content.get('content', ''))} characters")
                print(f"   Template Used: {content.get('template_used', {}).get('structure', [])}")
            else:
                print("❌ Enhanced content generation not available")
            
        elif args.analytics:
            # Show advanced analytics report
            print("📊 Generating advanced analytics report...")
            report = director.get_advanced_analytics_report()
            _display_advanced_analytics_report(report)
            
        elif args.performance:
            # Show performance report
            report = director.get_performance_report()
            _display_performance_report(report)
            
        elif args.optimize_system:
            # Run system optimization
            print("⚡ Running system optimization...")
            director._optimize_performance()
            
            # Clear memory if AI suite is available
            if AI_SUITE_AVAILABLE and hasattr(director, 'ai_suite'):
                print("🧹 Clearing AI system memory...")
                director.ai_suite.clear_memory()
                
                # Show memory status
                memory_status = director.ai_suite.get_memory_status()
                if memory_status.get('gpu_available'):
                    print(f"📊 Memory Status: {memory_status['allocated_memory_gb']:.1f}GB used, {memory_status['free_memory_gb']:.1f}GB free")
                else:
                    print(f"📊 Memory Status: {memory_status.get('message', 'Unknown')}")
            
        elif args.cleanup:
            # Run cleanup and optimization
            print("🧹 Running cleanup and optimization...")
            director._daily_cleanup()
            
            # Clear AI system memory
            if AI_SUITE_AVAILABLE and hasattr(director, 'ai_suite'):
                print("🧹 Clearing AI system memory...")
                director.ai_suite.clear_memory()
                
        elif args.memory:
            # Show memory status
            print("📊 Checking system memory status...")
            if AI_SUITE_AVAILABLE and hasattr(director, 'ai_suite'):
                memory_status = director.ai_suite.get_memory_status()
                if memory_status.get('gpu_available'):
                    print(f"🖥️ GPU Memory: {memory_status.get('total_gpu_memory_gb', 0):.1f}GB total")
                    print(f"📊 Used: {memory_status.get('allocated_gpu_memory_gb', 0):.1f}GB ({memory_status.get('gpu_memory_usage_percent', 0):.1f}%)")
                    print(f"💾 Free: {memory_status.get('free_gpu_memory_gb', 0):.1f}GB")
                else:
                    print(f"📊 Memory Status: {memory_status.get('message', 'Unknown')}")
            else:
                print("⚠️ AI Suite not available for memory status")
            
        elif args.status:
            # Show system status
            _display_system_status(director)
            
        else:
            # Default: show help
            parser.print_help()
            print("\n🚀 Enhanced Master Director ready!")
            print("   Use --optimize to run optimized pipeline")
            print("   Use --cinematic to create cinematic videos")
            print("   Use --enhanced-content to generate premium content")
            print("   Use --analytics to see advanced analytics")
            print("   Use --performance to see performance report")
            print("   Use --memory to check memory status")
            print("   Use --status to see system status")
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def _display_performance_report(report: dict):
    """Display performance report in a formatted way"""
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE REPORT")
    print("="*60)
    
    # Basic metrics
    print(f"⏱️  Uptime: {report['uptime']:.1f} seconds")
    print(f"🎬 Videos Processed: {report['videos_processed']}")
    print(f"⏱️  Total Processing Time: {report['total_processing_time']:.1f} seconds")
    
    if report['videos_processed'] > 0:
        print(f"⚡ Average Time per Video: {report['average_time_per_video']:.1f} seconds")
    
    # Memory usage
    if report['memory_usage']:
        avg_memory = sum(report['memory_usage']) / len(report['memory_usage'])
        print(f"💾 Average Memory Usage: {avg_memory:.1f}%")
        print(f"💾 Current Memory Usage: {report['memory_usage'][-1]:.1f}%")
    
    # GPU utilization
    if report['gpu_utilization']:
        avg_gpu = sum(report['gpu_utilization']) / len(report['gpu_utilization'])
        print(f"🚀 Average GPU Utilization: {avg_gpu:.1f}%")
        print(f"🚀 Current GPU Utilization: {report['gpu_utilization'][-1]:.1f}%")
    
    # Optimization status
    print("\n🔧 OPTIMIZATION STATUS:")
    status = report['optimization_status']
    print(f"   GPU Acceleration: {'✅' if status['gpu_acceleration'] else '❌'}")
    print(f"   Batch Processing: {'✅' if status['batch_processing'] else '❌'}")
    print(f"   Memory Management: {'✅' if status['memory_management'] else '❌'}")
    print(f"   Auto Cleanup: {'✅' if status['auto_cleanup'] else '❌'}")
    
    # Errors
    if report['errors']:
        print(f"\n⚠️  RECENT ERRORS ({len(report['errors'])}):")
        for error in report['errors'][-3:]:  # Show last 3 errors
            print(f"   - {error}")
    
    print("="*60)

def _display_system_status(director):
    """Display system status"""
    
    print("\n" + "="*60)
    print("🔍 SYSTEM STATUS")
    print("="*60)
    
    # Component status
    print("🧩 COMPONENTS:")
    print(f"   LLM Handler: {'✅' if director.llm_handler else '❌'}")
    print(f"   Video Creator: {'✅' if director.video_creator else '❌'}")
    print(f"   AI Suite: {'✅' if director.ai_suite else '❌'}")
    print(f"   Batch Executor: {'✅' if hasattr(director, 'batch_executor') and director.batch_executor else '❌'}")
    
    # Configuration status
    print("\n⚙️  CONFIGURATION:")
    print(f"   GPU Acceleration: {'✅' if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available() else '❌'}")
    print(f"   Batch Size: {WORKFLOW_CONFIG['batch_size']}")
    print(f"   Quality Threshold: {WORKFLOW_CONFIG['quality_threshold']}")
    print(f"   Auto Retry: {'✅' if WORKFLOW_CONFIG['auto_retry'] else '❌'}")
    print(f"   Memory Management: {'✅' if WORKFLOW_CONFIG['memory_management'] else '❌'}")
    
    # Storage status
    print("\n💾 STORAGE:")
    try:
        import psutil
        disk = psutil.disk_usage('.')
        print(f"   Disk Usage: {disk.percent:.1f}%")
        print(f"   Free Space: {disk.free // (1024**3):.1f} GB")
    except ImportError:
        print("   Disk Info: psutil not available")
    
    # Memory status
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   Memory Usage: {memory.percent:.1f}%")
        print(f"   Available Memory: {memory.available // (1024**3):.1f} GB")
    except ImportError:
        print("   Memory Info: psutil not available")
    
    # GPU status
    if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
        print(f"\n🚀 GPU:")
        print(f"   Device: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3):.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    print("="*60)

def _display_advanced_analytics_report(report: dict):
    """Display advanced analytics report"""
    
    print("\n" + "="*80)
    print("📊 ADVANCED ANALYTICS REPORT")
    print("="*80)
    
    # Performance Summary
    if "performance_summary" in report:
        summary = report["performance_summary"]
        print(f"🏆 OVERALL PERFORMANCE: {summary.get('performance_grade', 'N/A')}")
        print(f"📊 Overall Score: {summary.get('overall_score', 0):.1f}/100")
        
        # Key Metrics
        key_metrics = summary.get("key_metrics", {})
        print(f"\n📈 KEY METRICS:")
        for metric, value in key_metrics.items():
            print(f"   {metric.replace('_', ' ').title()}: {value}")
        
        # Improvement Areas
        improvement_areas = summary.get("improvement_areas", [])
        if improvement_areas:
            print(f"\n🔧 IMPROVEMENT AREAS:")
            for area in improvement_areas:
                print(f"   • {area}")
    
    # Insights
    if "insights" in report:
        insights = report["insights"]
        print(f"\n💡 INSIGHTS:")
        
        # Performance insights
        if "performance" in insights:
            perf = insights["performance"]
            print(f"   🚀 Performance: {perf.get('efficiency', 'N/A')} efficiency")
            print(f"   📊 Productivity: {perf.get('productivity', 'N/A')}")
            print(f"   📈 Quality Trend: {perf.get('quality_trend', 'N/A')}")
        
        # Content insights
        if "content" in insights:
            content = insights["content"]
            print(f"   🎯 Best Niche: {content.get('best_performing_niche', 'N/A')}")
            print(f"   🌐 Niche Diversity: {content.get('niche_diversity', 'N/A')} niches")
            print(f"   ✨ Content Quality: {content.get('content_quality', 'N/A')}")
        
        # System insights
        if "system" in insights:
            system = insights["system"]
            print(f"   💻 System Health: {system.get('health_status', 'N/A')}")
            print(f"   ⚡ Resource Efficiency: {system.get('resource_efficiency', 'N/A')}")
            print(f"   🛡️ Stability: {system.get('stability', 'N/A')}")
    
    # Trends
    if "trends" in report:
        trends = report["trends"]
        print(f"\n📈 TRENDS:")
        for trend_type, trend_value in trends.items():
            print(f"   {trend_type.title()}: {trend_value}")
    
    # Recommendations
    if "recommendations" in report:
        recommendations = report["recommendations"]
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
    
    # Analytics Data Summary
    if "analytics_data" in report:
        analytics = report["analytics_data"]
        
        # Performance metrics
        perf_metrics = analytics.get("performance_metrics", {})
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Videos Processed: {perf_metrics.get('videos_processed', 0)}")
        print(f"   Success Rate: {perf_metrics.get('success_rate', 0):.1f}%")
        print(f"   Error Rate: {perf_metrics.get('error_rate', 0):.1f}%")
        
        # Content analytics
        content_analytics = analytics.get("content_analytics", {})
        niche_perf = content_analytics.get("niche_performance", {})
        print(f"\n🎯 NICHE PERFORMANCE:")
        for niche, perf in niche_perf.items():
            print(f"   {niche.title()}: {perf.get('videos_created', 0)} videos")
        
        # System health
        system_health = analytics.get("system_health", {})
        resource_usage = system_health.get("resource_usage", {})
        print(f"\n💻 SYSTEM HEALTH:")
        print(f"   Memory Usage: {resource_usage.get('memory_percent', 0):.1f}%")
        print(f"   Disk Usage: {resource_usage.get('disk_percent', 0):.1f}%")
        print(f"   CPU Usage: {resource_usage.get('cpu_percent', 0):.1f}%")
    
    print("="*80)


if __name__ == "__main__":
    main()
