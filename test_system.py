#!/usr/bin/env python3
"""
AI Master Suite System Test Script
Tests all major components and reports system status
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    logger.info("🔍 Testing imports...")
    
    tests = [
        ("config", "Configuration module"),
        ("improved_llm_handler", "LLM Handler"),
        ("advanced_video_creator", "Video Creator"),
        ("moviepy", "Video processing"),
        ("ollama", "Ollama client"),
        ("requests", "HTTP requests"),
        ("numpy", "Numerical operations"),
        ("PIL", "Image processing"),
    ]
    
    results = {}
    for module, description in tests:
        try:
            if module == "PIL":
                import PIL
                results[module] = "✅ OK"
            else:
                __import__(module)
                results[module] = "✅ OK"
        except ImportError as e:
            results[module] = f"❌ Failed: {e}"
        except Exception as e:
            results[module] = f"⚠️ Error: {e}"
    
    for module, result in results.items():
        logger.info(f"  {module:25} {result}")
    
    return results

def test_configuration():
    """Test configuration loading"""
    logger.info("⚙️ Testing configuration...")
    
    try:
        from config import CHANNELS_CONFIG, AI_CONFIG, PEXELS_API_KEY
        
        # Test channels
        channel_count = len(CHANNELS_CONFIG)
        logger.info(f"  Channels configured: {channel_count}")
        for channel in CHANNELS_CONFIG.keys():
            logger.info(f"    - {channel}")
        
        # Test AI config
        ollama_model = AI_CONFIG.get("ollama_model", "Not set")
        logger.info(f"  Ollama model: {ollama_model}")
        
        # Test API keys
        pexels_enabled = bool(PEXELS_API_KEY)
        logger.info(f"  Pexels API: {'✅ Enabled' if pexels_enabled else '❌ Disabled'}")
        
        return True
        
    except Exception as e:
        logger.error(f"  Configuration test failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama server connection"""
    logger.info("🤖 Testing Ollama connection...")
    
    try:
        import ollama
        
        # Test basic connection
        client = ollama.Client()
        models = client.list()
        
        if models and hasattr(models, 'models'):
            model_count = len(models.models)
            logger.info(f"  Connected to Ollama server")
            logger.info(f"  Available models: {model_count}")
            for model in models.models[:3]:  # Show first 3
                logger.info(f"    - {model.name}")
            if model_count > 3:
                logger.info(f"    ... and {model_count - 3} more")
            return True
        else:
            logger.warning("  Ollama server responded but no models found")
            return False
            
    except Exception as e:
        logger.error(f"  Ollama connection failed: {e}")
        return False

def test_video_processing():
    """Test video processing capabilities"""
    logger.info("🎬 Testing video processing...")
    
    try:
        from moviepy.editor import VideoFileClip
        from moviepy.config import change_settings
        
        logger.info("  MoviePy: ✅ Available")
        
        # Test if we can create a simple video
        try:
            # Create a simple test clip
            from moviepy.video.VideoClip import ColorClip
            test_clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=1)
            logger.info("  Video creation: ✅ Working")
        except Exception as e:
            logger.warning(f"  Video creation: ⚠️ Limited - {e}")
        
        return True
        
    except ImportError:
        logger.error("  MoviePy: ❌ Not available")
        return False
    except Exception as e:
        logger.error(f"  Video processing test failed: {e}")
        return False

def test_file_structure():
    """Test required file structure"""
    logger.info("📁 Testing file structure...")
    
    required_dirs = [
        "assets/audio/music",
        "assets/videos/downloads",
        "assets/images",
        "logs",
        "outputs"
    ]
    
    required_files = [
        "config.py",
        "main.py",
        "improved_llm_handler.py",
        "advanced_video_creator.py"
    ]
    
    # Test directories
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            logger.info(f"  {dir_path}: ✅ Exists")
        else:
            logger.info(f"  {dir_path}: ❌ Missing")
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  {dir_path}: 🆕 Created")
    
    # Test files
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            logger.info(f"  {file_path}: ✅ Exists")
        else:
            logger.error(f"  {file_path}: ❌ Missing")

def test_environment():
    """Test environment variables"""
    logger.info("🌍 Testing environment...")
    
    env_vars = [
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "PEXELS_API_KEY",
        "ELEVENLABS_API_KEY"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if var.endswith("_KEY"):
                # Mask API keys
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                logger.info(f"  {var}: ✅ Set ({masked_value})")
            else:
                logger.info(f"  {var}: ✅ Set ({value})")
        else:
            if var in ["PEXELS_API_KEY", "ELEVENLABS_API_KEY"]:
                logger.warning(f"  {var}: ⚠️ Not set (optional)")
            else:
                logger.info(f"  {var}: ℹ️ Not set (using default)")

def test_optimization_features():
    """Test optimization features"""
    logger.info("🚀 Testing optimization features...")
    
    results = {}
    
    # Test GPU acceleration
    try:
        import torch
        if torch.cuda.is_available():
            results['gpu_acceleration'] = "✅ Available"
            results['gpu_device'] = torch.cuda.get_device_name()
            results['cuda_version'] = torch.version.cuda
        else:
            results['gpu_acceleration'] = "❌ Not available"
    except ImportError:
        results['gpu_acceleration'] = "⚠️ PyTorch not installed"
    
    # Test batch processing
    try:
        import concurrent.futures
        results['batch_processing'] = "✅ Available"
    except ImportError:
        results['batch_processing'] = "❌ Not available"
    
    # Test memory management
    try:
        import psutil
        memory = psutil.virtual_memory()
        results['memory_management'] = "✅ Available"
        results['total_memory'] = f"{memory.total // (1024**3):.1f} GB"
        results['available_memory'] = f"{memory.available // (1024**3):.1f} GB"
    except ImportError:
        results['memory_management'] = "❌ psutil not installed"
    
    # Test storage optimization
    try:
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        results['storage_optimization'] = "✅ Available"
        results['temp_dir'] = temp_dir
        # Cleanup
        os.rmdir(temp_dir)
    except Exception as e:
        results['storage_optimization'] = f"❌ Failed: {e}"
    
    # Test audio enhancement
    try:
        import librosa
        import soundfile
        results['audio_enhancement'] = "✅ Available"
    except ImportError:
        results['audio_enhancement'] = "⚠️ Audio libraries not installed"
    
    # Test video optimization
    try:
        from moviepy.editor import VideoFileClip
        results['video_optimization'] = "✅ Available"
    except ImportError:
        results['video_optimization'] = "❌ MoviePy not available"
    
    # Display results
    for feature, status in results.items():
        logger.info(f"  {feature:20} {status}")
    
    return results

def test_performance_benchmark():
    """Test performance benchmark"""
    logger.info("⚡ Running performance benchmark...")
    
    import time
    import psutil
    
    results = {}
    
    # CPU benchmark
    start_time = time.time()
    sum_result = sum(range(1000000))
    cpu_time = time.time() - start_time
    results['cpu_benchmark'] = f"{cpu_time:.4f}s"
    
    # Memory benchmark
    memory = psutil.virtual_memory()
    results['memory_usage'] = f"{memory.percent:.1f}%"
    results['available_memory'] = f"{memory.available // (1024**3):.1f} GB"
    
    # Disk benchmark
    try:
        disk = psutil.disk_usage('.')
        results['disk_usage'] = f"{disk.percent:.1f}%"
        results['free_space'] = f"{disk.free // (1024**3):.1f} GB"
    except Exception:
        results['disk_usage'] = "N/A"
    
    # GPU benchmark (if available)
    try:
        import torch
        if torch.cuda.is_available():
            start_time = time.time()
            # Simple GPU operation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            results['gpu_benchmark'] = f"{gpu_time:.4f}s"
            results['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory // (1024**3):.1f} GB"
        else:
            results['gpu_benchmark'] = "N/A"
    except ImportError:
        results['gpu_benchmark'] = "PyTorch not available"
    
    # Display results
    logger.info("📊 Performance Benchmark Results:")
    for metric, value in results.items():
        logger.info(f"  {metric:20} {value}")
    
    return results

def run_performance_test():
    """Run a simple performance test"""
    logger.info("⚡ Running performance test...")
    
    start_time = time.time()
    
    try:
        # Test LLM handler initialization
        from improved_llm_handler import ImprovedLLMHandler
        
        init_start = time.time()
        handler = ImprovedLLMHandler(ollama_model="llama3:8b")
        init_time = time.time() - init_start
        
        logger.info(f"  LLM Handler init: {init_time:.2f}s")
        
        # Test basic idea generation
        idea_start = time.time()
        ideas = handler.generate_viral_ideas("CKLegends", 1)
        idea_time = time.time() - idea_start
        
        if ideas:
            logger.info(f"  Idea generation: {idea_time:.2f}s ✅")
        else:
            logger.warning(f"  Idea generation: {idea_time:.2f}s ⚠️ (no ideas returned)")
        
        total_time = time.time() - start_time
        logger.info(f"  Total test time: {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"  Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting AI Master Suite System Test")
    logger.info("=" * 50)
    
    # Run tests
    test_imports()
    logger.info("")
    
    test_configuration()
    logger.info("")
    
    test_ollama_connection()
    logger.info("")
    
    test_video_processing()
    logger.info("")
    
    test_file_structure()
    logger.info("")
    
    test_environment()
    logger.info("")
    
    test_optimization_features()
    logger.info("")

    test_performance_benchmark()
    logger.info("")
    
    # Performance test (optional)
    try:
        run_performance_test()
    except Exception as e:
        logger.warning(f"Performance test skipped: {e}")
    
    logger.info("=" * 50)
    logger.info("✅ System test completed!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Set up your .env file with API keys")
    logger.info("2. Ensure Ollama server is running")
    logger.info("3. Run: python main.py --analyze")
    logger.info("4. Run: python main.py --single CKLegends")

if __name__ == "__main__":
    main()

