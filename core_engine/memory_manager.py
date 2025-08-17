# -*- coding: utf-8 -*-
"""
🧠 MEMORY MANAGER - GPU Memory Management & Optimization
Handles GPU memory allocation, cleanup, and optimization
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

@dataclass
class MemoryStatus:
    """Memory status information"""
    gpu_available: bool
    total_gpu_memory_gb: float
    allocated_gpu_memory_gb: float
    free_gpu_memory_gb: float
    gpu_memory_usage_percent: float
    system_memory_available_gb: float
    system_memory_usage_percent: float
    timestamp: float

class MemoryManager:
    """
    🧠 Memory Manager for GPU and System Memory Optimization
    
    Features:
    - GPU memory monitoring and cleanup
    - System memory monitoring
    - Automatic memory management
    - Memory-efficient model loading
    - Fallback to CPU when needed
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Memory Manager"""
        self.config = config or {}
        self.setup_logging()
        
        # Memory thresholds
        self.gpu_memory_limit_gb = self.config.get('gpu_memory_limit_gb', 2.0)
        self.cpu_fallback_threshold = self.config.get('cpu_fallback_threshold', 1.0)
        self.auto_cleanup = self.config.get('auto_memory_management', True)
        
        # Memory monitoring
        self.memory_history: List[MemoryStatus] = []
        self.cleanup_count = 0
        self.last_cleanup = time.time()
        
        # Initialize memory monitoring
        self._initialize_memory_monitoring()
        
        logging.info("🧠 Memory Manager initialized successfully!")
    
    def setup_logging(self):
        """Setup logging for memory manager"""
        self.logger = logging.getLogger(__name__)
    
    def _initialize_memory_monitoring(self):
        """Initialize memory monitoring systems"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.logger.info("✅ GPU memory monitoring enabled")
            # Set PyTorch memory management
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        else:
            self.logger.info("⚠️ GPU memory monitoring not available")
        
        if PSUTIL_AVAILABLE:
            self.logger.info("✅ System memory monitoring enabled")
        else:
            self.logger.info("⚠️ System memory monitoring not available")
    
    def get_memory_status(self) -> MemoryStatus:
        """Get comprehensive memory status"""
        try:
            # GPU memory status
            if TORCH_AVAILABLE and torch.cuda.is_available():
                total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated_gpu = torch.cuda.memory_allocated(0) / 1024**3
                free_gpu = total_gpu - allocated_gpu
                gpu_usage_percent = (allocated_gpu / total_gpu) * 100
                gpu_available = True
            else:
                total_gpu = allocated_gpu = free_gpu = gpu_usage_percent = 0.0
                gpu_available = False
            
            # System memory status
            if PSUTIL_AVAILABLE:
                system_memory = psutil.virtual_memory()
                system_available_gb = system_memory.available / 1024**3
                system_usage_percent = system_memory.percent
            else:
                system_available_gb = system_usage_percent = 0.0
            
            status = MemoryStatus(
                gpu_available=gpu_available,
                total_gpu_memory_gb=total_gpu,
                allocated_gpu_memory_gb=allocated_gpu,
                free_gpu_memory_gb=free_gpu,
                gpu_memory_usage_percent=gpu_usage_percent,
                system_memory_available_gb=system_available_gb,
                system_memory_usage_percent=system_usage_percent,
                timestamp=time.time()
            )
            
            # Store in history
            self.memory_history.append(status)
            if len(self.memory_history) > 100:  # Keep last 100 entries
                self.memory_history.pop(0)
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Error getting memory status: {e}")
            return MemoryStatus(
                gpu_available=False,
                total_gpu_memory_gb=0.0,
                allocated_gpu_memory_gb=0.0,
                free_gpu_memory_gb=0.0,
                gpu_memory_usage_percent=0.0,
                system_memory_available_gb=0.0,
                system_memory_usage_percent=0.0,
                timestamp=time.time()
            )
    
    def clear_gpu_memory(self) -> bool:
        """Clear GPU memory cache"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.cleanup_count += 1
                self.last_cleanup = time.time()
                self.logger.info("✅ GPU memory cleared successfully")
                return True
            else:
                self.logger.debug("GPU memory clearing not available")
                return False
        except Exception as e:
            self.logger.error(f"❌ GPU memory clearing failed: {e}")
            return False
    
    def should_use_cpu(self, required_memory_gb: float = 1.0) -> bool:
        """Determine if CPU should be used instead of GPU"""
        try:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                return True
            
            status = self.get_memory_status()
            if not status.gpu_available:
                return True
            
            # Check if GPU has enough free memory
            if status.free_gpu_memory_gb < required_memory_gb:
                self.logger.warning(f"⚠️ GPU memory insufficient ({status.free_gpu_memory_gb:.1f}GB < {required_memory_gb}GB), using CPU")
                return True
            
            # Check if GPU memory usage is too high
            if status.gpu_memory_usage_percent > 80:
                self.logger.warning(f"⚠️ GPU memory usage high ({status.gpu_memory_usage_percent:.1f}%), using CPU")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking CPU usage requirement: {e}")
            return True  # Default to CPU on error
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization"""
        try:
            optimization_results = {
                'gpu_memory_cleared': False,
                'system_memory_optimized': False,
                'recommendations': []
            }
            
            # Clear GPU memory
            if self.clear_gpu_memory():
                optimization_results['gpu_memory_cleared'] = True
            
            # System memory optimization
            if PSUTIL_AVAILABLE:
                # Force garbage collection
                import gc
                gc.collect()
                optimization_results['system_memory_optimized'] = True
            
            # Get current status
            current_status = self.get_memory_status()
            
            # Generate recommendations
            if current_status.gpu_available:
                if current_status.gpu_memory_usage_percent > 70:
                    optimization_results['recommendations'].append("GPU memory usage high - consider using CPU for some operations")
                
                if current_status.free_gpu_memory_gb < 0.5:
                    optimization_results['recommendations'].append("GPU memory critically low - switch to CPU mode")
            
            if current_status.system_memory_usage_percent > 80:
                optimization_results['recommendations'].append("System memory usage high - consider closing unnecessary applications")
            
            self.logger.info("✅ Memory optimization completed")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Memory optimization failed: {e}")
            return {'error': str(e)}
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []
        
        try:
            status = self.get_memory_status()
            
            if status.gpu_available:
                if status.gpu_memory_usage_percent > 80:
                    recommendations.append("GPU memory usage very high - clear cache or use CPU")
                
                if status.free_gpu_memory_gb < 1.0:
                    recommendations.append("GPU memory low - consider switching to CPU mode")
                
                if status.gpu_memory_usage_percent > 60:
                    recommendations.append("GPU memory usage moderate - monitor closely")
            
            if status.system_memory_usage_percent > 85:
                recommendations.append("System memory critically low - close applications")
            
            if not recommendations:
                recommendations.append("Memory usage is optimal")
            
        except Exception as e:
            self.logger.error(f"❌ Error getting memory recommendations: {e}")
            recommendations.append("Could not analyze memory status")
        
        return recommendations
    
    def force_cpu_mode(self) -> bool:
        """Force CPU mode for all operations"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Clear GPU memory first
                self.clear_gpu_memory()
                
                # Set environment variable to force CPU
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                self.logger.info("✅ Forced CPU mode - GPU operations disabled")
                return True
            else:
                self.logger.info("GPU not available - already in CPU mode")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to force CPU mode: {e}")
            return False
    
    def get_memory_history(self, limit: int = 10) -> List[MemoryStatus]:
        """Get memory usage history"""
        return self.memory_history[-limit:] if self.memory_history else []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            if not self.memory_history:
                return {'error': 'No memory history available'}
            
            recent_statuses = self.memory_history[-10:]  # Last 10 entries
            
            stats = {
                'total_cleanups': self.cleanup_count,
                'last_cleanup': self.last_cleanup,
                'average_gpu_usage': sum(s.gpu_memory_usage_percent for s in recent_statuses) / len(recent_statuses),
                'peak_gpu_usage': max(s.gpu_memory_usage_percent for s in recent_statuses),
                'average_system_usage': sum(s.system_memory_usage_percent for s in recent_statuses) / len(recent_statuses),
                'peak_system_usage': max(s.system_memory_usage_percent for s in recent_statuses),
                'memory_trend': 'stable'  # Could be calculated based on variance
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting memory stats: {e}")
            return {'error': str(e)}

# Factory function
def create_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
    """Create and configure Memory Manager"""
    try:
        return MemoryManager(config)
    except Exception as e:
        print(f"❌ Failed to create Memory Manager: {e}")
        return None

if __name__ == "__main__":
    # Test the memory manager
    print("🧪 Testing Memory Manager...")
    
    manager = create_memory_manager()
    if manager:
        # Get memory status
        status = manager.get_memory_status()
        print(f"📊 Memory Status: {status}")
        
        # Get recommendations
        recommendations = manager.get_memory_recommendations()
        print(f"💡 Recommendations: {recommendations}")
        
        # Optimize memory
        optimization = manager.optimize_memory()
        print(f"⚡ Optimization: {optimization}")
    else:
        print("❌ Memory Manager creation failed")

