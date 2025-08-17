# -*- coding: utf-8 -*-
"""
🚀 AI INTEGRATED SUITE - KUSURSUZ ENTEGRASYON
Tüm AI modüllerini birleştiren kusursuz sistem
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Core imports with better error handling
try:
    from improved_llm_handler import ImprovedLLMHandler
    IMPROVED_LLM_AVAILABLE = True
except ImportError as e:
    IMPROVED_LLM_AVAILABLE = False
    print(f"⚠️ Improved LLM Handler not available: {e}")

try:
    from advanced_video_creator import AdvancedVideoCreator
    ADVANCED_VIDEO_AVAILABLE = True
except ImportError as e:
    ADVANCED_VIDEO_AVAILABLE = False
    print(f"⚠️ Advanced Video Creator not available: {e}")

# PyTorch import for memory management
try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available for memory management")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available for memory management")

# Memory Manager import
try:
    from core_engine.memory_manager import MemoryManager, create_memory_manager
    MEMORY_MANAGER_AVAILABLE = True
    print("✅ Memory Manager available")
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    print(f"⚠️ Memory Manager not available: {e}")

# AI Suite imports with comprehensive error handling
AI_MODULES = {}

try:
    from ai_cinematic_director import CinematicAIDirector, StoryArc, CinematicStyle, PacingType
    AI_MODULES['cinematic'] = {
        'available': True,
        'class': CinematicAIDirector,
        'enums': {'StoryArc': StoryArc, 'CinematicStyle': CinematicStyle, 'PacingType': PacingType}
    }
    print("✅ Cinematic Director imported successfully")
except ImportError as e:
    AI_MODULES['cinematic'] = {'available': False, 'error': str(e)}
    print(f"⚠️ Cinematic Director not available: {e}")

try:
    from ai_advanced_voice_acting import AdvancedVoiceActingEngine, Emotion, VoiceStyle, CharacterPersonality
    AI_MODULES['voice_acting'] = {
        'available': True,
        'class': AdvancedVoiceActingEngine,
        'enums': {'Emotion': Emotion, 'VoiceStyle': VoiceStyle, 'CharacterPersonality': CharacterPersonality}
    }
    print("✅ Voice Acting imported successfully")
except ImportError as e:
    AI_MODULES['voice_acting'] = {'available': False, 'error': str(e)}
    print(f"⚠️ Voice Acting not available: {e}")

try:
    from ai_visual_suite import AIVisualSuite, VisualStyle, ImageFormat, EnhancementType
    AI_MODULES['visual'] = {
        'available': True,
        'class': AIVisualSuite,
        'enums': {'VisualStyle': VisualStyle, 'ImageFormat': ImageFormat, 'EnhancementType': EnhancementType}
    }
    print("✅ Visual Suite imported successfully")
except ImportError as e:
    AI_MODULES['visual'] = {'available': False, 'error': str(e)}
    print(f"⚠️ Visual Suite not available: {e}")

try:
    from ai_audio_suite import AIAudioSuite, VoiceStyle as AudioVoiceStyle, Emotion as AudioEmotion
    AI_MODULES['audio'] = {
        'available': True,
        'class': AIAudioSuite,
        'enums': {'VoiceStyle': AudioVoiceStyle, 'Emotion': AudioEmotion}
    }
    print("✅ Audio Suite imported successfully")
except ImportError as e:
    AI_MODULES['audio'] = {'available': False, 'error': str(e)}
    print(f"⚠️ Audio Suite not available: {e}")

try:
    from ai_content_suite import AIContentSuite, ContentType, ContentTone, TargetAudience
    AI_MODULES['content'] = {
        'available': True,
        'class': AIContentSuite,
        'enums': {'ContentType': ContentType, 'ContentTone': ContentTone, 'TargetAudience': TargetAudience}
    }
    print("✅ Content Suite imported successfully")
except ImportError as e:
    AI_MODULES['content'] = {'available': False, 'error': str(e)}
    print(f"⚠️ Content Suite not available: {e}")

try:
    from ai_video_suite import AIVideoSuite, ContentType as VideoContentType, Platform, QualityScore
    AI_MODULES['video'] = {
        'available': True,
        'class': AIVideoSuite,
        'enums': {'ContentType': VideoContentType, 'Platform': Platform, 'QualityScore': QualityScore}
    }
    print("✅ Video Suite imported successfully")
except ImportError as e:
    AI_MODULES['video'] = {'available': False, 'error': str(e)}
    print(f"⚠️ Video Suite not available: {e}")

try:
    from ai_analytics_suite import AIAnalyticsSuite, MetricType, OptimizationType, Platform as AnalyticsPlatform
    AI_MODULES['analytics'] = {
        'available': True,
        'class': AIAnalyticsSuite,
        'enums': {'MetricType': MetricType, 'OptimizationType': OptimizationType, 'Platform': AnalyticsPlatform}
    }
    print("✅ Analytics Suite imported successfully")
except ImportError as e:
    AI_MODULES['analytics'] = {'available': False, 'error': str(e)}
    print(f"⚠️ Analytics Suite not available: {e}")

try:
    from ai_realtime_director import AIContentDirector, OptimizationType as RealtimeOptimizationType
    AI_MODULES['realtime'] = {
        'available': True,
        'class': AIContentDirector,
        'enums': {'OptimizationType': RealtimeOptimizationType}
    }
    print("✅ Realtime Director imported successfully")
except ImportError as e:
    AI_MODULES['realtime'] = {'available': False, 'error': str(e)}
    print(f"⚠️ Realtime Director not available: {e}")

try:
    from ai_master_suite import AIMasterSuite
    AI_MODULES['master'] = {
        'available': True,
        'class': AIMasterSuite
    }
    print("✅ AI Master Suite imported successfully")
except ImportError as e:
    AI_MODULES['master'] = {'available': False, 'error': str(e)}
    print(f"⚠️ AI Master Suite not available: {e}")

# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class IntegrationStatus(Enum):
    AVAILABLE = "available"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"
    ERROR = "error"

class ContentQuality(Enum):
    EXCELLENT = "excellent"      # 90-100
    GOOD = "good"               # 75-89
    AVERAGE = "average"         # 60-74
    POOR = "poor"               # 40-59
    FAIL = "fail"               # 0-39

@dataclass
class IntegrationStatus:
    """AI modül entegrasyon durumu"""
    module_name: str
    status: IntegrationStatus
    available: bool
    error_message: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.now)
    capabilities: List[str] = field(default_factory=list)

@dataclass
class AISystemHealth:
    """AI sistem sağlık durumu"""
    overall_status: str
    available_modules: int
    total_modules: int
    critical_errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    last_health_check: datetime = field(default_factory=datetime.now)

# =============================================================================
# AI INTEGRATED SUITE - ANA SINIF
# =============================================================================

class AIIntegratedSuite:
    """
    🚀 AI INTEGRATED SUITE - KUSURSUZ ENTEGRASYON
    
    Tüm AI modüllerini birleştiren, hata toleranslı sistem
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AI Integrated Suite with configuration"""
        self.config = config or {}
        self.llm_handler = None
        self.video_creator = None
        self.ai_modules = {}
        self.system_health = None
        
        # Initialize memory manager
        self.memory_manager = None
        if MEMORY_MANAGER_AVAILABLE:
            try:
                self.memory_manager = create_memory_manager(self.config.get('memory_config', {}))
                print("✅ Memory Manager initialized")
            except Exception as e:
                print(f"⚠️ Memory Manager initialization failed: {e}")
        
        # Initialize core components
        self._initialize_core_components()
        
        # Initialize AI modules
        self._initialize_ai_modules()
        
        # Perform health check
        self._perform_health_check()
        
        print(f"🚀 AI Integrated Suite initialized with {len([m for m in self.ai_modules.values() if m.get('available')])} available modules")
    
    def _initialize_core_components(self):
        """Initialize core components with error handling"""
        try:
            if IMPROVED_LLM_AVAILABLE:
                self.llm_handler = ImprovedLLMHandler()
                print("✅ LLM Handler initialized")
            else:
                print("⚠️ LLM Handler not available")
        except Exception as e:
            print(f"❌ LLM Handler initialization failed: {e}")
        
        try:
            if ADVANCED_VIDEO_AVAILABLE:
                # Initialize video creator
                self.video_creator = AdvancedVideoCreator()
                print("✅ Video Creator initialized")
            else:
                print("⚠️ Video Creator not available")
        except Exception as e:
            print(f"❌ Video Creator initialization failed: {e}")
    
    def _initialize_ai_modules(self):
        """Initialize available AI modules sequentially to prevent memory conflicts"""
        print("🔄 Initializing AI modules sequentially to prevent conflicts...")
        
        # Initialize modules in priority order with delays
        priority_modules = ['master', 'cinematic', 'content', 'visual', 'audio', 'video', 'analytics', 'realtime']
        
        for module_name in priority_modules:
            if module_name in AI_MODULES and AI_MODULES[module_name].get('available'):
                try:
                    print(f"🔄 Initializing {module_name} module...")
                    module_class = AI_MODULES[module_name]['class']
                    
                    # Add small delay between initializations to prevent memory conflicts
                    import time
                    time.sleep(0.5)
                    
                    module_instance = module_class()
                    self.ai_modules[module_name] = {
                        'instance': module_instance,
                        'available': True,
                        'capabilities': self._get_module_capabilities(module_name, module_instance)
                    }
                    print(f"✅ {module_name} module initialized successfully")
                    
                    # Clear GPU cache if available to free memory
                    try:
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                        
                except Exception as e:
                    print(f"❌ {module_name} module initialization failed: {e}")
                    self.ai_modules[module_name] = {
                        'available': False,
                        'error': str(e)
                    }
                    
                    # Try to recover memory
                    try:
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
            else:
                if module_name in AI_MODULES:
                    self.ai_modules[module_name] = AI_MODULES[module_name]
                else:
                    self.ai_modules[module_name] = {'available': False, 'error': 'Module not found'}
        
        print(f"✅ AI module initialization completed. {len([m for m in self.ai_modules.values() if m.get('available')])} modules available.")
    
    def _get_module_capabilities(self, module_name: str, module_instance) -> List[str]:
        """Get capabilities of a specific module"""
        capabilities = []
        
        try:
            if hasattr(module_instance, 'get_capabilities'):
                capabilities = module_instance.get_capabilities()
            elif hasattr(module_instance, 'capabilities'):
                capabilities = module_instance.capabilities
            else:
                # Default capabilities based on module name
                default_capabilities = {
                    'cinematic': ['story_creation', 'pacing_control', 'emotional_arcs'],
                    'voice_acting': ['emotion_synthesis', 'voice_generation', 'character_voices'],
                    'visual': ['image_enhancement', 'style_transfer', 'quality_upscaling'],
                    'audio': ['voice_synthesis', 'music_generation', 'audio_enhancement'],
                    'content': ['script_generation', 'content_optimization', 'audience_targeting'],
                    'video': ['video_creation', 'quality_optimization', 'platform_adaptation'],
                    'analytics': ['performance_metrics', 'optimization_suggestions', 'trend_analysis'],
                    'realtime': ['live_optimization', 'dynamic_content', 'performance_monitoring'],
                    'master': ['comprehensive_content_creation', 'multi_platform_optimization', 'ai_orchestration']
                }
                capabilities = default_capabilities.get(module_name, ['basic_functionality'])
        except Exception as e:
            print(f"⚠️ Could not get capabilities for {module_name}: {e}")
            capabilities = ['basic_functionality']
        
        return capabilities
    
    def _perform_health_check(self):
        """Perform comprehensive system health check with memory management"""
        available_modules = len([m for m in self.ai_modules.values() if m.get('available')])
        total_modules = len(self.ai_modules)
        
        critical_errors = []
        warnings = []
        recommendations = []
        
        # Check core components
        if not self.llm_handler:
            critical_errors.append("LLM Handler not available")
            recommendations.append("Install improved_llm_handler or check dependencies")
        
        if not self.video_creator:
            critical_errors.append("Video Creator not available")
            recommendations.append("Install advanced_video_creator or check dependencies")
        
        # Check AI modules
        for module_name, module_info in self.ai_modules.items():
            if not module_info.get('available'):
                warnings.append(f"{module_name} module not available")
                if module_info.get('error'):
                    warnings.append(f"  - Error: {module_info['error']}")
        
        # Check memory usage
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
                free_memory = gpu_memory - allocated_memory
                
                if free_memory < 1.0:  # Less than 1GB free
                    warnings.append(f"GPU memory low: {free_memory:.1f}GB free out of {gpu_memory:.1f}GB")
                    recommendations.append("Consider clearing GPU memory or using CPU mode")
                
                print(f"📊 GPU Memory: {allocated_memory:.1f}GB used, {free_memory:.1f}GB free")
        except Exception as e:
            print(f"⚠️ Could not check GPU memory: {e}")
        
        # Determine overall status
        if critical_errors:
            overall_status = "CRITICAL"
        elif warnings:
            overall_status = "WARNING"
        else:
            overall_status = "HEALTHY"
        
        # Generate recommendations
        if available_modules < total_modules * 0.5:
            recommendations.append("Consider installing missing AI modules for full functionality")
        
        if not self.llm_handler and not self.video_creator:
            recommendations.append("Core components missing - system may not function properly")
        
        self.system_health = AISystemHealth(
            overall_status=overall_status,
            available_modules=available_modules,
            total_modules=total_modules,
            critical_errors=critical_errors,
            warnings=warnings,
            recommendations=recommendations
        )
        
        print(f"🔍 System Health: {overall_status} ({available_modules}/{total_modules} modules available)")
    
    def clear_memory(self):
        """Clear memory to prevent conflicts"""
        try:
            if self.memory_manager:
                return self.memory_manager.optimize_memory()
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✅ GPU memory cleared")
                return {'gpu_memory_cleared': True}
            else:
                print("⚠️ No memory management available")
                return {'error': 'No memory management available'}
        except Exception as e:
            print(f"⚠️ Could not clear memory: {e}")
            return {'error': str(e)}
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        try:
            if self.memory_manager:
                return self.memory_manager.get_memory_status().__dict__
            else:
                # Fallback to basic PyTorch memory check
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
                    free_memory = gpu_memory - allocated_memory
                    
                    return {
                        'gpu_available': True,
                        'total_gpu_memory_gb': gpu_memory,
                        'allocated_gpu_memory_gb': allocated_memory,
                        'free_gpu_memory_gb': free_memory,
                        'memory_usage_percent': (allocated_memory / gpu_memory) * 100
                    }
                else:
                    return {
                        'gpu_available': False,
                        'message': 'GPU not available or PyTorch not installed'
                    }
        except Exception as e:
            return {
                'gpu_available': False,
                'error': str(e)
            }
    
    def should_use_cpu(self, required_memory_gb: float = 1.0) -> bool:
        """Determine if CPU should be used instead of GPU"""
        try:
            if self.memory_manager:
                return self.memory_manager.should_use_cpu(required_memory_gb)
            else:
                # Basic fallback logic
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
                    free_memory = gpu_memory - allocated_memory
                    return free_memory < required_memory_gb
                else:
                    return True
        except Exception as e:
            print(f"⚠️ Error checking CPU usage requirement: {e}")
            return True  # Default to CPU on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_health': {
                'overall_status': self.system_health.overall_status,
                'available_modules': self.system_health.available_modules,
                'total_modules': self.system_health.total_modules,
                'critical_errors': self.system_health.critical_errors,
                'warnings': self.system_health.warnings,
                'recommendations': self.system_health.recommendations,
                'last_health_check': self.system_health.last_health_check.isoformat()
            },
            'core_components': {
                'llm_handler': self.llm_handler is not None,
                'video_creator': self.video_creator is not None
            },
            'ai_modules': {
                name: {
                    'available': info.get('available', False),
                    'capabilities': info.get('capabilities', []),
                    'error': info.get('error', None)
                }
                for name, info in self.ai_modules.items()
            }
        }
    
    def run_full_pipeline(self, channel_name: str, niche: str, target_duration: int = 15) -> Dict[str, Any]:
        """Run full AI-powered content creation pipeline"""
        try:
            print(f"🚀 Starting full AI pipeline for {channel_name} (niche: {niche})")
            
            # Step 1: Content Generation
            content_result = self._generate_content(channel_name, niche, target_duration)
            if not content_result.get('success'):
                return {'success': False, 'error': 'Content generation failed'}
            
            # Step 2: Visual Enhancement
            visual_result = self._enhance_visuals(content_result['content'], niche)
            
            # Step 3: Audio Generation
            audio_result = self._generate_audio(content_result['content'], niche)
            
            # Step 4: Video Creation
            video_result = self._create_video(content_result, visual_result, audio_result, target_duration)
            
            # Step 5: Quality Analysis
            quality_result = self._analyze_quality(video_result)
            
            return {
                'success': True,
                'channel': channel_name,
                'niche': niche,
                'content': content_result,
                'visuals': visual_result,
                'audio': audio_result,
                'video': video_result,
                'quality': quality_result,
                'pipeline': {
                    'content_generation': content_result.get('success', False),
                    'visual_enhancement': visual_result.get('success', False),
                    'audio_generation': audio_result.get('success', False),
                    'video_creation': video_result.get('success', False),
                    'quality_analysis': quality_result.get('success', False)
                }
            }
            
        except Exception as e:
            print(f"❌ Full pipeline failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_content(self, channel_name: str, niche: str, target_duration: int) -> Dict[str, Any]:
        """Generate content using available AI modules"""
        try:
            # Try AI Master Suite first (highest priority)
            if self.ai_modules.get('master', {}).get('available'):
                master_suite = self.ai_modules['master']['instance']
                if hasattr(master_suite, 'create_comprehensive_content'):
                    return master_suite.create_comprehensive_content(channel_name, niche, target_duration)
                elif hasattr(master_suite, 'generate_content'):
                    return master_suite.generate_content(channel_name, niche, target_duration)
            
            # Try content suite second
            if self.ai_modules.get('content', {}).get('available'):
                content_suite = self.ai_modules['content']['instance']
                if hasattr(content_suite, 'generate_content'):
                    return content_suite.generate_content(channel_name, niche, target_duration)
            
            # Fallback to LLM handler
            if self.llm_handler:
                script = self.llm_handler.write_script(f"{niche} content for {channel_name}", channel_name)
                return {
                    'success': True,
                    'content': script,
                    'type': 'script',
                    'source': 'llm_handler'
                }
            
            # Final fallback
            return {
                'success': True,
                'content': f"Default {niche} content for {channel_name}",
                'type': 'fallback',
                'source': 'fallback'
            }
            
        except Exception as e:
            print(f"❌ Content generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _enhance_visuals(self, content: str, niche: str) -> Dict[str, Any]:
        """Enhance visuals using AI visual suite"""
        try:
            if self.ai_modules.get('visual', {}).get('available'):
                visual_suite = self.ai_modules['visual']['instance']
                if hasattr(visual_suite, 'enhance_content'):
                    return visual_suite.enhance_content(content, niche)
            
            return {
                'success': True,
                'enhanced': False,
                'message': 'Visual enhancement not available'
            }
            
        except Exception as e:
            print(f"❌ Visual enhancement failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_audio(self, content: str, niche: str) -> Dict[str, Any]:
        """Generate audio using AI audio suite"""
        try:
            if self.ai_modules.get('audio', {}).get('available'):
                audio_suite = self.ai_modules['audio']['instance']
                if hasattr(audio_suite, 'generate_audio'):
                    return audio_suite.generate_audio(content, niche)
            
            return {
                'success': True,
                'generated': False,
                'message': 'Audio generation not available'
            }
            
        except Exception as e:
            print(f"❌ Audio generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_video(self, content_result: Dict, visual_result: Dict, audio_result: Dict, target_duration: int) -> Dict[str, Any]:
        """Create video using available components"""
        try:
            if self.video_creator:
                # Use video creator to generate video
                output_path = f"outputs/{content_result.get('channel', 'unknown')}/ai_generated_video.mp4"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Create basic video (this would be enhanced based on available AI modules)
                result = self.video_creator.create_video(
                    {'script': content_result.get('content', '')},
                    output_folder=os.path.dirname(output_path),
                    target_duration_minutes=target_duration
                )
                
                if result and result.get('success'):
                    return {
                        'success': True,
                        'output_path': result.get('output_path', output_path),
                        'duration': result.get('duration_minutes', target_duration),
                        'quality_score': result.get('quality_score', 0.8)
                    }
            
            return {
                'success': False,
                'error': 'Video creation not available'
            }
            
        except Exception as e:
            print(f"❌ Video creation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_quality(self, video_result: Dict) -> Dict[str, Any]:
        """Analyze video quality using AI analytics"""
        try:
            if self.ai_modules.get('analytics', {}).get('available'):
                analytics_suite = self.ai_modules['analytics']['instance']
                if hasattr(analytics_suite, 'analyze_video'):
                    return analytics_suite.analyze_video(video_result.get('output_path', ''))
            
            # Basic quality analysis
            return {
                'success': True,
                'quality_score': video_result.get('quality_score', 0.8),
                'analysis_type': 'basic',
                'recommendations': ['Consider using AI analytics for detailed analysis']
            }
            
        except Exception as e:
            print(f"❌ Quality analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_available_modules(self) -> List[str]:
        """Get list of available AI modules"""
        return [name for name, info in self.ai_modules.items() if info.get('available')]
    
    def get_module_capabilities(self, module_name: str) -> List[str]:
        """Get capabilities of a specific module"""
        module_info = self.ai_modules.get(module_name, {})
        return module_info.get('capabilities', [])
    
    def is_module_available(self, module_name: str) -> bool:
        """Check if a specific module is available"""
        return self.ai_modules.get(module_name, {}).get('available', False)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        available_modules = len([m for m in self.ai_modules.values() if m.get('available')])
        total_modules = len(self.ai_modules)
        
        # Determine overall status
        if available_modules == total_modules:
            overall_status = "EXCELLENT"
        elif available_modules >= total_modules * 0.8:
            overall_status = "GOOD"
        elif available_modules >= total_modules * 0.6:
            overall_status = "FAIR"
        elif available_modules >= total_modules * 0.4:
            overall_status = "POOR"
        else:
            overall_status = "CRITICAL"
        
        # Get module details
        module_details = {}
        for module_name, module_info in self.ai_modules.items():
            module_details[module_name] = {
                'available': module_info.get('available', False),
                'capabilities': module_info.get('capabilities', []),
                'error': module_info.get('error', None)
            }
        
        return {
            'overall_status': overall_status,
            'system_health': {
                'available_modules': available_modules,
                'total_modules': total_modules,
                'health_percentage': (available_modules / total_modules) * 100 if total_modules > 0 else 0
            },
            'core_components': {
                'llm_handler': bool(self.llm_handler),
                'video_creator': bool(self.video_creator)
            },
            'ai_modules': module_details,
            'recommendations': self._get_system_recommendations(available_modules, total_modules),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_system_recommendations(self, available_modules: int, total_modules: int) -> List[str]:
        """Get system improvement recommendations"""
        recommendations = []
        
        if available_modules < total_modules * 0.5:
            recommendations.append("Install missing AI modules to improve functionality")
        
        if not self.llm_handler:
            recommendations.append("Install improved_llm_handler for better content generation")
        
        if not self.video_creator:
            recommendations.append("Install advanced_video_creator for video processing")
        
        if available_modules >= total_modules * 0.8:
            recommendations.append("System is well configured - consider advanced optimizations")
        
        return recommendations

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_ai_suite(config: Optional[Dict[str, Any]] = None) -> AIIntegratedSuite:
    """Create and configure AI Integrated Suite"""
    try:
        suite = AIIntegratedSuite(config)
        return suite
    except Exception as e:
        print(f"❌ Failed to create AI suite: {e}")
        # Return minimal suite
        return AIIntegratedSuite(config)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_ai_dependencies() -> Dict[str, bool]:
    """Check availability of AI dependencies"""
    dependencies = {
        'improved_llm_handler': IMPROVED_LLM_AVAILABLE,
        'advanced_video_creator': ADVANCED_VIDEO_AVAILABLE,
        'ai_cinematic_director': AI_MODULES.get('cinematic', {}).get('available', False),
        'ai_voice_acting': AI_MODULES.get('voice_acting', {}).get('available', False),
        'ai_visual_suite': AI_MODULES.get('visual', {}).get('available', False),
        'ai_audio_suite': AI_MODULES.get('audio', {}).get('available', False),
        'ai_content_suite': AI_MODULES.get('content', {}).get('available', False),
        'ai_video_suite': AI_MODULES.get('video', {}).get('available', False),
        'ai_analytics_suite': AI_MODULES.get('analytics', {}).get('available', False),
        'ai_realtime_director': AI_MODULES.get('realtime', {}).get('available', False),
        'ai_master_suite': AI_MODULES.get('master', {}).get('available', False)
    }
    
    return dependencies

def get_ai_system_info() -> Dict[str, Any]:
    """Get comprehensive AI system information"""
    return {
        'dependencies': check_ai_dependencies(),
        'available_modules': len([m for m in AI_MODULES.values() if m.get('available')]),
        'total_modules': len(AI_MODULES),
        'system_ready': IMPROVED_LLM_AVAILABLE and ADVANCED_VIDEO_AVAILABLE
    }

if __name__ == "__main__":
    # Test the AI suite
    print("🧪 Testing AI Integrated Suite...")
    
    # Check dependencies
    deps = check_ai_dependencies()
    print(f"📊 Dependencies: {deps}")
    
    # Get system info
    info = get_ai_system_info()
    print(f"ℹ️ System Info: {info}")
    
    # Create suite
    suite = create_ai_suite()
    print(f"✅ Suite created: {suite.get_system_status()}")
