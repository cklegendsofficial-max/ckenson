#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 Pexels API Integration
High-quality stock photo and video integration
"""

import os
import requests
import logging
from typing import List, Dict, Optional
from urllib.parse import quote

class PexelsImageDownloader:
    """Pexels API ile görsel indirme"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.pexels.com/v1"
        self.headers = {"Authorization": api_key}
        
    def search_images(self, query: str, per_page: int = 15, page: int = 1) -> List[Dict]:
        """Pexels'den görsel ara"""
        try:
            url = f"{self.base_url}/search"
            params = {
                "query": query,
                "per_page": per_page,
                "page": page,
                "orientation": "landscape"  # Video için landscape tercih et
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("photos", [])
            
        except Exception as e:
            logging.error(f"❌ Pexels search failed: {e}")
            return []
    
    def download_image(self, image_url: str, output_path: str) -> bool:
        """Görseli indir ve kaydet"""
        try:
            # Dizin oluştur
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Görseli indir
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"✅ Image downloaded: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Image download failed: {e}")
            return False
    
    def get_popular_images(self, per_page: int = 15) -> List[Dict]:
        """Popüler görselleri getir"""
        try:
            url = f"{self.base_url}/curated"
            params = {"per_page": per_page}
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("photos", [])
            
        except Exception as e:
            logging.error(f"❌ Popular images fetch failed: {e}")
            return []

# Test fonksiyonu
if __name__ == "__main__":
    # Test için
    api_key = os.getenv("PEXELS_API_KEY")
    if api_key:
        downloader = PexelsImageDownloader(api_key)
        photos = downloader.search_images("nature landscape", 5)
        print(f"Found {len(photos)} photos")
    else:
        print("❌ Pexels API key not found")




