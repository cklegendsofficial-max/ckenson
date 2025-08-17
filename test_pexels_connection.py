#!/usr/bin/env python3
"""
Test Pexels API connectivity and image download
"""

import requests
import os
import time
from config import PEXELS_API_KEY

def test_pexels_connection():
    """Test Pexels API connectivity"""
    
    print("🔍 Testing Pexels API connectivity...")
    print(f"🔑 API Key: {PEXELS_API_KEY[:20]}...")
    
    # Test basic connectivity
    try:
        # Test DNS resolution
        import socket
        ip = socket.gethostbyname('api.pexels.com')
        print(f"✅ DNS Resolution: api.pexels.com -> {ip}")
    except Exception as e:
        print(f"❌ DNS Resolution failed: {e}")
        return False
    
    # Test HTTPS connection
    try:
        response = requests.get('https://api.pexels.com/v1/search?query=test&per_page=1', 
                              headers={'Authorization': PEXELS_API_KEY}, 
                              timeout=10)
        print(f"✅ HTTPS Connection: Status {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response: {len(data.get('photos', []))} photos found")
            return True
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        print("❌ Connection timeout - check firewall/proxy settings")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_image_download():
    """Test actual image download"""
    
    print("\n📥 Testing image download...")
    
    try:
        # Search for a test image
        url = 'https://api.pexels.com/v1/search'
        params = {
            'query': 'cinematic history',
            'per_page': 1,
            'orientation': 'landscape'
        }
        headers = {'Authorization': PEXELS_API_KEY}
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            photos = data.get('photos', [])
            
            if photos:
                photo = photos[0]
                image_url = photo['src']['large2x']
                print(f"✅ Found image: {photo['alt']}")
                print(f"📸 URL: {image_url}")
                
                # Download image
                print("⬇️ Downloading image...")
                img_response = requests.get(image_url, timeout=15)
                
                if img_response.status_code == 200:
                    # Save to test directory
                    test_dir = "test_pexels_download"
                    os.makedirs(test_dir, exist_ok=True)
                    
                    filename = f"{test_dir}/test_image_{int(time.time())}.jpg"
                    with open(filename, 'wb') as f:
                        f.write(img_response.content)
                    
                    file_size = len(img_response.content) / 1024
                    print(f"✅ Image downloaded: {filename} ({file_size:.1f}KB)")
                    return True
                else:
                    print(f"❌ Image download failed: {img_response.status_code}")
                    return False
            else:
                print("❌ No photos found in search results")
                return False
        else:
            print(f"❌ Search failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        return False

def test_network_connectivity():
    """Test general network connectivity"""
    
    print("\n🌐 Testing network connectivity...")
    
    test_urls = [
        'https://www.google.com',
        'https://api.pexels.com',
        'https://httpbin.org/get'
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"✅ {url}: {response.status_code}")
        except Exception as e:
            print(f"❌ {url}: {e}")

if __name__ == "__main__":
    print("🚀 Pexels API Connection Test")
    print("=" * 40)
    
    # Test network connectivity first
    test_network_connectivity()
    
    # Test Pexels API
    if test_pexels_connection():
        # Test image download
        test_image_download()
    else:
        print("\n❌ Pexels API connection failed. Check:")
        print("   - Internet connection")
        print("   - Firewall settings")
        print("   - Proxy configuration")
        print("   - API key validity")
    
    print("\n" + "=" * 40)
    print("🏁 Test completed")
