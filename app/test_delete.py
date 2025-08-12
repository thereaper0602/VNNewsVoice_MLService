# test_delete_enhanced.py
import requests

def test_enhanced_delete():
    """Test enhanced delete methods"""
    
    url = "http://127.0.0.1:8000/api/v1/tts/delete-by-url"
    data = {
        "audio_url": "https://res.cloudinary.com/dg66aou8q/raw/upload/v1753693204/vnnews/audio/tts_1753693242_zephyr.wav"
    }
    
    print("🧪 Testing Enhanced Delete...")
    print(f"🔗 URL: {data['audio_url']}")
    
    response = requests.delete(url, json=data)
    
    print(f"\n📊 Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("🎉 DELETE SUCCESS!")
        data_item = result['data'][0]
        print(f"✅ Deleted: {data_item['deleted']}")
        print(f"🔑 Public ID: {data_item['public_id']}")
        print(f"🔧 Method used: {data_item['method']}")
        
    elif response.status_code == 404:
        print("⚠️ FILE NOT FOUND")
        try:
            error = response.json()
            print(f"📝 Detail: {error.get('detail')}")
        except:
            print(f"📝 Raw: {response.text}")
    else:
        print(f"❌ ERROR {response.status_code}")
        print(f"📝 Response: {response.text}")

if __name__ == "__main__":
    test_enhanced_delete()