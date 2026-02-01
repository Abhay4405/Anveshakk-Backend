"""
Test script to verify face matching behavior with dummy images
"""
import requests
import json

BASE_URL = "http://localhost:9000"

# Test with URLs that don't have faces
test_cases = [
    {
        "name": "Document to Document",
        "img1": "https://via.placeholder.com/300x400?text=Document1",
        "img2": "https://via.placeholder.com/300x400?text=Document2"
    },
    {
        "name": "Random image to Random image",
        "img1": "https://picsum.photos/300/400?random=1",
        "img2": "https://picsum.photos/300/400?random=2"
    }
]

print("Testing Face Matching with enforce_detection=True\n")
print("=" * 60)

for test in test_cases:
    print(f"\n📝 Test: {test['name']}")
    print(f"   Image 1: {test['img1']}")
    print(f"   Image 2: {test['img2']}")
    
    payload = {
        "img1_url": test["img1"],
        "img2_url": test["img2"]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/match-face",
            json=payload,
            timeout=30
        )
        
        print(f"\n   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ❌ UNEXPECTED SUCCESS!")
            print(f"      Matched: {result.get('matched')}")
            print(f"      Confidence: {result.get('confidence')}")
        else:
            print(f"   ✓ Got Error (Expected!)")
            print(f"      Error: {response.text[:100]}")
            
    except Exception as e:
        print(f"   ✓ Exception (Expected!): {str(e)[:100]}")

print("\n" + "=" * 60)
print("\n✓ If all tests showed errors/exceptions, backend is working correctly!")
