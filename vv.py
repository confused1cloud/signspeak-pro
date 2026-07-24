import urllib.request
import os

url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
filename = "hand_landmarker.task"

print("Downloading hand landmark model...")
print("This may take a minute...")

try:
    urllib.request.urlretrieve(url, filename)
    print(f"✅ Success! Downloaded {filename}")
    print(f"File size: {os.path.getsize(filename)} bytes")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTry Method 1 instead - right-click the link and 'Save as'")