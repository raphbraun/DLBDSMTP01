import time
import random
import requests

url = "http://127.0.0.1:5000/api/predict"

while True:
    # Generate random sensor values
    payload = {
        "temperature": round(random.uniform(50, 100), 2),  # °F
        "vibration": round(random.uniform(0.1, 5.0), 2),   # arbitrary vibration scale
        "humidity": round(random.uniform(10, 90), 2)       # %
    }

    try:
        response = requests.post(url, json=payload)
        print(f"\nSent: {payload}")
        print(f"Received: {response.json()}")
    except Exception as e:
        print("Request failed:", e)

    time.sleep(5)  # wait 5 seconds before next post
