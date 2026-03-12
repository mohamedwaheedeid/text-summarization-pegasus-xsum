import requests

url = "http://127.0.0.1:8000/summarize"  #summarize

text_to_summarize = """
Artificial intelligence is transforming the world in many ways. From healthcare to finance, 
AI systems are being used to analyze large amounts of data and make predictions. 
In the future, we can expect AI to become even more integrated into our daily lives, 
helping us solve complex problems and improve efficiency across various industries.
"""

data = {"text": text_to_summarize}

print("it might take a couple of seconds.")

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        summary = response.json().get("summary")
        print("\n--- summery (Summary) ---")
        print(summary)
    else:
        print(f"error in server: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"failed to connect to the server: {e}")