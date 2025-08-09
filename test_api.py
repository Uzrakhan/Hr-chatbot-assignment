import requests
import json

url = "http://127.0.0.1:8000/chat"
headers = {"Content-Type": "application/json"}
data = {"query": "Who has worked on healthcare projects?"}

response = requests.post(url, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    print(json.dumps(response.json(), indent=2))
else:
    print("Error:", response.status_code, response.text)