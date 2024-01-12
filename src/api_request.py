import requests
import emoji_class_unicode_mapping as mp

url = "http://127.0.0.1:8000/emoji_prediction/"
request = {"text": input("Enter a text: ")}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=request, headers=headers)

print(response.status_code)
response_dict: dict = response.json()

for key, value in response_dict.items():
    print(f"{key}: class {value} ( {mp.emoji_unicode[value]} )")
