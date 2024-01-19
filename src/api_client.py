import requests
import emoji_class_unicode_mapping as mp


class APIClient:
    def __init__(self):
        self.url: str = "http://127.0.0.1:8000/emoji_prediction/"
        self.headers: dict[str, str] = {"Content-Type": "application/json"}

    def run(self, request_text: str) -> None:
        print("REQUEST")
        request: dict[str, str] = {"text": request_text}

        response = requests.post(self.url, json=request, headers=self.headers)

        print("RESPONSE")
        print("response_status_code:", response.status_code)
        response_dict: dict = response.json()

        for key, value in response_dict.items():
            print(f"{key}: class {value} ( {mp.emoji_unicode[value]} )")


if __name__ == "__main__":
    api_client: APIClient = APIClient()
    input_text: str = input("Enter a text (or a \"quit\" to stop):\n")
    while input_text != "quit":
        api_client.run(input_text)
        input_text = input("Enter a text (or a \"quit\" to stop):\n")
