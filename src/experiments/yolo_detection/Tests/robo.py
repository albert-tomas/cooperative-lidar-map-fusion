import requests

api_key = "m3mVIcAJXOFxaob0eVAt"
url = "https://api.roboflow.com/workspaces"
headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)
print(response.json())