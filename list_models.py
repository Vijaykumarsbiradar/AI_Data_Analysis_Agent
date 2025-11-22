import requests

# API_KEY = "AIzaSyAUs1b0QQouCQzz00Rwt6Bz75IALgNQ0YM"   # <-- paste your real Gemini API key here

endpoint = "https://generativelanguage.googleapis.com/v1/models"
params = {"key": API_KEY}

r = requests.get(endpoint, params=params, timeout=10)

print("Status:", r.status_code)
print(r.text)
