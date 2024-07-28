import http.client
import json
import os
import load_dotenv

load_dotenv.load_dotenv()
def scrape_github_secrets_guide(api_key, url="https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions"):
    conn = http.client.HTTPSConnection("scrape.serper.dev")
    payload = json.dumps({
        "url": url,
    })
    headers = {
        'X-API-KEY': os.getenv('SERPER_API_KEY'),
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/", payload, headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return data.decode("utf-8")