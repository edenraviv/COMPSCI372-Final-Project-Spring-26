import requests

def send_get_request(self, url, api_key):
    params = {'api_dev_key':api_key,
            'api_option':'paste',
            'api_paste_format':'python'}
    response = requests.get(url, params)
    return response.json()

print("testing git username config")