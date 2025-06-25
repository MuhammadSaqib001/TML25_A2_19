import requests

### REQUESTING NEW API ###
TOKEN = "54614611" # to be changed according to your token (given to you for the assignments via email)

response = requests.get("http://34.122.51.94:9090" + "/stealing_launch", headers={"token": TOKEN})
answer = response.json()

print(answer)  # {"seed": "SEED", "port": PORT}