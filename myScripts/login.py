import credentials as crs
from fyers_apiv3 import fyersModel
import hashlib

def generate_sha256_hash(client_id, secret_key):
    combined = f"{client_id}{secret_key}"
    return hashlib.sha256(combined.encode()).hexdigest()

hashed_key = generate_sha256_hash(crs.client_id, crs.secret_key)

session = fyersModel.SessionModel(
    client_id = crs.client_id,
    secret_key = hashed_key,
    redirect_uri =crs.redirect_url,
    response_type =crs.response_type,
    grant_type = 'authorization_code'
)

response = session.generate_authcode()
print(response)

response_link = input('Enter The Response Url')

if response_link == 'skip':
    exit()
auth_code = response_link[response_link.index('auth_code')+10:response_link.index('&state')]
print(auth_code)

session = fyersModel.SessionModel(
    client_id = crs.client_id,
    secret_key = crs.secret_key,
    redirect_uri =crs.redirect_url,
    response_type =crs.response_type,
    grant_type = 'authorization_code'
)

session.set_token(auth_code)
response = session.generate_token()

access_token = response["access_token"]
with open('bufferFiles/access_token.txt','w') as file:
    file.write(access_token)


#Success!!

