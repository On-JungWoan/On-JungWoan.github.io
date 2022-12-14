from oauth2client.service_account import ServiceAccountCredentials
import httplib2
import json
import os

URL = "https://on-jungwoan.github.io/dl/cs231n_52/"
TYPE = "URL_UPDATED"


SCOPES = [ "https://www.googleapis.com/auth/indexing" ]
ENDPOINT = "https://indexing.googleapis.com/v3/urlNotifications:publish"

# service_account_file.json is the private key that you created for your service account.
JSON_KEY_FILE = {
  "type": "service_account",
  "project_id": "jekyll-blog-371100",
  "private_key_id": os.getenv("PRIVATE_KEY_ID"),
  "private_key": os.getenv("PRIVATE_KEY"),
  "client_email": os.getenv("CLIENT_EMAIL"),
  "client_id": os.getenv("CLIENT_ID"),
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL")
}


credentials = ServiceAccountCredentials.from_json_keyfile_dict(JSON_KEY_FILE, scopes=SCOPES)

http = credentials.authorize(httplib2.Http())

# Define contents here as a JSON string.
# This example shows a simple update request.
# Other types of requests are described in the next step.

content = """{
  \"url\": \"""" + URL + """\",
  \"type\": \"""" + TYPE + """\"
}"""

response, content = http.request(ENDPOINT, method="POST", body=content)
content_dict = json.loads( content.decode('utf-8') )

if response['status'] != '200':
  print(response['status'], content_dict['error']['message'], sep="\n")

print()
