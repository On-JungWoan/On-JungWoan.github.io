from oauth2client.service_account import ServiceAccountCredentials
import httplib2
import json

SCOPES = [ "https://www.googleapis.com/auth/indexing" ]
ENDPOINT = "https://indexing.googleapis.com/v3/urlNotifications:publish"

# service_account_file.json is the private key that you created for your service account.
JSON_KEY_FILE = "C:/Users/USER/Downloads/jekyll-blog-371100-5977418fcddb.json"

URL = "https://on-jungwoan.github.io/dl/cs231n_6/"

credentials = ServiceAccountCredentials.from_json_keyfile_name(JSON_KEY_FILE, scopes=SCOPES)

http = credentials.authorize(httplib2.Http())

# Define contents here as a JSON string.
# This example shows a simple update request.
# Other types of requests are described in the next step.

content = """{
  \"url\": \"""" + URL + """\",
  \"type\": \"URL_UPDATED\"
}"""

response, content = http.request(ENDPOINT, method="POST", body=content)
content_dict = json.loads( content.decode('utf-8') )

print()