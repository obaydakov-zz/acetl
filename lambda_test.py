import requests
import pprint

import pandas as pd

import jsonpatch

import json

import time



headers = {
  
    'Content-Type': 'application/json',
}

k=10
total=0
for l in range(10):
    start_time = int(round(time.time()*1000))
    response = requests.get('https://prbs9bz5k6.execute-api.eu-central-1.amazonaws.com/default/test_janitor', headers=headers)
    elapsed = (int(round(time.time()*1000)) - start_time)/1000
    print(response)
    if l>4:
        total+=elapsed
        

print('Time '+str(total/5))
#print(response.text) 
#* S3 - 1.5 sec
#* S3 + inital column lists - 7 sec

# curl -i -X GET 'https://prbs9bz5k6.execute-api.eu-central-1.amazonaws.com/default/test_janitor'