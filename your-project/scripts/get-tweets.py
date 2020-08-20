"""
    Generate tweets IDs
    
    @see configuration
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Mar Cánovas <mariamar.canovasg@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import pandas as pd
import random
import requests
import sys
import zipfile
import netrc
import os.path
import configuration

from pathlib import Path
from requests.auth import AuthBase


"""
   PLNAuth
   
   We use a custom authentication because the default authentication
   was adding Basic authentication and do not allowed to manually 
   introcuded the token
   
   @link https://requests.readthedocs.io/en/master/user/advanced/#custom-authentication
"""
class PLNAuth (AuthBase):
    def __init__(self, username):
        self.username = username

    def __call__(self, r):
        r.headers['Authorization'] = self.username
        return r


# Read from the .netrc file in your home directory
secrets = netrc.netrc ()
email, account, password = secrets.authenticators ('collaborativehealth.inf.um.es')


# @var umutextstats_api_endpoint String
umutextstats_api_endpoint = 'https://collaborativehealth.inf.um.es/umutextstats/api/'


# @var certificate String
certificate = str (Path.home ()) + '/certificates/CA.pem'
print (certificate)


# @var reponse Response
print ("Loading credentials...")
response = requests.post (
    umutextstats_api_endpoint + 'login', 
    json={'email': email, 'password': password}, 
    verify=certificate
)


# Transform to JSON
response = response.json ()


# @var auth_token String
auth_token = str (response['data']['token'])
print ("Credentials acquired...")
print (auth_token)


# @var umucorpus_ids int|string The Corpus IDs
for key, corpus_ids in configuration.ids['misogyny'].items ():

    # @var zip_filename String 
    filename = './../tweets/' + key + '.csv'
    
    
    # Show log
    print ("Processing " + filename + " ...")
    
    
    # If file exists, skipe it
    """
    if os.path.isfile (filename):
        print ("Skip file...")
        continue
    """
    
    
    # @var request_payload Dictionary Prepare the request according to the features we want to retrieve
    request_payload = {
        'max': configuration.max,
        'umucorpus': ','.join (str(x) for x in corpus_ids),
        'class-strategy': configuration.strategy['classic'],
        'balance-strategy': 'no-balance',
    }
    
    
    # @var reponse Response
    response = requests.post (
        umutextstats_api_endpoint + 'tweets.csv', 
        json=request_payload, 
        verify=certificate,
        auth=PLNAuth (auth_token)
    )
    
    
    # Check response
    if (response.status_code != 200):
        print ("Authentication failed: " + str (response.status_code))
        print (response.text)
        sys.exit ()
        

    with open(filename, 'w+') as f:
        print (response.text, file=f)
