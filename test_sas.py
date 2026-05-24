import saspy

# create a temporary sascfg
temp_cfg = """
SAS_config_names = ['viya_test']
viya_test = {
    'url'      : 'https://vfl-053.engage.sas.com',
    'context'  : 'SAS Studio compute context',
    'options'  : ['authcode'],
    'pkce'     : True
}
"""
with open('test_cfg.py', 'w') as f:
    f.write(temp_cfg)

import test_cfg

saspy.SAScfg = test_cfg

import urllib.request
import http.client

old_request_http = http.client.HTTPSConnection.request
def mock_request_http(self, method, url, body=None, headers={}, *args, **kwargs):
    print('MOCK HTTP REQUEST URL:', url)
    print('MOCK HTTP REQUEST DATA:', body)
    print('MOCK HTTP REQUEST HEADERS:', headers)
    # Return fake response object
    class FakeResponse:
        status = 200
        def read(self):
            return b'{"access_token":"FAKE_TOKEN","refresh_token":"FAKE_REFRESH"}'
    return FakeResponse()

# http.client.HTTPSConnection.request = mock_request_http
# Wait, I don't want to actually connect, so I'll patch the whole HTTPConn
class MockHTTPConn:
    def connect(self):
        pass
    def request(self, method, url, body=None, headers={}):
        print('MOCK HTTP REQUEST URL:', url)
        print('MOCK HTTP REQUEST DATA:', body)
        print('MOCK HTTP REQUEST HEADERS:', headers)
    def getresponse(self):
        class FakeResponse:
            status = 200
            def read(self):
                return b'{"access_token":"FAKE_TOKEN","refresh_token":"FAKE_REFRESH"}'
        return FakeResponse()
    def close(self):
        pass

original_prompt = saspy.sasbase.SASconfig._prompt
def mock_prompt(self, msg, pw=False):
    print(f"MOCK PROMPT CALLED WITH MSG: {msg}")
    return "fake_auth_code_123"

saspy.sasbase.SASconfig._prompt = mock_prompt

import secrets
original_token_urlsafe = secrets.token_urlsafe
def mock_token_urlsafe(nbytes=None):
    if nbytes == 32:
        return 'fake_cv_fake_cv_fake_cv_fake_cv_f'
    return original_token_urlsafe(nbytes)
secrets.token_urlsafe = mock_token_urlsafe

try:
    sas = saspy.SASsession(cfgname='viya_test', cfgfile='test_cfg.py')
    sas._io.HTTPConn = MockHTTPConn()
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {str(e)}")
