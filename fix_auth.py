import os

# 1. Update sascfg_personal.py
cfg_path = r'C:\Users\georg\AppData\Local\Programs\Python\Python312\Lib\site-packages\saspy\sascfg_personal.py'
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = f.read()

new_cfg = """SAS_config_names = ['viya']

viya = {
    'url'      : 'https://vfl-053.engage.sas.com',
    'authkey'  : 'viya',
    'context'  : 'SAS Studio compute context',
    'options'  : ['authcode'],
    'pkce'     : True
}
"""

with open(cfg_path, 'w', encoding='utf-8') as f:
    f.write(new_cfg)

# 2. Update app.py mock_prompt
with open('app.py', 'r', encoding='utf-8') as f:
    app_content = f.read()

app_content = app_content.replace(
    'if "Please enter authcode" in msg or "url to authenticate" in msg:',
    'if "authcode" in msg.lower() or "url" in msg.lower():'
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_content)

print("Both files updated successfully.")
