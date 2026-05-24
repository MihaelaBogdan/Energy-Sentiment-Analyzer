import os
cfg_path = r'C:\Users\georg\AppData\Local\Programs\Python\Python312\Lib\site-packages\saspy\sascfg_personal.py'
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = f.read()

cfg = cfg.replace("'authkey'  : 'viya',", "")
with open(cfg_path, 'w', encoding='utf-8') as f:
    f.write(cfg)

with open('app.py', 'r', encoding='utf-8') as f:
    app = f.read()

old_mock = '''if "authcode" in msg.lower() or "url" in msg.lower():
                return auth_code'''

new_mock = '''if "default=authcode" in msg.lower():
                return "authcode"
            elif "authcode" in msg.lower() or "url" in msg.lower():
                return auth_code'''

app = app.replace(old_mock, new_mock)
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app)

print('Updated both')
