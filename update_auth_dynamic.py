import re

with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

# 1. Replace the URL definition
old_url_code = r'purl = f"https://vfl-053.engage.sas.com/SASLogon/oauth/authorize\?client_id=SASPy&response_type=code&code_challenge_method=S256&code_challenge=\{cc\}"'
new_url_code = '''server_url_input = st.text_input("Pasul 1: Introdu URL-ul serverului tău SAS Viya:", value="https://vfl-053.engage.sas.com")
    server_url_input = server_url_input.rstrip('/')
    purl = f"{server_url_input}/SASLogon/oauth/authorize?client_id=SASPy&response_type=code&code_challenge_method=S256&code_challenge={cc}"'''
app_code = re.sub(old_url_code, new_url_code, app_code)

# 2. Update connect_sas
old_def = r'def connect_sas\(auth_code, _cv\):'
new_def = '''def connect_sas(auth_code, _cv, server_url):
        import os, tempfile
        cfg_content = f"""
SAS_config_names=['viya']
viya = {{'url': '{server_url}', 'context': 'SAS Studio compute context', 'authkey': 'saspy_viya', 'client_id': 'SASPy'}}
"""
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
        tf.write(cfg_content)
        tf.close()'''
app_code = re.sub(old_def, new_def, app_code)

# 3. Update saspy call
old_sas_call = r'sas = saspy\.SASsession\(cfgname=\'viya\'\)'
new_sas_call = '''sas = saspy.SASsession(cfgname='viya', cfgfile=tf.name)
            os.remove(tf.name)'''
app_code = re.sub(old_sas_call, new_sas_call, app_code)

# 4. Update the connect_sas call
old_connect_call = r'sas_conn = connect_sas\(auth_code_input, cv\)'
new_connect_call = r'sas_conn = connect_sas(auth_code_input, cv, server_url_input)'
app_code = re.sub(old_connect_call, new_connect_call, app_code)

# 5. Add server_url_input check
old_if = r'if auth_code_input:'
new_if = r'if auth_code_input and server_url_input:'
app_code = re.sub(old_if, new_if, app_code)

# 6. Change text
app_code = app_code.replace('**Pasul A:**', '**Pasul 2:**')
app_code = app_code.replace('Pasul B: Lipește AuthCode-ul generat mai sus:', 'Pasul 3: Lipește AuthCode-ul generat la Pasul 2:', 1)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print("Actualizare finalizata!")
