import sys

with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

old_auth_code = """    cv = st.session_state['cv']
    cvh = hashlib.sha256(cv.encode('utf-8')).digest()
    cc = base64.urlsafe_b64encode(cvh).decode('utf-8').rstrip('=')
    purl = f"https://vfl-053.engage.sas.com/SASLogon/oauth/authorize?client_id=SASPy&response_type=code&code_challenge_method=S256&code_challenge={cc}"

    st.markdown("### 1. Autentificare SAS Viya for Learners")
    st.info("Pentru a rula cod SAS real pe serverele educaționale Viya, ai nevoie de un cod de autorizare (AuthCode).")

    st.markdown(f"**Pasul A:** [Apasă aici pentru a genera codul de autorizare]({purl}) (Se va deschide într-o pagină nouă)")

    auth_code_input = st.text_input("Pasul B: Lipește AuthCode-ul generat mai sus:", type="password")

    @st.cache_resource(show_spinner=False)
    def connect_sas(auth_code, _cv):
        # Mocking secrets and prompt to intercept saspy's terminal flow
        original_token_urlsafe = secrets.token_urlsafe
        def mock_token_urlsafe(nbytes=None):
            if nbytes == 32:
                return _cv
            return original_token_urlsafe(nbytes)
        secrets.token_urlsafe = mock_token_urlsafe

        original_prompt = saspy.sasbase.SASconfig._prompt
        def mock_prompt(self, msg, pw=False):
            if "default=authcode" in msg.lower():
                return "authcode"
            elif "authcode" in msg.lower() or "url" in msg.lower():
                return auth_code
            return original_prompt(self, msg, pw)
        saspy.sasbase.SASconfig._prompt = mock_prompt

        try:
            sas = saspy.SASsession(cfgname='viya')
            return sas
        except Exception as e:
            return str(e)

    if auth_code_input:
        with st.spinner("Conectare la SAS Viya în curs..."):
            sas_conn = connect_sas(auth_code_input, cv)"""

new_auth_code = """    cv = st.session_state['cv']
    cvh = hashlib.sha256(cv.encode('utf-8')).digest()
    cc = base64.urlsafe_b64encode(cvh).decode('utf-8').rstrip('=')

    st.markdown("### 1. Autentificare SAS Viya for Learners")
    st.info("Pentru a rula cod SAS real pe serverele educaționale Viya, ai nevoie de un cod de autorizare (AuthCode).")
    
    server_url_input = st.text_input("Pasul 1: Introdu URL-ul serverului tău SAS Viya:", value="https://vfl-053.engage.sas.com")
    
    # Strip trailing slash just in case
    server_url_input = server_url_input.rstrip('/')
    
    purl = f"{server_url_input}/SASLogon/oauth/authorize?client_id=SASPy&response_type=code&code_challenge_method=S256&code_challenge={cc}"

    st.markdown(f"**Pasul 2:** [Apasă aici pentru a genera codul de autorizare]({purl}) (Se va deschide într-o pagină nouă)")

    auth_code_input = st.text_input("Pasul 3: Lipește AuthCode-ul generat mai sus:", type="password")

    @st.cache_resource(show_spinner=False)
    def connect_sas(auth_code, _cv, server_url):
        # Create dynamic sascfg based on server_url
        import os, tempfile
        cfg_content = f\"\"\"
SAS_config_names=['viya']
viya = {{'url': '{server_url}', 'context': 'SAS Studio compute context', 'authkey': 'saspy_viya', 'client_id': 'SASPy'}}
\"\"\"
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
        tf.write(cfg_content)
        tf.close()

        # Mocking secrets and prompt to intercept saspy's terminal flow
        original_token_urlsafe = secrets.token_urlsafe
        def mock_token_urlsafe(nbytes=None):
            if nbytes == 32:
                return _cv
            return original_token_urlsafe(nbytes)
        secrets.token_urlsafe = mock_token_urlsafe

        original_prompt = saspy.sasbase.SASconfig._prompt
        def mock_prompt(self, msg, pw=False):
            if "default=authcode" in msg.lower():
                return "authcode"
            elif "authcode" in msg.lower() or "url" in msg.lower():
                return auth_code
            return original_prompt(self, msg, pw)
        saspy.sasbase.SASconfig._prompt = mock_prompt

        try:
            sas = saspy.SASsession(cfgname='viya', cfgfile=tf.name)
            return sas
        except Exception as e:
            return str(e)
        finally:
            try:
                os.remove(tf.name)
            except:
                pass

    if auth_code_input and server_url_input:
        with st.spinner("Conectare la SAS Viya în curs..."):
            sas_conn = connect_sas(auth_code_input, cv, server_url_input)"""

if old_auth_code in app_code:
    app_code = app_code.replace(old_auth_code, new_auth_code)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(app_code)
    print("Rezolvat!")
else:
    print("Codul vechi nu s-a potrivit exact. Verifica app.py!")
