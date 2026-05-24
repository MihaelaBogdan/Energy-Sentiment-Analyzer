with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

inject_log = '''
            st.markdown("### SAS Log")
            st.code(st.session_state.get('sas_last_log', 'No log found'), language='sas')
            
            # --- SECTION 1: PROC REG RESULTS ---
'''

app_code = app_code.replace("# --- SECTION 1: PROC REG RESULTS ---", inject_log)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)
print('Updated app.py to show SAS log unconditionally.')
