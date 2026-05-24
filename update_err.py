with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

old_err = "st.error(\"Eroare la preluarea tabelelor PROC REG: \" + results['reg_error'])"
new_err = "st.error(\"Eroare la preluarea tabelelor PROC REG: \" + results['reg_error'] + \"\\nLOG SAS:\\n\" + str(st.session_state.get('sas_last_log', 'Fara LOG')))"
app_code = app_code.replace(old_err, new_err)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)
print('Updated app.py to show SAS log in error.')
