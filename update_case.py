with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

app_code = app_code.replace("sas.sd2df('reg_params', 'WORK')", "sas.sd2df('REG_PARAMS', 'WORK')")
app_code = app_code.replace("sas.sd2df('reg_fit', 'WORK')", "sas.sd2df('REG_FIT', 'WORK')")
app_code = app_code.replace("sas.sd2df('reg_anova', 'WORK')", "sas.sd2df('REG_ANOVA', 'WORK')")
app_code = app_code.replace("sas.sd2df('fit_stats', 'WORK')", "sas.sd2df('FIT_STATS', 'WORK')")
app_code = app_code.replace("sas.sd2df('var_imp', 'WORK')", "sas.sd2df('VAR_IMP', 'WORK')")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)
print('Updated app.py to use uppercase dataset names.')
