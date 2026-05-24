import re

with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

# Fix the dict.get() error
app_code = app_code.replace("results.get('reg_anova', use_container_width=True)", "results.get('reg_anova')")
app_code = app_code.replace("results.get('reg_fit', use_container_width=True)", "results.get('reg_fit')")
app_code = app_code.replace("results.get('reg_params', use_container_width=True)", "results.get('reg_params')")
app_code = app_code.replace("results.get('fit_stats', use_container_width=True)", "results.get('fit_stats')")
app_code = app_code.replace("results.get('var_imp', use_container_width=True)", "results.get('var_imp')")

# And remove width=True, replace with use_container_width=True
app_code = app_code.replace(", width=True, use_container_width=True", ", use_container_width=True")
app_code = app_code.replace(", width=True", ", use_container_width=True")

# Check for any other messed up get()
app_code = re.sub(r"\.get\('([^']+)', use_container_width=True\)", r".get('\1')", app_code)

# Check for empty dataframe calls that I might have broken
# like st.dataframe(..., use_container_width=True)
app_code = app_code.replace("st.dataframe(results.get('reg_anova'), use_container_width=True)", "st.dataframe(results.get('reg_anova'), use_container_width=True)")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print('Fixed dataframe syntax errors.')
