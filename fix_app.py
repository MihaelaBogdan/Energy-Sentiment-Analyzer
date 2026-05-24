import os

with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

# 1. Fix SAS proc reg syntax for options
old_opts = """        method_str = f" / selection={method_opt}" if method_opt != "none" else ""
        vif_str = " vif" if enable_vif else ""
        clb_str = " clb" if enable_clb else ""
        reg_opts = f"{method_str}{vif_str}{clb_str}" if (method_str or vif_str or clb_str) else "" """

new_opts = """        method_str = f"selection={method_opt}" if method_opt != "none" else ""
        vif_str = "vif" if enable_vif else ""
        clb_str = "clb" if enable_clb else ""
        opts_list = [opt for opt in [method_str, vif_str, clb_str] if opt]
        reg_opts = " / " + " ".join(opts_list) if opts_list else "" """

if "method_str = f\" / selection={method_opt}\"" in app_code:
    app_code = app_code.replace(old_opts, new_opts)
    print("Fixed SAS options syntax.")
else:
    print("Could not find old_opts block to replace.")

# 2. Fix use_container_width warnings
# Warning: For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use `width='content'`.
# This warning is from st.dataframe in recent Streamlit versions? Or st.plotly_chart?
# The message says `Please replace use_container_width with width.`
app_code = app_code.replace("use_container_width=True", "width=True") # Actually, we will just remove it. Wait, `width=True` is not valid.
# Let's remove `, use_container_width=True`
app_code = app_code.replace(", use_container_width=True", "")
app_code = app_code.replace("use_container_width=True", "")
print("Removed use_container_width to fix warnings.")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print("app.py updated successfully.")
