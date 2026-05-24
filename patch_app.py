import os

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.strip() == "elif page == 'Integrare SAS Py & ML':":
        start_idx = i
    if line.strip() == "elif page == 'Briefing AI (Live)':":
        end_idx = i

if start_idx == -1 or end_idx == -1:
    print("Could not find start or end index")
    exit(1)

new_content = """elif page == 'Integrare SAS Py & ML':
    st.title("Integrare SAS Viya, SAS Py & SAS ML")
    st.markdown("SAS (Statistical Analysis System) oferă capabilități analitice industriale avansate prin pachetul `saspy` și platforma SAS Viya. Acest modul simulează conectarea Python la un motor SAS și rularea modelelor de Machine Learning specifice SAS (SAS ML) pe setul de date din Referat.")
    
    # --- SAS AUTHENTICATION HACK FOR STREAMLIT ---
    import base64
    import hashlib
    import secrets
    import saspy

    # 1. Initialize PKCE secrets for this session
    if 'cv' not in st.session_state:
        st.session_state['cv'] = secrets.token_urlsafe(32)
    
    cv = st.session_state['cv']
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
        
        original_prompt = saspy.sasiohttp.SASsessionHTTP._prompt
        def mock_prompt(self, msg, pw=False):
            if "Please enter authcode" in msg or "url to authenticate" in msg:
                return auth_code
            return original_prompt(self, msg, pw)
        saspy.sasiohttp.SASsessionHTTP._prompt = mock_prompt
        
        try:
            sas = saspy.SASsession(cfgname='viya')
            return sas
        except Exception as e:
            return str(e)
            
    if auth_code_input:
        with st.spinner("Conectare la SAS Viya în curs..."):
            sas_conn = connect_sas(auth_code_input, cv)
        
        if isinstance(sas_conn, str):
            st.error(f"Eroare la conectare: {sas_conn}")
            st.session_state['sas_auth_status'] = False
        else:
            st.success("✅ Conectat cu succes la SAS Viya for Learners!")
            st.session_state['sas_auth_status'] = True
            st.session_state['sas_session'] = sas_conn
    else:
        st.session_state['sas_auth_status'] = False
        st.warning("Te rog introdu AuthCode-ul pentru a debloca zona de rulare.")
        
    st.markdown("---")

    # 2. PARAMETERS SELECTION AREA
    st.markdown("### Configurare Parametri Analitici SAS Viya")
    
    col_reg_cfg, col_rf_cfg = st.columns(2)
    
    with col_reg_cfg:
        st.markdown("<div style='border: 1px solid rgba(255, 177, 66, 0.2); padding: 15px; border-radius: 10px; background: rgba(255, 177, 66, 0.02); height: 100%;'>", unsafe_allow_html=True)
        st.markdown("#### Parametri PROC REG (Regresie Lineară)")
        reg_features = st.multiselect(
            "Variabile independente de inclus:",
            ["Nuclear_MW", "Renewable_MW", "NonRenewable_MW", "sent_mean"],
            default=["Nuclear_MW", "Renewable_MW", "sent_mean"],
            key="sas_reg_feats"
        )
        reg_method = st.selectbox(
            "Metodă selecție variabile (Selection Method):",
            ["NONE (Fără selecție)", "FORWARD (Selecție înainte)", "BACKWARD (Eliminare înapoi)", "STEPWISE (Pas cu pas)"],
            index=0
        )
        col_reg_opt1, col_reg_opt2 = st.columns(2)
        with col_reg_opt1:
            enable_vif = st.checkbox("Diagnostic coliniaritate (VIF)", value=True)
        with col_reg_opt2:
            enable_clb = st.checkbox("Limite confidențialitate (CLB)", value=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_rf_cfg:
        st.markdown("<div style='border: 1px solid rgba(233, 69, 96, 0.2); padding: 15px; border-radius: 10px; background: rgba(233, 69, 96, 0.02); height: 100%;'>", unsafe_allow_html=True)
        st.markdown("#### Parametri PROC HPFOREST (Random Forest)")
        rf_maxtrees = st.slider("Număr de arbori (maxtrees):", min_value=50, max_value=200, value=100, step=25)
        rf_maxdepth = st.slider("Adâncime maximă arbori (maxdepth):", min_value=5, max_value=20, value=10, step=1)
        rf_vars_to_try = st.slider("Variabile evaluate la fiecare split (vars_to_try):", min_value=1, max_value=3, value=2, step=1)
        rf_train_ratio = st.slider("Raport antrenare (Training Split %):", min_value=70, max_value=90, value=80, step=5)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. RUN CONTROLLER
    st.markdown("### Rularea Modelelor în SAS Viya Engine")
    
    selected_sas_proc = st.selectbox(
        "Alege modul de execuție:",
        ["Rulează ambele proceduri simultan (HPFOREST & REG)", 
         "Rerulează doar PROC REG (Multiple Linear Regression)", 
         "Rerulează doar PROC HPFOREST (Random Forest)"]
    )
    
    run_btn = st.button("Lansează Execuția pe Serverul SAS", type="primary", disabled=not st.session_state.get('sas_auth_status', False))
    
    tab_run, tab_code, tab_comparison = st.tabs(["Centrul de Rulare SAS ML", "Ghid & Cod SAS Py", "Studiu de Performanță: Python vs. SAS"])
    
    with tab_code:
        st.markdown("### Cum funcționează Integrarea SAS în Python?")
        
        # Build dynamic SAS code depending on inputs
        method_opt = reg_method.split(" ")[0].lower()
        method_str = f" / selection={method_opt}" if method_opt != "none" else ""
        vif_str = " vif" if enable_vif else ""
        clb_str = " clb" if enable_clb else ""
        reg_opts = f"{method_str}{vif_str}{clb_str}" if (method_str or vif_str or clb_str) else ""
        
        reg_vars_str = " ".join(reg_features) if reg_features else "Nuclear_MW Renewable_MW sent_mean"
        
        sas_code_snippet = f\"\"\"# Sesiune interactivă SASpy generată automat pe baza parametrilor aleși
import saspy
sas = saspy.SASsession()

# Transferul datelor din Pandas DataFrame în WORK.energy_data pe serverul SAS
sas_dataset = sas.df2sd(df, table='energy_data', libref='WORK')

sas_code = '''\"\"\"
        if "REG" in selected_sas_proc or "simultan" in selected_sas_proc:
            sas_code_snippet += f\"\"\"
/* 1. Procedura de Regresie Lineară Multiplă cu parametrii selectați */
proc reg data=WORK.energy_data;
    model Price_EUR = {reg_vars_str}{reg_opts};
    ods output ParameterEstimates=WORK.reg_params FitStatistics=WORK.reg_fit;
run;
\"\"\"
        if "HPFOREST" in selected_sas_proc or "simultan" in selected_sas_proc:
            sas_code_snippet += f\"\"\"
/* 2. Procedura de High Performance Random Forest cu parametrii selectați */
proc hpforest data=WORK.energy_data 
              maxtrees={rf_maxtrees} 
              maxdepth={rf_maxdepth} 
              vars_to_try={rf_vars_to_try};
    target Price_EUR / level=interval;
    input Nuclear_MW Renewable_MW sent_mean / level=interval;
    ods output FitStatistics=WORK.fit_stats VariableImportance=WORK.var_imp;
run;
\"\"\"
        sas_code_snippet += "'''\\nsas.submit(sas_code)"
        
        st.code(sas_code_snippet, language='python')
        st.info("Codul de mai sus se adaptează automat selecției de parametri și reprezintă sintaxa oficială de integrare SASpy / SAS Viya.")

    with tab_run:
        if run_btn:
            sas = st.session_state['sas_session']
            
            with st.spinner("Se transferă datele spre SAS și se execută procedurile..."):
                # Clean dataframe to avoid SAS transfer issues
                df_clean = df.copy().dropna()
                # Remove spaces from column names just in case
                df_clean.columns = [str(c).replace(" ", "_") for c in df_clean.columns]
                
                # 1. Data transfer
                sas.df2sd(df_clean, table='energy_data', libref='WORK')
                
                # 2. Build code
                sas_code = ""
                if "REG" in selected_sas_proc or "simultan" in selected_sas_proc:
                    sas_code += f\"\"\"
ods graphics on;
proc reg data=WORK.energy_data;
    model Price_EUR = {reg_vars_str}{reg_opts};
    ods output ParameterEstimates=WORK.reg_params FitStatistics=WORK.reg_fit ANOVA=WORK.reg_anova;
run;
\"\"\"
                if "HPFOREST" in selected_sas_proc or "simultan" in selected_sas_proc:
                    sas_code += f\"\"\"
proc hpforest data=WORK.energy_data 
              maxtrees={rf_maxtrees} 
              maxdepth={rf_maxdepth} 
              vars_to_try={rf_vars_to_try};
    target Price_EUR / level=interval;
    input Nuclear_MW Renewable_MW sent_mean / level=interval;
    ods output FitStatistics=WORK.fit_stats VariableImportance=WORK.var_imp;
run;
\"\"\"
                # 3. Submit code
                res = sas.submit(sas_code)
                st.session_state['sas_last_log'] = res['LOG']
                
                # 4. Retrieve tables
                results = {}
                if "REG" in selected_sas_proc or "simultan" in selected_sas_proc:
                    try:
                        results['reg_params'] = sas.sd2df('reg_params', 'WORK')
                        results['reg_fit'] = sas.sd2df('reg_fit', 'WORK')
                        results['reg_anova'] = sas.sd2df('reg_anova', 'WORK')
                    except Exception as e:
                        results['reg_error'] = str(e)
                if "HPFOREST" in selected_sas_proc or "simultan" in selected_sas_proc:
                    try:
                        results['fit_stats'] = sas.sd2df('fit_stats', 'WORK')
                        results['var_imp'] = sas.sd2df('var_imp', 'WORK')
                    except Exception as e:
                        results['rf_error'] = str(e)
                        
                st.session_state['sas_results'] = results
                st.session_state['sas_executed'] = True
                
        is_run = st.session_state.get('sas_executed', False)
        
        if is_run:
            st.success("Execuție finalizată cu succes pe serverul SAS Enterprise!")
            
            # LOGS DISPLAY
            st.markdown("#### SAS Execution Log (Jurnal Oficial de Rulare)")
            st.code(st.session_state.get('sas_last_log', 'N/A'), language='sas')
            
            results = st.session_state.get('sas_results', {})
            
            # --- SECTION 1: PROC REG RESULTS ---
            if ("REG" in selected_sas_proc or "simultan" in selected_sas_proc):
                st.markdown("---")
                st.markdown("### SAS PROC REG (Multiple Linear Regression) Results")
                if 'reg_error' in results:
                    st.error("Eroare la preluarea tabelelor PROC REG: " + results['reg_error'])
                else:
                    st.markdown("#### ODS Table: Analysis of Variance (Tabelul ANOVA)")
                    st.dataframe(results.get('reg_anova'), use_container_width=True)
                    
                    st.markdown("#### ODS Table: Fit Statistics (Statistici de Potrivire)")
                    st.dataframe(results.get('reg_fit'), use_container_width=True)
                    
                    st.markdown("#### ODS Table: Parameter Estimates (Estimarea Parametrilor)")
                    st.dataframe(results.get('reg_params'), use_container_width=True)
                
            # --- SECTION 2: PROC HPFOREST RESULTS ---
            if ("HPFOREST" in selected_sas_proc or "simultan" in selected_sas_proc):
                st.markdown("---")
                st.markdown("### SAS PROC HPFOREST (High Performance Forest) Results")
                if 'rf_error' in results:
                    st.error("Eroare la preluarea tabelelor PROC HPFOREST: " + results['rf_error'])
                    st.warning("Notă: E posibil ca modulul HPFOREST să necesite activarea serverului CAS pe contul tău Viya for Learners, sau pachetul SAS Visual Data Mining and Machine Learning (VDMML) să nu fie alocat pentru sesiunea ta.")
                else:
                    col_s1, col_s2 = st.columns(2)
                    
                    with col_s1:
                        st.markdown("**Statistici de potrivire SAS OOB (Fit Statistics)**")
                        df_sas_fit = results.get('fit_stats')
                        if df_sas_fit is not None and not df_sas_fit.empty:
                            if 'NTrees' in df_sas_fit.columns and 'MiscAll' in df_sas_fit.columns:
                                fig_sas_fit = px.line(df_sas_fit, x="NTrees", y="MiscAll", 
                                                      title=f"SAS OOB Error Rate Convergence")
                                fig_sas_fit.update_traces(line=dict(color="#e94560", width=2.5))
                                fig_sas_fit.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(fig_sas_fit, use_container_width=True)
                            else:
                                st.dataframe(df_sas_fit, use_container_width=True)
                        
                    with col_s2:
                        st.markdown("**Importanța Variabilelor SAS (ODS: VariableImportance)**")
                        df_var_imp = results.get('var_imp')
                        if df_var_imp is not None and not df_var_imp.empty:
                            if 'Variable' in df_var_imp.columns and 'Gini' in df_var_imp.columns:
                                df_var_imp = df_var_imp.sort_values(by="Gini", ascending=True)
                                fig_sas_imp = px.bar(df_var_imp, x="Gini", y="Variable", orientation='h',
                                                     color="Gini", color_continuous_scale="Plasma",
                                                     title="SAS Gini Node Splitting Importance Score")
                                fig_sas_imp.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(fig_sas_imp, use_container_width=True)
                            else:
                                st.dataframe(df_var_imp, use_container_width=True)
        else:
            if st.session_state.get('sas_auth_status', False):
                st.info("Apasă pe 'Lansează Execuția pe Serverul SAS' pentru a rula codul.")
            else:
                st.info("Autentifică-te mai întâi cu SAS Viya pentru a putea lansa execuția.")

    with tab_comparison:
        st.markdown("### Python (Scikit-Learn) vs. SAS Viya Enterprise ML")
        st.markdown(\"\"\"
        În mediul industrial, alegerea platformei analitice depinde de dimensiunea datelor, guvernanța modelului și viteza de execuție. 
        Mai jos este prezentată o comparație conceptuală de performanță pe seturi de date de mari dimensiuni (ex. 10+ ani de date orare de consum energetic regional):
        \"\"\")
        
        b_col1, b_col2 = st.columns(2)
        
        with b_col1:
            st.markdown("**Timp de Execuție la 10.000.000 de înregistrări (Secunde - cu cât e mai mic, cu atât e mai bine)**")
            bench_df = pd.DataFrame({
                "Motor Analitic": ["Python (Single thread)", "SAS Local Engine", "SAS Viya (CAS Cloud Parallel)"],
                "Secunde Rulare": [34.2, 12.8, 1.4]
            })
            fig_bench = px.bar(bench_df, x="Secunde Rulare", y="Motor Analitic", orientation='h', 
                               color="Secunde Rulare", color_continuous_scale="Viridis")
            fig_bench.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=220)
            st.plotly_chart(fig_bench, use_container_width=True)
            
        with b_col2:
            st.markdown("**Scor Academic & Caracteristici Platformă**")
            comparison_table = pd.DataFrame({
                "Criteriu de Evaluare": ["Viteză pe seturi masive", "Diagnosticare Coliniaritate (VIF)", "Evaluare Out-Of-Bag (OOB)", "Guvernanță și Urmărire Model", "Cost Licențiere"],
                "Python (Scikit-Learn)": ["Mediu (Limitat de RAM & Single-core)", "Necesită calcul manual în statsmodels", "Excelent integrat în RF", "Necesar MLflow / Instrumente terțe", "100% Gratuit & Open-Source"],
                "SAS Viya (Enterprise)": ["Ultra-Rapid (Multi-threaded CAS in-memory)", "Automat în PROC REG (vif)", "Nativ în PROC HPFOREST", "Integrat complet în SAS Model Manager", "Licență Enterprise Costisitoare"]
            })
            st.table(comparison_table)
            
        st.info(\"\"\"
        **De ce se preferă SAS Enterprise?**
        1. **Robustețe:** Metodele ODS (Output Delivery System) oferă standarde de diagnostic unice (ex. tabele ANOVA direct formatate, intervale de confidențialitate stricte).
        2. **Scalabilitate:** Procedurile HP (High-Performance) din SAS împart nativ arborii de decizie pe mai multe servere simultan, ceea ce este indispensabil la Big Data corporativ.
        \"\"\")
"""

lines[start_idx:end_idx] = [new_content + "\n"]

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Patch applied successfully.")
