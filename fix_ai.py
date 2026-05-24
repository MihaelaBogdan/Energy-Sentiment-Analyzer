import sys

with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

old_code = """        elif "scumpă" in q_low or "scumpa" in q_low or "pret" in q_low or "preț" in q_low:
            # Find most expensive country in current API list
            if not df_eu.empty:
                max_row = df_eu.loc[df_eu['Preț (EUR/MWh)'].idxmax()]
                response_text = f\"\"\"
                Pe baza datelor prelevate în direct prin API-ul ENTSO-E/Energy-Charts, cea mai scumpă țară analizată în acest moment este **{max_row['Țară']}**, având un preț spot de **{max_row['Preț (EUR/MWh)']:.2f} EUR/MWh**.
                Prin comparație, în România prețul spot actual este de **{current_price:.1f} EUR/MWh**, ceea ce ne plasează pe o poziție {'competitivă' if current_price < max_row['Preț (EUR/MWh)'] else 'ridicată'} regional.
                \"\"\"
            else:
                response_text = "Nu am putut prelua datele live ale prețurilor din Europa în acest moment pentru a stabili clasamentul exact. Vă rugăm să reîncărcați pagina.\""""

new_code = """        elif "scumpă" in q_low or "scumpa" in q_low or "pret" in q_low or "preț" in q_low:
            # Find most expensive country in current API list
            valid_prices = {k: v for k, v in real_eu_prices.items() if v is not None}
            if valid_prices:
                max_country = max(valid_prices, key=valid_prices.get)
                max_price = valid_prices[max_country]
                response_text = f\"\"\"
                Pe baza datelor prelevate în direct prin API-ul ENTSO-E/Energy-Charts, cea mai scumpă țară analizată în acest moment este **{max_country}**, având un preț spot de **{max_price:.2f} EUR/MWh**.
                Prin comparație, în România prețul spot actual este de **{current_price:.1f} EUR/MWh**, ceea ce ne plasează pe o poziție {'competitivă' if current_price < max_price else 'ridicată'} regional.
                \"\"\"
            else:
                response_text = "Nu am putut prelua datele live ale prețurilor din Europa în acest moment pentru a stabili clasamentul exact. Vă rugăm să reîncărcați pagina.\""""

if old_code in app_code:
    app_code = app_code.replace(old_code, new_code)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(app_code)
    print('Rezolvat!')
else:
    print('Codul vechi nu a fost gasit. Cautam partial...')
    # Alternative fix just replacing df_eu.empty
    if "if not df_eu.empty:" in app_code and "max_row = df_eu.loc" in app_code:
        app_code = app_code.replace("if not df_eu.empty:", "valid_prices = {k: v for k, v in real_eu_prices.items() if v is not None}\n            if valid_prices:")
        app_code = app_code.replace("max_row = df_eu.loc[df_eu['Preț (EUR/MWh)'].idxmax()]", "max_country = max(valid_prices, key=valid_prices.get)\n                max_price = valid_prices[max_country]")
        app_code = app_code.replace("{max_row['Țară']}", "{max_country}")
        app_code = app_code.replace("{max_row['Preț (EUR/MWh)']:.2f}", "{max_price:.2f}")
        app_code = app_code.replace("max_row['Preț (EUR/MWh)']", "max_price")
        with open('app.py', 'w', encoding='utf-8') as f:
            f.write(app_code)
        print('Rezolvat partial!')
