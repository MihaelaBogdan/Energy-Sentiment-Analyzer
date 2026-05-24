# EnergyPulse RO

Interactive web application for analyzing and predicting electricity spot prices in Romania, developed at the Faculty of Cybernetics, Statistics and Economic Informatics (ASE Bucharest).

🔗 **Live app:** [https://energy-pulse.streamlit.app/](https://energy-pulse.streamlit.app/)

## About the project

The core idea was to test whether we can better predict electricity prices by combining classical data (nuclear output, renewables, etc.) with news from the press. We built a pipeline that collects historical data, extracts sentiment from the Economica.net RSS feed using a lexicon-based NLP method, and feeds everything into a Machine Learning model (Random Forest).

For statistical rigor, we also connected the Python application to a cloud SAS Viya for Learners environment.

### Features

- **Live Dashboard:** Displays the current price, short-term prediction and a live sentiment score.
- **What-If Simulator:** The most interactive part. Use sliders to manually adjust production levels or sentiment — the model recalculates the predicted price instantly.
- **Correlation Analysis:** Plotly charts showing clear relationships in the data — for example, how higher renewable output pulls prices down.
- **SAS Viya Integration:** The app connects via `saspy` to a SAS server, sends a pandas dataframe, runs statistical procedures (`PROC REG`, `PROC HPFOREST`) remotely and brings the results back into the interface.

## Running the app locally

If you prefer to run it on your own machine instead of using the live link, you need Python installed. Steps are straightforward:

1. Download or clone the repository.
2. Navigate to the project folder in your terminal.
3. Install the required libraries:
```bash
   pip install -r requirements.txt
```
4. Start the Streamlit server:
```bash
   streamlit run app.py
```

**Note on SAS Viya connection:** To test the SAS analysis features, you will need an active SAS Viya for Learners account. When you open the SAS section in the app, you will be prompted to enter your server URL and generate an AuthCode.

## Authors

* Bogdan Mihaela
* Georgescu Leonard-Dimitrie

*(Students in Business Computer Science, CSIE, ASE Bucharest)*
