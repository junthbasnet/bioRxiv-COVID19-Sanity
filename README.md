 <img src="./static/favicon.png" border="10">
   <p align="center">
   <h1 align="center">bioRxiv medRxiv COVID19-Sanity</h1>
</p>

<p align="center">
  <img src="./static/bioRxiv_COVID-19.png" border="10">
</p>
<p align="center">
  <strong>Junth Basnet</strong>
</p>

---

This project organizes `COVID-19 SARS-CoV-2 preprints from medRxiv and bioRxiv`. This project makes data searchable, sortable, etc. The most similar search uses **Exemplar SVM** :smiley: trained on `TF-IDF feature vectors` from the `abstract, title and author` of these papers.

<p align="center">
  <img src="./assets/bioRxiv-medRxiv.png" border="10">                                <img src="https://imgur.com/zIWk2E5.png" border="10" width="288" height="162">
</p>

Raw data is available at [bioRxiv and medRxiv Page.](https://connect.biorxiv.org/relate/collection_json.php?grp=181)

<p align="center">
  <a href="https://connect.biorxiv.org/relate/collection_json.php?grp=181" target="_blank"><img src="https://imgur.com/rpe0MaJ.png"
alt="Raw Data from bioRxiv and medRxiv Page." border="10" /></a>
</p>
  
# Installation Dependencies  
```
Python 3.8.1
Flask 1.1.1
requests 2.23.0
numpy 1.18.2
sklearn
```

# How to Run (Development Server)?
```js
git clone https://github.com/Junth/bioRxiv-COVID19-Sanity.git
```

```js
cd bioRxiv-COVID19-Sanity
```

```js
pip install -r requirements.txt
```
<p align="center">
<img src="./assets/Step4.PNG" border="10"></p>

```js
python run.py
export FLASK_APP=serve.py
flask run
```
<p align="center">
<img src="./assets/Step5.PNG" border="10"></p>

# Results

**Latest Papers:** :smiley:
<p align="center">
<img src="./assets/Screenshot1.PNG" border="10"></p>

**Search results:** :heart_eyes:
<p align="center">
<img src="./assets/Screenshot2.PNG" border="10"></p>

**Similar results:** :open_mouth:
<p align="center">
<img src="./assets/bioRxiv_COVID-19_Sanity_Similarity.gif" border="10"></p>

# Liscense
MIT
