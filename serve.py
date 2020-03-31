"""
Simple flask server for the interface.
"""

import json
from flask import Flask, request, redirect, url_for
from flask import render_template

# -----------------------------------------------------------------------------

app = Flask(__name__)

# Load raw paper data.
with open('jall.json', 'r') as f:
    jall = json.load(f)

# Load computed paper similarities.
with open('sim_tfidf_svm.json', 'r') as f:
    sim_dict = json.load(f)

# Load search dictionary for each paper.
with open('search.json', 'r') as f:
    search_dict = json.load(f)

# Do some precomputation since we're going to be doing lookups of doi -> doc index.
doi_to_ix = {}
for i, j in enumerate(jall['rels']):
    doi_to_ix[j['rel_doi']] = i


# -----------------------------------------------------------------------------
# Routes.


@app.route("/search", methods=['GET'])
def search():
    q = request.args.get('q', '')   # Get the search request.
    if not q:
        # If someone just hits enter with empty field.
        return redirect(url_for('main'))

    qparts = q.lower().strip().split()  # Split by spaces.

    # Accumulate scores.
    n = len(jall['rels'])
    scores = []
    for i, sd in enumerate(search_dict):
        score = sum(sd.get(q, 0) for q in qparts)
        if score == 0:
            continue  # No match whatsoever, don't include.
        # Give a small boost to more recent papers (low index).
        score += 1.0 * (n - i) / n
        scores.append((score, jall['rels'][i]))

    # Sorting scores in descending order.
    scores.sort(reverse=True, key=lambda x: x[0])
    papers = [x[1] for x in scores if x[0] > 0]
    if len(papers) > 40:
        papers = papers[:40]
    gvars = {'sort_order': 'search', 'search_query': q,
             'num_papers': len(jall['rels'])}
    context = {'papers': papers, 'gvars': gvars}
    return render_template('index.html', **context)


@app.route('/sim/<doi_prefix>/<doi_suffix>')
def sim(doi_prefix=None, doi_suffix=None):
    # Reconstruct the full DOI(Digital Object Identifier).
    doi = f"{doi_prefix}/{doi_suffix}"
    pix = doi_to_ix.get(doi)
    if pix is None:
        papers = []
    else:
        sim_ix = sim_dict[pix]
        papers = [jall['rels'][cix] for cix in sim_ix]
    gvars = {'sort_order': 'sim', 'num_papers': len(jall['rels'])}
    context = {'papers': papers, 'gvars': gvars}
    return render_template('index.html', **context)


@app.route('/')
def main():
    papers = jall['rels'][:40]
    gvars = {'sort_order': 'latest', 'num_papers': len(jall['rels'])}
    context = {'papers': papers, 'gvars': gvars}
    return render_template('index.html', **context)
