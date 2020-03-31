"""
Run to update all the database json files that can be served from the website.
"""

import json
import requests
import numpy as np

# -----------------------------------------------------------------------------


def write_json(obj, filename, msg=''):
    suffix = f'; {msg}' if msg else ''
    print(f"writing {filename}{suffix}.")
    with open(filename, 'w') as f:
        json.dump(obj, f)


def calculate_tfidf_features(rels, max_features=5000, max_df=1.0, min_df=3):
    """ Compute tfidf features with scikit-learn: Convert a collection of raw documents to a matrix of TF-IDF features. """

    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(input='content',
                        encoding='utf-8', decode_error='replace', strip_accents='unicode',
                        lowercase=True, analyzer='word', stop_words='english',
                        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_-]+\b',
                        ngram_range=(1, 1), max_features=max_features,
                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                        max_df=max_df, min_df=min_df)
    corpus = [(a['rel_title'] + '. ' + a['rel_abs']) for a in rels]
    X = v.fit_transform(corpus)
    X = np.asarray(X.astype(np.float32).todense())
    print("TF-IDF calculated array of shape: ", X.shape, ".")
    return X, v


def calculate_sim_dot_product(X, ntake=40):
    """ Take X (N, D) features and for each index return closest ntake indices via dot product. """
    S = np.dot(X, X.T)
    # Take last ntake sorted backwards.
    IX = np.argsort(S, axis=1)[:, :-ntake-1:-1]
    return IX.tolist()


def calculate_sim_svm(X, ntake=40):
    """ Take X (N,D) features and for each index return closest ntake indices using Exemplar SVM. """
    from sklearn import svm
    n, d = X.shape
    IX = np.zeros((n, ntake), dtype=np.int64)
    print(f"Training {n} svms for each paper...")
    for i in range(n):
        # Set all examples as negative except this one.
        y = np.zeros(X.shape[0], dtype=np.float32)
        y[i] = 1
        # Train SVM.
        clf = svm.LinearSVC(class_weight='balanced',
                            verbose=False, max_iter=10000, tol=1e-4, C=0.1)
        clf.fit(X, y)
        # Predict confidence scores for samples.
        s = clf.decision_function(X)
        # Take last ntake sorted backwards.
        ix = np.argsort(s)[:-ntake-1:-1]
        IX[i] = ix
    return IX.tolist()


def build_search_index(rels, v):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    # Construct a reverse index for suppoorting search.
    vocab = v.vocabulary_
    idf = v.idf_
    # Removed hyphen from string punctuation.
    punc = "'!\"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'"
    trans_table = {ord(c): None for c in punc}

    def makedict(s, forceidf=None):
        words = set(s.lower().translate(trans_table).strip().split())
        words = set(w for w in words if len(w) >
                    1 and (not w in ENGLISH_STOP_WORDS))
        idfd = {}
        # TODO: If we're using bigrams in vocab then this won't search over them.
        for w in words:
            if forceidf is None:
                if w in vocab:
                    idfval = idf[vocab[w]]  # we have a computed idf for this.
                else:
                    # Some words we don't know; assume idf 1.0 (low).
                    idfval = 1.0
            else:
                idfval = forceidf
            idfd[w] = idfval
        return idfd

    def merge_dicts(dlist):
        m = {}
        for d in dlist:
            for k, v in d.items():
                m[k] = m.get(k, 0) + v
        return m

    search_dict = []
    for p in rels:
        dict_title = makedict(p['rel_title'], forceidf=10)
        dict_authors = makedict(p['rel_authors'], forceidf=5)
        dict_summary = makedict(p['rel_abs'])
        qdict = merge_dicts([dict_title, dict_authors, dict_summary])
        search_dict.append(qdict)

    return search_dict


if __name__ == '__main__':

    # Fetch the raw data from bioRxiv.
    jstr = requests.get(
        'https://connect.biorxiv.org/relate/collection_json.php?grp=181')
    jall = jstr.json()
    write_json(jall, 'jall.json', f"{len(jall['rels'])} papers")

    # Calculate similarities using various techniques.
    X, v = calculate_tfidf_features(jall['rels'])

    # Similarity using simple dot product on tf-idf.
    sim_tfidf = calculate_sim_dot_product(X)
    write_json(sim_tfidf, 'sim_tfidf_dot.json')

    # Similarity using an Exemplar SVM on tf-idf.
    sim_svm = calculate_sim_svm(X)
    write_json(sim_svm, 'sim_tfidf_svm.json')

    # Calculate the search index to support search.
    search_dict = build_search_index(jall['rels'], v)
    write_json(search_dict, 'search.json')
