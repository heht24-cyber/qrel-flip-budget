"""Minimum additional binary qrels that reverse mean nDCG@k in a fixed pool."""
import argparse
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import numpy as np


def weights(run, k):
    if len(set(run)) != len(run):
        raise ValueError("duplicate document in a ranking")
    return {d: 1 / math.log2(i + 2) for i, d in enumerate(run[:k])}


def options(a, b, labels, k):
    """Minimum A-minus-B margin for each exact number of added positives."""
    wa, wb = weights(a, k), weights(b, k)
    pool = set(wa) | set(wb)
    coeff = {d: wa.get(d, 0.) - wb.get(d, 0.) for d in pool}
    known = sum(v > 0 for v in labels.values())
    numerator = math.fsum(v for d, v in coeff.items() if labels.get(d, 0) > 0)
    order = sorted(pool - labels.keys(), key=lambda d: (coeff[d], d))
    ideal = np.cumsum([1 / math.log2(i+2) for i in range(k)])
    values = []
    for i in range(len(order) + 1):
        r = min(k, known + i)
        values.append(numerator / ideal[r-1] if r else 0.)
        if i < len(order):
            numerator += coeff[order[i]]
    return np.asarray(values), order


def frontier(curves, budget):
    """Min-plus convolution with backpointers, one curve per query."""
    dp = np.full(budget + 1, np.inf)
    dp[0] = 0
    history = []
    for values in curves:
        new = np.full_like(dp, np.inf)
        took = np.zeros(budget + 1, dtype=int)
        for n, v in enumerate(values[:budget+1]):
            trial = dp[:budget+1-n] + v
            improve = trial < new[n:]
            new[n:][improve] = trial[improve]
            took[n:][improve] = n
        history.append(took)
        dp = new
    return dp, history


def direct(run, positive, k):
    discounts = list(weights(run, k).values())
    ideal = math.fsum(1/math.log2(i+2) for i in range(min(k, len(positive))))
    return sum(v for d, v in zip(run, discounts) if d in positive) / ideal if ideal else 0.


def demo(folder, k):
    import duckdb
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = duckdb.read_parquet(str(folder / "corpus.parquet")).fetchall()
    queries = {str(i): text for i, _, text in duckdb.read_parquet(str(folder / "queries.parquet")).fetchall()}
    labels = {}
    with (folder / "test.tsv").open() as f:
        for q, d, v in csv.reader(list(f)[1:], delimiter="\t"):
            labels.setdefault(q, {})[d] = int(v)
    qids = sorted(labels)
    docs = sorted((str(i), (title or "") + " " + (text or "")) for i, title, text in docs)
    ids, texts = zip(*docs)
    models = []
    for analyzer, grams in [("word", (1, 1)), ("char_wb", (3, 5))]:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=grams, sublinear_tf=True)
        corpus = vectorizer.fit_transform(texts)
        query = vectorizer.transform([queries[q] for q in qids])
        scores = (query @ corpus.T).toarray()
        models.append([[ids[i] for i in np.argsort(-row, kind="stable")[:k]] for row in scores])
    return [(q, a, b, labels[q]) for q, a, b in zip(qids, *models)]


def analyze(rows, k, budget):
    if not rows or len(set(q for q, *_ in rows)) != len(rows):
        raise ValueError("need distinct query identifiers")
    baseline = [np.mean([direct(row[r], {d for d,v in row[3].items() if v>0}, k)
                         for row in rows]) for r in (1, 2)]
    reverse = baseline[1] > baseline[0]
    oriented = [(q, b, a, l) if reverse else (q, a, b, l) for q,a,b,l in rows]
    curves, orders = zip(*(options(a, b, l, k) for _, a, b, l in oriented))
    dp, history = frontier(curves, budget)
    negative = np.flatnonzero(dp < -1e-12)
    minimum = int(negative[0]) if len(negative) and abs(baseline[0]-baseline[1]) > 1e-12 else None
    witness = {}
    if minimum is not None:
        remaining = minimum
        for i in reversed(range(len(rows))):
            n = int(history[i][remaining])
            if n:
                witness[rows[i][0]] = orders[i][:n]
            remaining -= n
        replay = []
        for q,a,b,l in oriented:
            positives = {d for d,v in l.items() if v>0} | set(witness.get(q, []))
            replay.append(direct(a, positives, k) - direct(b, positives, k))
        assert remaining == 0 and np.isclose(sum(replay), dp[minimum]) and sum(replay) < 0
    return dict(queries=len(rows), cutoff=k, budget=budget, baseline_A_B=baseline,
                baseline_leader="B" if reverse else ("A" if baseline[0]>baseline[1] else "tie"),
                minimum_additions=minimum, witness=witness,
                worst_mean_margin_by_budget=(np.minimum.accumulate(dp)/len(rows)).tolist())


def test():
    a,b,l,k = ["a","b","c"], ["d","b","a"], {"a":1,"c":0}, 3
    values, order = options(a,b,l,k)
    for n in range(len(order)+1):
        vals = [direct(a, {"a"} | set(x), k)-direct(b, {"a"} | set(x), k)
                for x in combinations(order,n)]
        assert abs(min(vals)-values[n]) < 1e-12
    dp,_ = frontier([values, values],4)
    for t in range(5):
        exhaustive=min(values[i]+values[t-i] for i in range(3) if 0<=t-i<3)
        assert abs(dp[t]-exhaustive)<1e-12
    print("Exhaustive per-query and global-budget checks passed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scifact", type=Path, help="folder with corpus.parquet, queries.parquet, test.tsv")
    src.add_argument("--input", type=Path, help="JSON array of [query_id, ranking_A, ranking_B, labels]")
    src.add_argument("--test", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--budget", type=int, default=100)
    args = p.parse_args()
    if args.test:
        return test()
    if args.k < 1 or args.budget < 0:
        p.error("k must be positive and budget nonnegative")
    rows = demo(args.scifact,args.k) if args.scifact else json.loads(args.input.read_text())
    result = analyze(rows,args.k,args.budget)
    if args.scifact:
        result["input_sha256"] = {n: hashlib.sha256((args.scifact/n).read_bytes()).hexdigest()
                                  for n in ("corpus.parquet","queries.parquet","test.tsv")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
