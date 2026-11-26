# Qrel Flip Budget

How many extra positive judgments could reverse two retrieval systems on mean binary nDCG@k? The program returns the minimum budget, the worst score gap at each budget, and a concrete set of query/document pairs that attains a reversal. Existing judgments are fixed. Each new positive must come from the union of the systems' top-k for that query. Unjudged documents outside this pool retain zero gain.

## A reproducible SciFact reversal

On 300 BEIR SciFact test queries and 5,183 documents, word-unigram TF-IDF scored 0.6285531 and character 3–5-gram TF-IDF scored 0.6639211 at k=10. The initial leader was the character model. With up to 13 new positives its smallest possible mean lead was still +0.00132916. With 14 it could fall to −0.00071466. These are hypothetical labels in a sensitivity analysis. The experiment makes no inference about their true relevance.

| Query | Document | Query | Document |
|---:|---:|---:|---:|
|5|17333231|784|16999023|
|72|1744752|820|3346812|
|536|23746332|821|3346812|
|575|39758684|834|12631182|
|660|7662206|1241|4335423|
|690|36653415|1278|31293581|
|715|16550075|1279|31293581|

The table lists fourteen previously unjudged pairs. The reported score difference can be reproduced by assigning each pair gain one and scoring both rankings from scratch. Document IDs may repeat across distinct queries.

## Run it

Install NumPy 2.3.5, scikit-learn 1.8.0 and DuckDB 1.4.3 to reproduce the retrievers. Save the [corpus](https://huggingface.co/datasets/BeIR/scifact/resolve/59b30fec6450351bab7c8061dc75d0afd98e0814/corpus/corpus-00000-of-00001.parquet), [queries](https://huggingface.co/datasets/BeIR/scifact/resolve/59b30fec6450351bab7c8061dc75d0afd98e0814/queries/queries-00000-of-00001.parquet) and [test judgments](https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv) as data/corpus.parquet, data/queries.parquet and data/test.tsv.


```sh
python qrel_budget.py --test
python qrel_budget.py --scifact data --budget 100 --k 10 > result.json
```

For existing rankings, pass --input with a JSON array of rows. Each row contains a query ID, ranked A document IDs, ranked B document IDs and an object mapping judged document IDs to scores. Supply every judged positive, including those outside both top-k lists, so the ideal DCG is correct. Positive scores have gain one.

## Why the budget is exact

Fix the number of added positives for one query. Its ideal DCG is then fixed, so the smallest attainable gap comes from the documents with the smallest differences in rank discounts. Repeat for all local budgets. A multiple-choice knapsack combines these per-query frontiers under the global budget and stores the choices needed to replay the witness. The self-tests compare with exhaustive enumeration on small inputs. Floating-point comparisons use finite precision. The result is specific to the two rankings, the cutoff, the test set, and the stated judgment pool.
