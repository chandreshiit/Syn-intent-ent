# Contamination check — multiatis (Airline travel)

- Synthetic utterances: **500**
- Original utterances: **6,382**

## Exact match
- Exact normalized matches: **2 (0.40%)**

## Corpus n-gram Jaccard

| n | Jaccard | Synth unique | Orig unique | Intersection |
|---|--------:|-------------:|------------:|-------------:|
| 1 | 0.2973 | 342 | 618 | 220 |
| 2 | 0.0985 | 1,083 | 3,278 | 391 |
| 3 | 0.0384 | 1,814 | 8,134 | 368 |
| 4 | 0.0175 | 2,067 | 12,437 | 249 |
| 5 | 0.0076 | 2,005 | 15,294 | 131 |

## ROUGE-L (max per synthetic utterance over all originals)
- mean: **0.5931**
- p50:  0.5882
- p90:  0.7368
- p99:  0.8571
- max:  1.0000

## BLEU-4 (max per synthetic utterance over any original)
- mean: **28.18**
- p50:  24.45
- p90:  47.49
- p99:  75.98
- max:  100.00

## Interpretation

Exact-match >5% or 4-gram Jaccard >0.50 would indicate likely contamination / memorization.
High ROUGE-L p99 (>0.80) flags individual near-copies worth manual inspection.
BLEU-4 max >75 indicates at least one synthetic utterance is a near-replica of an original.