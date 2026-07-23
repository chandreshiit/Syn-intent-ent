# Contamination check — snips (Smart-lights)

- Synthetic utterances: **500**
- Original utterances: **1,765**

## Exact match
- Exact normalized matches: **19 (3.80%)**

## Corpus n-gram Jaccard

| n | Jaccard | Synth unique | Orig unique | Intersection |
|---|--------:|-------------:|------------:|-------------:|
| 1 | 0.2557 | 119 | 323 | 90 |
| 2 | 0.1081 | 297 | 1,620 | 187 |
| 3 | 0.0857 | 637 | 3,076 | 293 |
| 4 | 0.0623 | 829 | 3,930 | 279 |
| 5 | 0.0384 | 759 | 3,733 | 166 |

## ROUGE-L (max per synthetic utterance over all originals)
- mean: **0.8001**
- p50:  0.8000
- p90:  0.9091
- p99:  1.0000
- max:  1.0000

## BLEU-4 (max per synthetic utterance over any original)
- mean: **48.38**
- p50:  44.83
- p90:  70.14
- p99:  84.09
- max:  100.00

## Interpretation

Exact-match >5% or 4-gram Jaccard >0.50 would indicate likely contamination / memorization.
High ROUGE-L p99 (>0.80) flags individual near-copies worth manual inspection.
BLEU-4 max >75 indicates at least one synthetic utterance is a near-replica of an original.