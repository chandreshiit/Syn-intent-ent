# Contamination check — skit_s2i (Banking)

- Synthetic utterances: **500**
- Original utterances: **11,845**

## Exact match
- Exact normalized matches: **6 (1.20%)**

## Corpus n-gram Jaccard

| n | Jaccard | Synth unique | Orig unique | Intersection |
|---|--------:|-------------:|------------:|-------------:|
| 1 | 0.3761 | 320 | 130 | 123 |
| 2 | 0.1750 | 1,018 | 251 | 189 |
| 3 | 0.1061 | 1,642 | 318 | 188 |
| 4 | 0.0672 | 2,120 | 327 | 154 |
| 5 | 0.0441 | 2,289 | 270 | 108 |

## ROUGE-L (max per synthetic utterance over all originals)
- mean: **0.6011**
- p50:  0.6000
- p90:  0.8235
- p99:  1.0000
- max:  1.0000

## BLEU-4 (max per synthetic utterance over any original)
- mean: **26.03**
- p50:  20.16
- p90:  51.33
- p99:  100.00
- max:  100.00

## Interpretation

Exact-match >5% or 4-gram Jaccard >0.50 would indicate likely contamination / memorization.
High ROUGE-L p99 (>0.80) flags individual near-copies worth manual inspection.
BLEU-4 max >75 indicates at least one synthetic utterance is a near-replica of an original.