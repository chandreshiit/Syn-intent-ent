#!/usr/bin/env python3
"""
N-gram contamination / benchmark-memorization check (ACL R4).

For each (synthetic, original) dataset pair, computes:
  - Exact match rate: % of synthetic utts that literally exist in original
  - Corpus n-gram Jaccard (n=1..5): |S_ng INT O_ng| / |S_ng UN O_ng|
  - Per-utterance max ROUGE-L F1 (over all originals): mean + p50/p90/p99
  - Per-utterance max BLEU-4 (over all originals): mean + p50/p90/p99

Direction: synthetic -> original only (the standard contamination question
is "did the LLM regurgitate originals?").

Datasets (3): SNIPS smart-lights, Skit-S2I, MultiATIS++ (EN).

Outputs:
  - analysis/contamination/<dataset>_report.md   per-dataset narrative report
  - analysis/contamination/summary.csv           flat table for the paper appendix
"""

import argparse
import csv
import glob
import json
import os
import re
from statistics import mean

# Heavy deps
try:
    import pandas as pd
except ImportError:
    pd = None
from rouge_score import rouge_scorer
import sacrebleu


# ----- text helpers -----

_TOKEN_RE = re.compile(r"\w+(?:'\w+)?")


def normalize_text(text):
    """Lowercase + collapse whitespace. Used for exact-match + tokenization."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def get_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ----- dataset loaders -----

def load_snips_original(snips_dataset_json="SNIPS/smart-lights-en-close-field/dataset.json"):
    """Reconstruct each original utterance by concatenating data[*].text segments."""
    with open(snips_dataset_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    utts = []
    for intent_name, intent_data in d.get("intents", {}).items():
        for utt in intent_data.get("utterances", []):
            text_parts = [seg.get("text", "") for seg in utt.get("data", [])]
            utts.append(" ".join(text_parts).strip())
    return utts


def load_snips_synthetic(seq_in_path="data/snips_multilingual_pipeline/processed_data/en/all/seq.in"):
    with open(seq_in_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_skit_s2i_original(parquet_dir="skit-s2i/data"):
    if pd is None:
        raise RuntimeError("pandas required to read parquet")
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    dfs = [pd.read_parquet(f, columns=["template"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Strip Skit-S2I template placeholders so they don't artificially deflate overlap
    # with the expanded synthetic forms.
    def expand(t):
        if not isinstance(t, str):
            return ""
        out = t
        # Drop placeholder tokens entirely from the original side; the synthetic side
        # would substitute concrete values that would mismatch the placeholder tokens.
        out = re.sub(r"<[^>]+>", "", out)         # <numeric>, <Month>
        out = re.sub(r"\{\{[^}]+\}\}", "", out)    # {{Axis Bank}}
        out = re.sub(r"\(([^)]+)\)", r"\1", out)   # (credit/debit) -> credit/debit
        return out.strip()
    return [expand(t) for t in df["template"].tolist() if isinstance(t, str) and t.strip()]


def load_skit_s2i_synthetic(csv_path="data/skit_s2i_synthesis_pipeline/banking_commands_dataset_v2.csv"):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["command"].strip() for row in reader if row.get("command", "").strip()]


def load_multiatis_original(tsv_dir="multiatis_evaluation_v2/data_v2"):
    if pd is None:
        raise RuntimeError("pandas required to read tsv")
    utts = []
    for split in ("train_EN.tsv", "dev_EN.tsv", "test_EN.tsv"):
        path = os.path.join(tsv_dir, split)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t")
        utts.extend(df["utterance"].astype(str).tolist())
    return [u.strip() for u in utts if u.strip()]


def load_multiatis_synthetic(json_path="data/multiatis_multilingual_pipeline/multiatis_commands_v3.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d["command"].strip() for d in data if d.get("command", "").strip()]


# ----- metric functions -----

def exact_match_rate(synthetics, originals):
    """% of synthetic that literally match (after normalize) any original."""
    norm_originals = set(normalize_text(o) for o in originals)
    matches = 0
    for s in synthetics:
        if normalize_text(s) in norm_originals:
            matches += 1
    return 100.0 * matches / len(synthetics) if synthetics else 0.0, matches


def corpus_ngram_jaccard(synthetics, originals, max_n=5):
    """Jaccard of n-gram sets across the full corpus, per n in 1..max_n."""
    results = {}
    syn_token_lists = [tokenize(s) for s in synthetics]
    org_token_lists = [tokenize(o) for o in originals]
    for n in range(1, max_n + 1):
        syn_ngrams = set()
        for toks in syn_token_lists:
            syn_ngrams.update(get_ngrams(toks, n))
        org_ngrams = set()
        for toks in org_token_lists:
            org_ngrams.update(get_ngrams(toks, n))
        if not syn_ngrams and not org_ngrams:
            jaccard = 0.0
        else:
            inter = len(syn_ngrams & org_ngrams)
            union = len(syn_ngrams | org_ngrams)
            jaccard = (inter / union) if union > 0 else 0.0
        results[n] = {
            "jaccard": jaccard,
            "synthetic_unique": len(syn_ngrams),
            "original_unique": len(org_ngrams),
            "intersection": len(syn_ngrams & org_ngrams),
        }
    return results


def per_utterance_max_rouge_l(synthetics, originals):
    """For each synthetic, max ROUGE-L F1 over all originals. Returns list of floats."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # Pre-tokenize originals once
    org_texts = originals
    scores = []
    from tqdm import tqdm
    for s in tqdm(synthetics, desc="ROUGE-L"):
        best = 0.0
        for o in org_texts:
            r = scorer.score(o, s)["rougeL"].fmeasure
            if r > best:
                best = r
                if best >= 0.999:
                    break
        scores.append(best)
    return scores


def per_utterance_max_bleu4(synthetics, originals):
    """For each synthetic, BLEU-4 against the full original corpus as references.

    Use sacrebleu's sentence_bleu with each synthetic vs ALL originals (as multi-ref).
    This is conservative: BLEU rewards when ANY reference matches well.
    """
    scores = []
    from tqdm import tqdm
    # sacrebleu sentence_bleu expects a list of reference lists (one ref list per sentence).
    # To compute synthetic-vs-corpus, we pass all originals as multiple references.
    refs_for_one = originals  # all originals usable as references for each synthetic
    for s in tqdm(synthetics, desc="BLEU-4"):
        # Find max BLEU vs any one original (single-ref per pair).
        best = 0.0
        for o in refs_for_one:
            b = sacrebleu.sentence_bleu(s, [o], smooth_method="exp").score
            if b > best:
                best = b
                if best >= 99.999:
                    break
        scores.append(best)
    return scores


def percentile(xs, p):
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


# ----- per-dataset orchestration -----

DATASETS = {
    "snips": {
        "loader_synth": load_snips_synthetic,
        "loader_orig":  load_snips_original,
        "domain": "Smart-lights",
    },
    "skit_s2i": {
        "loader_synth": load_skit_s2i_synthetic,
        "loader_orig":  load_skit_s2i_original,
        "domain": "Banking",
    },
    "multiatis": {
        "loader_synth": load_multiatis_synthetic,
        "loader_orig":  load_multiatis_original,
        "domain": "Airline travel",
    },
}


def run_dataset(name, sample_n=None, skip_rouge=False, skip_bleu=False):
    cfg = DATASETS[name]
    print(f"\n{'=' * 70}\nDataset: {name} ({cfg['domain']})\n{'=' * 70}")
    synthetics = cfg["loader_synth"]()
    originals = cfg["loader_orig"]()
    print(f"  synthetic: {len(synthetics)} utts; original: {len(originals)} utts")
    if sample_n and sample_n < len(synthetics):
        import random
        random.seed(42)
        synthetics = random.sample(synthetics, sample_n)
        print(f"  (subsampled synthetic to {sample_n} utts for ROUGE/BLEU)")

    em_pct, em_count = exact_match_rate(synthetics, originals)
    print(f"  Exact-match rate: {em_pct:.2f}% ({em_count} / {len(synthetics)})")

    jac = corpus_ngram_jaccard(synthetics, originals, max_n=5)
    for n, r in jac.items():
        print(f"  {n}-gram Jaccard: {r['jaccard']:.4f}  "
              f"(synth_unique={r['synthetic_unique']}, orig_unique={r['original_unique']}, "
              f"intersection={r['intersection']})")

    rouge_l = None
    bleu_4 = None
    if not skip_rouge:
        rouge_l = per_utterance_max_rouge_l(synthetics, originals)
        print(f"  ROUGE-L F1 (max per synth utt): mean={mean(rouge_l):.4f}  "
              f"p50={percentile(rouge_l, 50):.4f}  p90={percentile(rouge_l, 90):.4f}  "
              f"p99={percentile(rouge_l, 99):.4f}  max={max(rouge_l):.4f}")
    if not skip_bleu:
        bleu_4 = per_utterance_max_bleu4(synthetics, originals)
        print(f"  BLEU-4 (max per synth utt): mean={mean(bleu_4):.2f}  "
              f"p50={percentile(bleu_4, 50):.2f}  p90={percentile(bleu_4, 90):.2f}  "
              f"p99={percentile(bleu_4, 99):.2f}  max={max(bleu_4):.2f}")

    return {
        "name": name,
        "domain": cfg["domain"],
        "synth_count": len(synthetics),
        "orig_count": len(originals),
        "exact_match_pct": em_pct,
        "exact_match_count": em_count,
        "jaccard_by_n": jac,
        "rouge_l": rouge_l,
        "bleu_4": bleu_4,
    }


def write_markdown_report(result, out_dir):
    name = result["name"]
    path = os.path.join(out_dir, f"{name}_report.md")
    lines = []
    lines.append(f"# Contamination check — {name} ({result['domain']})")
    lines.append("")
    lines.append(f"- Synthetic utterances: **{result['synth_count']:,}**")
    lines.append(f"- Original utterances: **{result['orig_count']:,}**")
    lines.append("")
    lines.append("## Exact match")
    lines.append(f"- Exact normalized matches: **{result['exact_match_count']} ({result['exact_match_pct']:.2f}%)**")
    lines.append("")
    lines.append("## Corpus n-gram Jaccard")
    lines.append("")
    lines.append("| n | Jaccard | Synth unique | Orig unique | Intersection |")
    lines.append("|---|--------:|-------------:|------------:|-------------:|")
    for n, r in result["jaccard_by_n"].items():
        lines.append(
            f"| {n} | {r['jaccard']:.4f} | {r['synthetic_unique']:,} | "
            f"{r['original_unique']:,} | {r['intersection']:,} |"
        )
    lines.append("")
    if result["rouge_l"] is not None:
        rl = result["rouge_l"]
        lines.append("## ROUGE-L (max per synthetic utterance over all originals)")
        lines.append(f"- mean: **{mean(rl):.4f}**")
        lines.append(f"- p50:  {percentile(rl, 50):.4f}")
        lines.append(f"- p90:  {percentile(rl, 90):.4f}")
        lines.append(f"- p99:  {percentile(rl, 99):.4f}")
        lines.append(f"- max:  {max(rl):.4f}")
        lines.append("")
    if result["bleu_4"] is not None:
        b = result["bleu_4"]
        lines.append("## BLEU-4 (max per synthetic utterance over any original)")
        lines.append(f"- mean: **{mean(b):.2f}**")
        lines.append(f"- p50:  {percentile(b, 50):.2f}")
        lines.append(f"- p90:  {percentile(b, 90):.2f}")
        lines.append(f"- p99:  {percentile(b, 99):.2f}")
        lines.append(f"- max:  {max(b):.2f}")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Exact-match >5% or 4-gram Jaccard >0.50 would indicate likely contamination / memorization.")
    lines.append("High ROUGE-L p99 (>0.80) flags individual near-copies worth manual inspection.")
    lines.append("BLEU-4 max >75 indicates at least one synthetic utterance is a near-replica of an original.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def write_summary_csv(results, out_dir):
    path = os.path.join(out_dir, "summary.csv")
    fieldnames = [
        "dataset", "domain", "synth_count", "orig_count",
        "exact_match_pct", "exact_match_count",
        "jaccard_1", "jaccard_2", "jaccard_3", "jaccard_4", "jaccard_5",
        "rouge_l_mean", "rouge_l_p50", "rouge_l_p90", "rouge_l_p99", "rouge_l_max",
        "bleu_4_mean",  "bleu_4_p50",  "bleu_4_p90",  "bleu_4_p99",  "bleu_4_max",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "dataset": r["name"],
                "domain": r["domain"],
                "synth_count": r["synth_count"],
                "orig_count": r["orig_count"],
                "exact_match_pct": f"{r['exact_match_pct']:.2f}",
                "exact_match_count": r["exact_match_count"],
            }
            for n in (1, 2, 3, 4, 5):
                row[f"jaccard_{n}"] = f"{r['jaccard_by_n'][n]['jaccard']:.4f}"
            if r.get("rouge_l") is not None:
                rl = r["rouge_l"]
                row.update({
                    "rouge_l_mean": f"{mean(rl):.4f}",
                    "rouge_l_p50":  f"{percentile(rl, 50):.4f}",
                    "rouge_l_p90":  f"{percentile(rl, 90):.4f}",
                    "rouge_l_p99":  f"{percentile(rl, 99):.4f}",
                    "rouge_l_max":  f"{max(rl):.4f}",
                })
            if r.get("bleu_4") is not None:
                b = r["bleu_4"]
                row.update({
                    "bleu_4_mean": f"{mean(b):.2f}",
                    "bleu_4_p50":  f"{percentile(b, 50):.2f}",
                    "bleu_4_p90":  f"{percentile(b, 90):.2f}",
                    "bleu_4_p99":  f"{percentile(b, 99):.2f}",
                    "bleu_4_max":  f"{max(b):.2f}",
                })
            w.writerow(row)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="N-gram contamination check (ACL R4)")
    parser.add_argument("--datasets", nargs="+", default=["snips", "skit_s2i", "multiatis"],
                        choices=list(DATASETS.keys()),
                        help="Which datasets to evaluate")
    parser.add_argument("--out-dir", type=str, default="analysis/contamination",
                        help="Output directory")
    parser.add_argument("--rouge-bleu-sample", type=int, default=None,
                        help="If set, subsample synthetic to this size for ROUGE/BLEU (Jaccard always uses full corpora)")
    parser.add_argument("--skip-rouge", action="store_true",
                        help="Skip ROUGE-L (faster)")
    parser.add_argument("--skip-bleu", action="store_true",
                        help="Skip BLEU-4 (faster)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for name in args.datasets:
        r = run_dataset(name, sample_n=args.rouge_bleu_sample,
                        skip_rouge=args.skip_rouge, skip_bleu=args.skip_bleu)
        write_markdown_report(r, args.out_dir)
        results.append(r)
    write_summary_csv(results, args.out_dir)


if __name__ == "__main__":
    main()
