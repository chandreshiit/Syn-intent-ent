#!/usr/bin/env python3
"""
Diagnostic: train Snips NLU on synth-only, dump errors on the 353 real test utts.

Uses the same 80/20 split (seed=42) as snips_domain_tuning_snipsnlu.py, so the
353 real_test utts are IDENTICAL to those the sweep evaluated on.

Outputs:
  phase3/results/snips_synthonly_errors.json  — every real test utt with:
      { text, gold_intent, pred_intent, gold_slots, pred_slots,
        intent_correct, slot_correct_all, slot_missed, slot_wrong }
  phase3/results/snips_synthonly_errors_summary.txt — printable breakdown:
      * intent confusion matrix (gold → pred)
      * missed-slot frequency by (intent, slot_name)
      * per-intent error rate
      * short lexical/pattern analysis (opener words, contractions, length)

Runs in WSL .venv_wsl.
"""
import json
import os
import re
from collections import Counter, defaultdict


REAL_DATASET = "data/snips_real_close/dataset.json"
SYNTH_DATASET = "data/snips_synth_for_snipsnlu/dataset.json"
SEED = 42


def flatten(ds):
    return [(intent, u) for intent, d in ds["intents"].items() for u in d["utterances"]]


def utt_to_text(u):
    return "".join(c.get("text", "") for c in u.get("data", [])).strip()


def gold_slots(u):
    return [{"slot_name": c["slot_name"], "raw_value": c.get("text", "").strip()}
            for c in u.get("data", []) if c.get("slot_name")]


def predicted_slots(pred):
    return [{"slot_name": s.get("slotName"), "raw_value": s.get("rawValue", "").strip()}
            for s in pred.get("slots", [])]


def build_dataset(language, intents_list, entities, utts_by_intent):
    return {
        "language": language,
        "intents": {i: {"utterances": utts_by_intent.get(i, [])} for i in intents_list},
        "entities": entities,
    }


def main():
    from sklearn.model_selection import train_test_split
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.default_configs import CONFIG_EN

    real_ds = json.load(open(REAL_DATASET, encoding="utf-8"))
    synth_ds = json.load(open(SYNTH_DATASET, encoding="utf-8"))
    intents = list(real_ds["intents"].keys())

    real_flat = flatten(real_ds)
    synth_flat = flatten(synth_ds)
    ri = [it for it, _ in real_flat]
    _, real_test, _, _ = train_test_split(
        real_flat, ri, test_size=0.20, stratify=ri, random_state=SEED)
    print(f"real_test: {len(real_test)}  synth: {len(synth_flat)}")

    # Train Snips NLU on synth only, using the union of entity values from both
    # datasets (so gazetteer coverage isn't the reason for false negatives; we
    # want to isolate text-distribution problems).
    from copy import deepcopy
    merged_ents = deepcopy(real_ds.get("entities", {}))
    for k, se in synth_ds.get("entities", {}).items():
        if k not in merged_ents:
            merged_ents[k] = deepcopy(se)
            continue
        if "data" in merged_ents[k] and "data" in se:
            seen = {i["value"].lower() for i in merged_ents[k]["data"]}
            for i in se["data"]:
                if i["value"].lower() not in seen:
                    merged_ents[k]["data"].append(i)
                    seen.add(i["value"].lower())

    utts_by_intent = defaultdict(list)
    for intent, u in synth_flat:
        utts_by_intent[intent].append(u)
    train_ds = build_dataset(real_ds.get("language", "en"), intents,
                              merged_ents, utts_by_intent)

    print("Training Snips NLU on synth only ...")
    engine = SnipsNLUEngine(config=CONFIG_EN)
    engine.fit(train_ds)

    # Predict + record every test utt
    per_utt = []
    intent_confuse = Counter()
    slot_missed = Counter()
    slot_wrong = Counter()
    slot_correct_by_intent = Counter()
    slot_total_by_intent = Counter()
    for gold_intent, u in real_test:
        text = utt_to_text(u)
        pred = engine.parse(text)
        pi = pred.get("intent", {}).get("intentName")
        gs = gold_slots(u)
        ps = predicted_slots(pred) if pi == gold_intent else []
        gs_pairs = [(s["slot_name"], s["raw_value"].lower()) for s in gs]
        ps_pairs = [(s["slot_name"], s["raw_value"].lower()) for s in ps]
        gold_used = [False] * len(gs_pairs)
        missed_here = []
        wrong_here = []
        for ppair in ps_pairs:
            matched = False
            for j, gpair in enumerate(gs_pairs):
                if not gold_used[j] and ppair == gpair:
                    gold_used[j] = True; matched = True; break
            if not matched:
                wrong_here.append(ppair)
        for j, gpair in enumerate(gs_pairs):
            if not gold_used[j]:
                missed_here.append(gpair)

        intent_ok = pi == gold_intent
        if not intent_ok:
            intent_confuse[(gold_intent, str(pi))] += 1
        for sn, sv in missed_here:
            slot_missed[(gold_intent, sn)] += 1
        for sn, sv in wrong_here:
            slot_wrong[(gold_intent, sn)] += 1
        slot_correct_by_intent[gold_intent] += (len(gs_pairs) - len(missed_here))
        slot_total_by_intent[gold_intent] += len(gs_pairs)

        per_utt.append({
            "text": text,
            "gold_intent": gold_intent, "pred_intent": pi,
            "gold_slots": gs, "pred_slots": ps,
            "intent_correct": intent_ok,
            "slot_missed": missed_here, "slot_wrong": wrong_here,
            "slot_correct_all": intent_ok and not missed_here and not wrong_here,
        })

    # Summary
    n_intent_wrong = sum(1 for r in per_utt if not r["intent_correct"])
    n_slot_any_err = sum(1 for r in per_utt if r["slot_missed"] or r["slot_wrong"])
    print(f"\nintent wrong: {n_intent_wrong} / {len(per_utt)} = {n_intent_wrong/len(per_utt)*100:.2f}%")
    print(f"any slot err (missed or wrong): {n_slot_any_err} / {len(per_utt)}")

    # Lexical / structural analysis on ERROR cases
    errs = [r for r in per_utt if not r["intent_correct"] or r["slot_missed"] or r["slot_wrong"]]
    openers_all = Counter(r["text"].split()[0].lower() if r["text"].split() else "" for r in per_utt)
    openers_err = Counter(r["text"].split()[0].lower() if r["text"].split() else "" for r in errs)
    contractions_all = sum(1 for r in per_utt if re.search(r"'\w", r["text"]))
    contractions_err = sum(1 for r in errs if re.search(r"'\w", r["text"]))
    lens = [len(r["text"].split()) for r in per_utt]
    lens_err = [len(r["text"].split()) for r in errs]

    out_dir = "phase3/results"
    with open(os.path.join(out_dir, "snips_synthonly_errors.json"), "w") as f:
        json.dump({"per_utt": per_utt,
                    "n_test": len(per_utt),
                    "n_intent_wrong": n_intent_wrong,
                    "n_slot_err": n_slot_any_err}, f, indent=2)

    lines = []
    def p(s=""):
        lines.append(s); print(s)
    p("=== Intent confusion (gold -> pred) : top ===")
    for (g, pred), n in intent_confuse.most_common():
        p(f"  {g:>20} -> {pred:<20} : {n}")
    p("")
    p("=== Missed slots (gold present, model missed) : (intent, slot) ===")
    for (i, s), n in slot_missed.most_common():
        p(f"  ({i}, {s}) : {n}")
    p("")
    p("=== Wrong slots (model extracted, not in gold) : (intent, slot) ===")
    for (i, s), n in slot_wrong.most_common():
        p(f"  ({i}, {s}) : {n}")
    p("")
    p("=== Per-intent error breakdown ===")
    err_by_intent = Counter(r["gold_intent"] for r in per_utt if not r["intent_correct"])
    total_by_intent = Counter(r["gold_intent"] for r in per_utt)
    p(f"  {'intent':>22} {'n_total':>8} {'n_intent_err':>12} {'n_slot_err':>11} {'slot_recall':>12}")
    slot_err_by_intent = Counter()
    for r in per_utt:
        if r["slot_missed"] or r["slot_wrong"]:
            slot_err_by_intent[r["gold_intent"]] += 1
    for it in intents:
        recall = slot_correct_by_intent[it] / slot_total_by_intent[it] if slot_total_by_intent[it] > 0 else 0.0
        p(f"  {it:>22} {total_by_intent[it]:>8} {err_by_intent[it]:>12} {slot_err_by_intent[it]:>11} {recall:>12.4f}")
    p("")
    p("=== Openers of error utterances (top 15) ===")
    p(f"  {'opener':<15} {'err':>5} {'total':>6} {'err_rate':>10}")
    for w, n in openers_err.most_common(15):
        rate = n / openers_all[w] if openers_all[w] > 0 else 0.0
        p(f"  {w:<15} {n:>5} {openers_all[w]:>6} {rate:>10.2f}")
    p("")
    p(f"Contractions in errors: {contractions_err} / {len(errs)}  vs  all: {contractions_all} / {len(per_utt)}")
    if lens_err:
        p(f"Length (words) — errors mean/median: {sum(lens_err)/len(lens_err):.1f} / {sorted(lens_err)[len(lens_err)//2]}")
    p(f"Length (words) — all    mean/median: {sum(lens)/len(lens):.1f} / {sorted(lens)[len(lens)//2]}")
    p("")
    p("=== Sample of error utterances (up to 30) ===")
    shown = 0
    for r in per_utt:
        if r["intent_correct"] and not r["slot_missed"] and not r["slot_wrong"]:
            continue
        gs_str = ",".join(f"{s['slot_name']}={s['raw_value']}" for s in r["gold_slots"]) or "-"
        ps_str = ",".join(f"{s['slot_name']}={s['raw_value']}" for s in r["pred_slots"]) or "-"
        marks = []
        if not r["intent_correct"]:
            marks.append(f"intent:{r['gold_intent']}->{r['pred_intent']}")
        if r["slot_missed"]:
            marks.append(f"missed:{r['slot_missed']}")
        if r["slot_wrong"]:
            marks.append(f"wrong:{r['slot_wrong']}")
        p(f"  [{'; '.join(marks)}]")
        p(f"    text  : {r['text']!r}")
        p(f"    gold  : {gs_str}")
        p(f"    pred  : {ps_str}")
        shown += 1
        if shown >= 30:
            break

    with open(os.path.join(out_dir, "snips_synthonly_errors_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
