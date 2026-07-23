#!/usr/bin/env python3
"""
One-off fix: re-project BIO tags using the same tokenization that step 04
writes to seq.in.

The previous step 02 used `re.findall(r"\\w+(?:'\\w+)?", ...)` which:
  - Splits Devanagari (Hindi) on virama (्) — "अद्यतन" becomes "अद" + "यतन"
  - Splits French/Portuguese contractions inconsistently
  - Produces token counts that don't match step 04's whitespace split

Result: 99.8% of Hindi BIO files had token count != tag count,
        French 25.3%, Portuguese 14%, Turkish 9.9%, German 6.5%.

This script reads `multiatis_bio_all_languages.json`, re-projects BIO using
whitespace tokenization (post-normalize), and writes a fixed JSON. Then you
run `04_process_multilingual_dataset.py` again to regenerate processed_data.

Usage:
    python fix_bio_tokenization.py
"""
import json
import re
from collections import defaultdict


# Mirror of normalize_text in 04_process_multilingual_dataset.py — but
# language-aware. English contraction expansion only fires for English,
# otherwise it corrupts Turkish/French (e.g. "detroit'den" → "detroit woulden").
_EN_CONTRACTIONS = {
    "'s": " is", "'ve": " have", "'t": " not", "'re": " are",
    "'m": " am", "'ll": " will", "'d": " would", "n't": " not",
}


def normalize_text(text, language_code="en"):
    text = text.lower()
    if language_code == "en":
        for c, e in _EN_CONTRACTIONS.items():
            text = text.replace(c, e)
    text = re.sub(r"(\w)\.", r"\1", text)
    # Collapse internal whitespace (newlines etc) to single space so seq.in
    # rows can never split across multiple lines.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_suffix(tok):
    """Strip apostrophe-suffix from a token: "beach'a" -> "beach"."""
    return re.sub(r"'\w*$", "", tok)


def project_bio_whitespace(translated_command, slot_translations, language_code):
    """Re-project BIO tags using whitespace tokenization.

    For matching, we also try apostrophe-stripped tokens to handle Turkish-style
    suffix attachment ("Beach'a" containing slot value "beach").
    """
    norm = normalize_text(translated_command, language_code)
    tokens = norm.split()
    # Derive a "matchable" view where apostrophe-glued suffixes are stripped
    match_tokens = [_strip_suffix(t) for t in tokens]
    bio = ["O"] * len(tokens)

    for slot in slot_translations:
        slot_name = slot.get("slot_type") or slot.get("name")
        translated_value = (slot.get("translated") or slot.get("translated_value")
                            or slot.get("value", ""))
        if not slot_name or not translated_value:
            continue
        norm_value = normalize_text(translated_value, language_code).split()
        if not norm_value:
            continue
        match_value = [_strip_suffix(t) for t in norm_value]
        # Find first occurrence
        found = False
        # 1) Exact match on raw tokens
        for i in range(len(tokens) - len(norm_value) + 1):
            if tokens[i:i + len(norm_value)] == norm_value:
                if any(b != "O" for b in bio[i:i + len(norm_value)]):
                    continue
                bio[i] = f"B-{slot_name}"
                for j in range(i + 1, i + len(norm_value)):
                    bio[j] = f"I-{slot_name}"
                found = True
                break
        # 2) Apostrophe-suffix-aware match
        if not found:
            for i in range(len(match_tokens) - len(match_value) + 1):
                if match_tokens[i:i + len(match_value)] == match_value:
                    if any(b != "O" for b in bio[i:i + len(match_value)]):
                        continue
                    bio[i] = f"B-{slot_name}"
                    for j in range(i + 1, i + len(match_value)):
                        bio[j] = f"I-{slot_name}"
                    found = True
                    break

    return " ".join(bio), tokens


def project_bio_chars(translated_command, slot_translations):
    """Character-level projection for ZH/JA (matches step 04 seq.in writer).

    Tries exact match first, then case-insensitive fallback (matches the
    original step 02 behavior). The case-insensitive fallback recovers slots
    where the embedded English entity is recorded in mixed case but appears
    differently in the translated_command.
    """
    chars = [c for c in translated_command if c.strip()]
    chars_lower = [c.lower() for c in chars]
    bio = ["O"] * len(chars)

    for slot in slot_translations:
        slot_name = slot.get("slot_type") or slot.get("name")
        translated_value = (slot.get("translated") or slot.get("translated_value")
                            or slot.get("value", ""))
        if not slot_name or not translated_value:
            continue
        v_chars = [c for c in translated_value if c.strip()]
        if not v_chars:
            continue
        # 1) exact match
        found = False
        for i in range(len(chars) - len(v_chars) + 1):
            if chars[i:i + len(v_chars)] == v_chars:
                if any(b != "O" for b in bio[i:i + len(v_chars)]):
                    continue
                bio[i] = f"B-{slot_name}"
                for j in range(i + 1, i + len(v_chars)):
                    bio[j] = f"I-{slot_name}"
                found = True
                break
        # 2) case-insensitive fallback
        if not found:
            v_lower = [c.lower() for c in v_chars]
            for i in range(len(chars_lower) - len(v_lower) + 1):
                if chars_lower[i:i + len(v_lower)] == v_lower:
                    if any(b != "O" for b in bio[i:i + len(v_lower)]):
                        continue
                    bio[i] = f"B-{slot_name}"
                    for j in range(i + 1, i + len(v_lower)):
                        bio[j] = f"I-{slot_name}"
                    break
    return " ".join(bio), chars


def main():
    in_path = "data/multiatis_multilingual_pipeline/multiatis_bio_all_languages.json"
    out_path = "data/multiatis_multilingual_pipeline/multiatis_bio_all_languages_fixed.json"

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = defaultdict(int)
    n_fixed = defaultdict(int)
    n_still_mismatched = defaultdict(int)
    n_slots_found_before = defaultdict(int)
    n_slots_found_after = defaultdict(int)

    for ex in data:
        lang = ex.get("language_code", "en")
        n_total[lang] += 1

        # Count slots found in old projection
        old_bio = ex.get("bio_tags", "").split()
        n_slots_found_before[lang] += sum(1 for t in old_bio if t.startswith("B-"))

        slot_translations = ex.get("slot_translations", [])
        cmd = ex.get("translated_command", "")

        if lang in ("zh", "ja"):
            new_bio, tokens = project_bio_chars(cmd, slot_translations)
        else:
            new_bio, tokens = project_bio_whitespace(cmd, slot_translations, lang)

        new_bio_tags = new_bio.split()
        if len(tokens) != len(new_bio_tags):
            n_still_mismatched[lang] += 1

        # Did we change anything?
        if new_bio != ex.get("bio_tags", ""):
            n_fixed[lang] += 1

        ex["bio_tags"] = new_bio
        ex["tokens"] = tokens
        ex["token_count"] = len(tokens)
        n_slots_found_after[lang] += sum(1 for t in new_bio_tags if t.startswith("B-"))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fixed JSON written to: {out_path}")
    print()
    print(f"{'lang':>5} {'n':>6} {'fixed':>7} {'still_mismatch':>16} {'slots_before':>13} {'slots_after':>13}")
    for lang in sorted(n_total):
        print(f"{lang:>5} {n_total[lang]:>6} {n_fixed[lang]:>7} "
              f"{n_still_mismatched[lang]:>16} {n_slots_found_before[lang]:>13} {n_slots_found_after[lang]:>13}")


if __name__ == "__main__":
    main()
