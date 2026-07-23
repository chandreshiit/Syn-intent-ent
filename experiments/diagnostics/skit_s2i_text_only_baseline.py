#!/usr/bin/env python3
"""
Skit-S2I — text-only intent classification.

Train BERT (bert-base-uncased) on the synthetic Skit-S2I command TEXT,
test on the real Skit-S2I gold transcripts. Same intent label space (14
banking intents). Isolates the text generation pipeline from the audio
channel mismatch.

If this gets high accuracy while the audio-side Whisper-tiny.en model
(49.6% on synth-only) fails on the same real test set, then the
generation pipeline's text is fine and the bottleneck is purely audio.

Inputs:
  - synth: data/skit_s2i_synthesis_pipeline/banking_commands_dataset_v2.csv
           columns: command, intent (name), category
  - real:  data/skit_s2i_real_audio/metadata.csv
           columns: split, idx, intent_class (0..13), template, speaker_id, audio_path
  - mapping: data/skit_s2i_synthesis_pipeline/output/intent_info.csv
           columns: intent_class, intent (name), description

Output:
  phase3/results/skit_s2i_text_only.json
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_intent_mapping():
    path = os.path.join(REPO, "data/skit_s2i_synthesis_pipeline/output/intent_info.csv")
    name_to_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name_to_id[row["intent"].strip()] = int(row["intent_class"])
    assert len(name_to_id) == 14, f"expected 14 intents, got {len(name_to_id)}"
    return name_to_id


def load_synth(name_to_id):
    path = os.path.join(REPO, "data/skit_s2i_synthesis_pipeline/banking_commands_dataset_v2.csv")
    examples = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row["command"].strip()
            name = row["intent"].strip()
            if not text or name not in name_to_id:
                skipped += 1
                continue
            examples.append({"text": text, "label": name_to_id[name]})
    return examples, skipped


def load_real():
    path = os.path.join(REPO, "data/skit_s2i_real_audio/metadata.csv")
    examples = []
    skipped = 0
    by_split = {"train": [], "test": []}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("template", "").strip()
            try:
                label = int(row["intent_class"])
            except (KeyError, ValueError):
                skipped += 1
                continue
            split = row.get("split", "test").strip().lower()
            entry = {"text": text, "label": label, "split": split}
            examples.append(entry)
            by_split.setdefault(split, []).append(entry)
    return examples, by_split, skipped


class TextDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, k):
        ex = self.examples[k]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
        }


class BertIntentClassifier(nn.Module):
    def __init__(self, n_class=14, backbone="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(backbone)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, n_class)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_tp = {}
    per_class_n = {}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                per_class_n[l] = per_class_n.get(l, 0) + 1
                if p == l:
                    per_class_tp[l] = per_class_tp.get(l, 0) + 1
    acc = correct / total if total else 0.0
    return acc, per_class_tp, per_class_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--backbone", type=str, default="bert-base-uncased")
    ap.add_argument("--out", type=str,
                    default=os.path.join(REPO, "phase3/results/skit_s2i_text_only.json"))
    ap.add_argument("--test-split", type=str, default="test",
                    help="Which split of the real data to test on: test (default), train, or all.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    name_to_id = load_intent_mapping()
    id_to_name = {v: k for k, v in name_to_id.items()}
    print(f"intent mapping: {name_to_id}")

    synth, synth_skipped = load_synth(name_to_id)
    real, real_by_split, real_skipped = load_real()
    print(f"synth: {len(synth)} examples (skipped {synth_skipped})")
    print(f"real:  {len(real)} examples (skipped {real_skipped})")
    print(f"real splits: " + ", ".join(f"{k}={len(v)}" for k, v in real_by_split.items()))

    if args.test_split == "all":
        test_examples = real
    else:
        test_examples = real_by_split.get(args.test_split, [])
    print(f"using test split: {args.test_split} ({len(test_examples)} utts)")

    # Hold out 10% of synth for dev
    rng = random.Random(args.seed)
    rng.shuffle(synth)
    n_dev = max(1, int(0.1 * len(synth)))
    dev_ex = synth[:n_dev]
    train_ex = synth[n_dev:]
    print(f"train: {len(train_ex)}  dev: {len(dev_ex)}  test: {len(test_examples)}")

    tokenizer = BertTokenizer.from_pretrained(args.backbone)
    train_ds = TextDataset(train_ex, tokenizer, max_length=args.max_length)
    dev_ds = TextDataset(dev_ex, tokenizer, max_length=args.max_length)
    test_ds = TextDataset(test_examples, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BertIntentClassifier(n_class=14, backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_dev = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total = 0
        correct = 0
        sum_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)
        train_acc = correct / total if total else 0.0
        dev_acc, _, _ = evaluate(model, dev_loader, device)
        secs = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs}: train_loss={sum_loss/total:.4f} "
              f"train_acc={train_acc:.4f}  dev_acc={dev_acc:.4f}  secs={secs:.1f}", flush=True)
        history.append({"epoch": epoch + 1, "train_acc": train_acc, "dev_acc": dev_acc})
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    test_acc, tp, n = evaluate(model, test_loader, device)
    per_class = {id_to_name[i]: {"acc": tp.get(i, 0) / n.get(i, 1) if n.get(i) else 0,
                                  "tp": tp.get(i, 0), "n": n.get(i, 0)} for i in range(14)}

    # Partition-aware eval: split test by overlap with synth train texts
    synth_text_set = {ex["text"].strip().lower() for ex in train_ex} | \
                     {ex["text"].strip().lower() for ex in dev_ex}
    overlap_ex = [e for e in test_examples if e["text"].strip().lower() in synth_text_set]
    novel_ex = [e for e in test_examples if e["text"].strip().lower() not in synth_text_set]
    overlap_loader = DataLoader(TextDataset(overlap_ex, tokenizer, args.max_length),
                                 batch_size=args.batch_size, shuffle=False)
    novel_loader = DataLoader(TextDataset(novel_ex, tokenizer, args.max_length),
                               batch_size=args.batch_size, shuffle=False)
    overlap_acc, _, _ = evaluate(model, overlap_loader, device) if overlap_ex else (None, None, None)
    novel_acc, _, _ = evaluate(model, novel_loader, device) if novel_ex else (None, None, None)
    print(f"\n=== Test partition breakdown ===")
    print(f"  In synth train (literal match): {len(overlap_ex)} utts, acc={overlap_acc:.4f}" if overlap_ex else "  no overlap")
    print(f"  Novel (not in synth train):     {len(novel_ex)} utts, acc={novel_acc:.4f}" if novel_ex else "  no novel")

    summary = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "train_size": len(train_ex),
        "dev_size": len(dev_ex),
        "test_split": args.test_split,
        "test_size": len(test_examples),
        "best_dev_acc": best_dev,
        "test_intent_acc": test_acc,
        "test_overlap_size": len(overlap_ex),
        "test_overlap_acc": overlap_acc,
        "test_novel_size": len(novel_ex),
        "test_novel_acc": novel_acc,
        "per_class": per_class,
        "history": history,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== test intent acc on real Skit-S2I {args.test_split}: {test_acc:.4f} ===")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
