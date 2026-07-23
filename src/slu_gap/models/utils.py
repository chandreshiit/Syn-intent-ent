"""
Utility functions for IC/SL baseline training.
Adapted from the original MultiATIS++ repository.
"""

import csv
import torch
import numpy as np
from collections import defaultdict

PAD = '[PAD]'


def load_tsv(filepath):
    """Load data from TSV file.
    
    Expected columns: u_id, utterance, slot-labels, intent
    
    Returns:
        example_ids: List of utterance IDs
        utterances: List of tokenized utterances (list of words)
        labels: List of slot labels (list of BIO tags)
        intents: List of intent labels
    """
    example_ids = []
    utterances = []
    labels = []
    intents = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            example_ids.append(int(row['u_id']))
            utterances.append(row['utterance'].split())
            labels.append(row['slot-labels'].split())
            intents.append(row['intent'])
    
    return example_ids, utterances, labels, intents


def get_label_indices(input_file):
    """Build label-to-index mappings from training data.
    
    Returns:
        intent2idx: Dict mapping intent labels to indices
        label2idx: Dict mapping slot labels to indices
    """
    _, _, train_labels, train_intents = load_tsv(input_file)
    
    # Build intent vocabulary
    intent2idx = {}
    for intent in train_intents:
        if intent not in intent2idx:
            intent2idx[intent] = len(intent2idx)
    
    # Build slot label vocabulary
    label2idx = {}
    for labels in train_labels:
        for label in labels:
            if label not in label2idx:
                label2idx[label] = len(label2idx)
    
    # Ensure all I- tags exist for corresponding B- tags
    new_labels = []
    for label in list(label2idx.keys()):
        if label.startswith('B-'):
            cont_label = 'I-' + label[2:]
            if cont_label not in label2idx:
                new_labels.append(cont_label)
    
    for label in new_labels:
        label2idx[label] = len(label2idx)
    
    # Add PAD token
    if PAD not in label2idx:
        label2idx[PAD] = len(label2idx)
    
    return intent2idx, label2idx


def label2index(mapping, key):
    """Convert label to index, returning len(mapping) for unknown labels."""
    return mapping[key] if key in mapping else len(mapping)


def process_seq_labels(labels, predictions, ignore_id=-1):
    """Process sequence labels for metric computation.
    
    Args:
        labels: Tensor of shape (batch_size, seq_length)
        predictions: Tensor of shape (batch_size, seq_length, num_labels)
        ignore_id: Label ID to ignore (e.g., padding)
    
    Returns:
        Tuple of (filtered_labels, filtered_preds)
    """
    # Flatten
    labels = labels.view(-1).cpu().numpy()
    predictions = predictions.view(-1, predictions.size(-1)).cpu().numpy()
    
    # Filter out ignored labels
    keep_idx = np.where(labels != ignore_id)[0]
    filtered_labels = labels[keep_idx]
    filtered_preds = predictions[keep_idx]
    
    return filtered_labels, filtered_preds


def merge_slots(slot_predictions, alignment):
    """Merge subword slot predictions back to word level.
    
    For BERT tokenization, a word may be split into multiple subwords.
    This function merges predictions using a simple voting scheme:
    - Take the first B- tag if any
    - Otherwise take the first I- tag if any
    - Otherwise take O
    
    Args:
        slot_predictions: List of predicted slot labels for subwords
        alignment: List of indices indicating word boundaries
    
    Returns:
        List of merged slot labels at word level
    """
    merged_slots = []
    if not alignment:
        # Defensive: some character-level inputs (Japanese, Chinese) can
        # truncate to an empty alignment under BERT's max_length. Return
        # empty list and let the CoNLL evaluator handle it.
        return merged_slots
    n_pred = len(slot_predictions)
    start_idx = alignment[0]

    for end_idx in alignment[1:]:
        # BERT truncates to max_length, so end_idx (and even start_idx) may
        # exceed the predicted slot sequence for long character-level inputs
        # (Japanese, Chinese). Pad missing positions with 'O'.
        if start_idx >= n_pred:
            merged_slots.append('O')
        else:
            clamped_end = min(end_idx, n_pred)
            tag = slot_predictions[start_idx]
            for slot in slot_predictions[start_idx:clamped_end]:
                if slot.startswith('B-') and tag == 'O':
                    tag = slot
                elif slot.startswith('I-') and tag == 'O':
                    tag = slot
            merged_slots.append(tag)
        start_idx = end_idx
    
    return merged_slots


class ICSlDataset(torch.utils.data.Dataset):
    """Dataset for Intent Classification and Slot Labeling.
    
    Handles tokenization and label alignment for BERT-based models.
    """
    
    def __init__(self, filepath, tokenizer, label2idx, intent2idx, max_length=128):
        """Initialize dataset.
        
        Args:
            filepath: Path to TSV file
            tokenizer: HuggingFace tokenizer
            label2idx: Dict mapping slot labels to indices
            intent2idx: Dict mapping intent labels to indices
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.intent2idx = intent2idx
        self.max_length = max_length
        self.pad_label_id = label2idx[PAD]
        
        # Load and process data
        self.examples = []
        example_ids, utterances, labels, intents = load_tsv(filepath)
        
        for eid, words, slot_labels, intent in zip(example_ids, utterances, labels, intents):
            example = self._process_example(eid, words, slot_labels, intent)
            if example is not None:
                self.examples.append(example)
    
    def _process_example(self, eid, words, slot_labels, intent):
        """Process a single example with BERT tokenization and label alignment."""
        # Tokenize with alignment
        bert_tokens = ['[CLS]']
        bert_labels = []
        alignment = []
        
        for word, label in zip(words, slot_labels):
            alignment.append(len(bert_labels))
            word_tokens = self.tokenizer.tokenize(word)
            
            if not word_tokens:
                word_tokens = ['[UNK]']
            
            bert_tokens.extend(word_tokens)
            
            # Extend labels: first token gets original label, rest get I- or original
            if label.startswith('B-'):
                cont_label = 'I-' + label[2:]
                bert_labels.extend([label] + [cont_label] * (len(word_tokens) - 1))
            else:
                bert_labels.extend([label] * len(word_tokens))
        
        bert_tokens.append('[SEP]')
        bert_labels.append(PAD)
        alignment.append(len(bert_labels) - 1)
        
        # Convert to IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        label_ids = [label2index(self.label2idx, lbl) for lbl in bert_labels]
        intent_id = label2index(self.intent2idx, intent)
        
        # Truncate if needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            label_ids = label_ids[:self.max_length - 1]
        
        return {
            'example_id': eid,
            'token_ids': token_ids,
            'label_ids': label_ids,
            'intent_id': intent_id,
            'alignment': alignment,
            'valid_length': len(token_ids)
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id=0, pad_label_id=0):
    """Collate function for DataLoader.
    
    Pads sequences to the maximum length in the batch.
    """
    max_len = max(item['valid_length'] for item in batch)
    
    example_ids = []
    token_ids = []
    label_ids = []
    intent_ids = []
    valid_lengths = []
    alignments = []
    
    for item in batch:
        example_ids.append(item['example_id'])
        
        # Pad token_ids
        padded_tokens = item['token_ids'] + [pad_token_id] * (max_len - len(item['token_ids']))
        token_ids.append(padded_tokens)
        
        # Pad label_ids
        padded_labels = item['label_ids'] + [pad_label_id] * (max_len - len(item['label_ids']))
        label_ids.append(padded_labels)
        
        intent_ids.append(item['intent_id'])
        valid_lengths.append(item['valid_length'])
        alignments.append(item['alignment'])
    
    return {
        'example_ids': example_ids,
        'token_ids': torch.tensor(token_ids, dtype=torch.long),
        'label_ids': torch.tensor(label_ids, dtype=torch.long),
        'intent_ids': torch.tensor(intent_ids, dtype=torch.long),
        'valid_lengths': torch.tensor(valid_lengths, dtype=torch.long),
        'alignments': alignments
    }


class LSTMDataset(torch.utils.data.Dataset):
    """Dataset for LSTM-based IC/SL model.
    
    Uses simple word-level tokenization without subword splitting.
    """
    
    def __init__(self, filepath, vocab, label2idx, intent2idx, max_length=128):
        """Initialize dataset.
        
        Args:
            filepath: Path to TSV file
            vocab: Dict mapping words to indices
            label2idx: Dict mapping slot labels to indices
            intent2idx: Dict mapping intent labels to indices
            max_length: Maximum sequence length
        """
        self.vocab = vocab
        self.label2idx = label2idx
        self.intent2idx = intent2idx
        self.max_length = max_length
        self.pad_label_id = label2idx[PAD]
        
        # Load and process data
        self.examples = []
        example_ids, utterances, labels, intents = load_tsv(filepath)
        
        for eid, words, slot_labels, intent in zip(example_ids, utterances, labels, intents):
            example = self._process_example(eid, words, slot_labels, intent)
            if example is not None:
                self.examples.append(example)
    
    def _process_example(self, eid, words, slot_labels, intent):
        """Process a single example."""
        # Add [CLS] token at the beginning
        tokens = ['[CLS]'] + words + ['[SEP]']
        labels = [PAD] + slot_labels + [PAD]
        
        # Convert to IDs
        token_ids = [self.vocab.get(w.lower(), self.vocab.get('[UNK]', 0)) for w in tokens]
        label_ids = [label2index(self.label2idx, lbl) for lbl in labels]
        intent_id = label2index(self.intent2idx, intent)
        
        # Truncate if needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            label_ids = label_ids[:self.max_length]
        
        return {
            'example_id': eid,
            'token_ids': token_ids,
            'label_ids': label_ids,
            'intent_id': intent_id,
            'valid_length': len(token_ids),
            'words': words
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def build_vocab(filepaths, min_freq=1):
    """Build vocabulary from training data.
    
    Args:
        filepaths: List of TSV file paths
        min_freq: Minimum word frequency to include
    
    Returns:
        vocab: Dict mapping words to indices
    """
    word_counts = defaultdict(int)
    
    for filepath in filepaths:
        _, utterances, _, _ = load_tsv(filepath)
        for words in utterances:
            for word in words:
                word_counts[word.lower()] += 1
    
    # Build vocabulary
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
    for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab
