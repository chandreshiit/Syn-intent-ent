"""
Bi-LSTM baseline for intent classification and slot labelling.

Adapted from the official MultiATIS++ repository. Provides the randomly
initialised (no pretraining) comparison arm for the MultiATIS++ per-language
results.

Callers configure the module before training by assigning `model_dir` and
calling `set_seed`:

    from slu_gap.models import lstm_alone

    lstm_alone.set_seed(42)
    lstm_alone.model_dir = "/path/to/checkpoints"
    model = lstm_alone.train(name, train_tsv, dev_tsv, vocab, intent2idx, label2idx)
"""

import logging
import os
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from .utils import (
    load_tsv, LSTMDataset, PAD
)
from .conlleval import evaluate_conll, format_conll_output

warnings.filterwarnings('ignore')

# Module-level configuration. Callers override `model_dir` (and occasionally
# `data_dir`) after import; see the docstring above.
random_seed = 42
data_dir = "./data/"
model_dir = "./checkpoints/"
conll_prediction_file = os.path.join(data_dir, "conll.pred")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log = logging.getLogger('slu_gap.lstm')
log.setLevel(logging.DEBUG)
_formatter = logging.Formatter(
    fmt='[%(levelname)s] %(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(_formatter)
log.addHandler(_console)


def set_seed(seed=42):
    """Seed the torch RNGs.

    Call once immediately after import and before constructing any model.
    Seeding at import time (as the original script did) made the module read
    `sys.argv`, which forced every caller to swap out its own CLI arguments.
    """
    global random_seed
    random_seed = seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def log_to_file(path):
    """Additionally write this module's logs to `path`."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    handler = logging.FileHandler(path, mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(_formatter)
    log.addHandler(handler)


class ICSL(nn.Module):
    """Bidirectional LSTM model for Intent Classification and Slot Labeling.
    
    The model feeds token ids into a biLSTM to get sequence representations,
    then applies dense layers for IC/SL classification.
    """
    
    def __init__(self, vocab_size, num_slot_labels, num_intents, 
                 embed_size=256, rnn_hidden_size=128, rnn_layers=1, 
                 rnn_dropout=0.1, embed_dropout=0.1):
        super(ICSL, self).__init__()
        
        # Embedding layer
        self.word_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            embed_size, 
            rnn_hidden_size, 
            num_layers=rnn_layers,
            bidirectional=True, 
            dropout=rnn_dropout if rnn_layers > 1 else 0,
            batch_first=True
        )
        
        # Classifiers
        self.intent_classifier = nn.Linear(rnn_hidden_size * 2, num_intents)
        self.slot_classifier = nn.Linear(rnn_hidden_size * 2, num_slot_labels)
    
    def forward(self, input_ids):
        """Forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
        
        Returns:
            intent_logits: Intent predictions, shape (batch_size, num_intents)
            slot_logits: Slot predictions, shape (batch_size, seq_length-1, num_slot_labels)
        """
        # Embedding
        embed = self.word_embed(input_ids)  # (batch_size, seq_length, embed_size)
        embed = self.embed_dropout(embed)
        
        # LSTM encoding
        hidden, _ = self.rnn(embed)  # (batch_size, seq_length, hidden_size * 2)
        
        # Intent prediction from first token ([CLS])
        intent_logits = self.intent_classifier(hidden[:, 0, :])
        
        # Slot prediction from all tokens (excluding [CLS])
        slot_logits = self.slot_classifier(hidden[:, 1:, :])
        
        return intent_logits, slot_logits


class ICSLLoss(nn.Module):
    """Combined loss for Intent Classification and Slot Labeling."""
    
    def __init__(self, pad_label_id):
        super(ICSLLoss, self).__init__()
        self.intent_loss = nn.CrossEntropyLoss()
        self.slot_loss = nn.CrossEntropyLoss(ignore_index=pad_label_id)
    
    def forward(self, intent_logits, slot_logits, intent_labels, slot_labels, valid_lengths):
        """Compute combined loss."""
        # Intent loss
        intent_loss = self.intent_loss(intent_logits, intent_labels)
        
        # Slot loss (flatten for cross entropy)
        batch_size, seq_len, num_labels = slot_logits.size()
        slot_logits_flat = slot_logits.view(-1, num_labels)
        slot_labels_flat = slot_labels[:, 1:].contiguous().view(-1)  # Skip [CLS] label
        
        slot_loss = self.slot_loss(slot_logits_flat, slot_labels_flat)
        
        return intent_loss + slot_loss


def lstm_collate_fn(batch, pad_token_id=0, pad_label_id=0):
    """Collate function for LSTM DataLoader."""
    max_len = max(item['valid_length'] for item in batch)
    
    example_ids = []
    token_ids = []
    label_ids = []
    intent_ids = []
    valid_lengths = []
    words_list = []
    
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
        words_list.append(item['words'])
    
    return {
        'example_ids': example_ids,
        'token_ids': torch.tensor(token_ids, dtype=torch.long),
        'label_ids': torch.tensor(label_ids, dtype=torch.long),
        'intent_ids': torch.tensor(intent_ids, dtype=torch.long),
        'valid_lengths': torch.tensor(valid_lengths, dtype=torch.long),
        'words': words_list
    }


def train(model_name, train_input, dev_input, vocab, intent2idx, label2idx, epochs=None):
    """Training function. `epochs` overrides the default 50 if provided."""
    # Hyperparameters (optimized for synthetic data)
    log_interval = 100
    batch_size = 32
    lr = 1e-3  # Optimal for LSTM with random embeddings
    if epochs is None:
        epochs = 50  # default; was hardcoded prior to this change
    
    log.info(f"Training configuration: batch_size={batch_size}, lr={lr}, epochs={epochs}")
    log.info(f"Vocabulary size: {len(vocab)}")
    log.info(f"Number of intents: {len(intent2idx)}, Number of slot labels: {len(label2idx)}")
    
    # Create datasets
    train_dataset = LSTMDataset(train_input, vocab, label2idx, intent2idx)
    dev_dataset = LSTMDataset(dev_input, vocab, label2idx, intent2idx)
    
    log.info(f"Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")
    
    pad_label_id = label2idx[PAD]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: lstm_collate_fn(b, pad_token_id=0, pad_label_id=pad_label_id)
    )
    
    # Initialize model
    model = ICSL(
        vocab_size=len(vocab),
        num_slot_labels=len(label2idx),
        num_intents=len(intent2idx)
    ).to(device)
    
    # Loss and optimizer
    loss_fn = ICSLLoss(pad_label_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_score = (0, 0)  # (intent_acc, slot_f1)
    epoch_tic = time.time()
    total_num = 0
    
    for epoch in range(epochs):
        model.train()
        step_loss = 0
        log_num = 0
        tic = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            token_ids = batch['token_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            intent_ids = batch['intent_ids'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            # Forward pass
            intent_logits, slot_logits = model(token_ids)
            loss = loss_fn(intent_logits, slot_logits, intent_ids, label_ids, valid_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step_loss += loss.item()
            log_num += len(token_ids)
            total_num += len(token_ids)
            
            if (batch_idx + 1) % log_interval == 0:
                toc = time.time()
                # Compute batch metrics
                intent_preds = intent_logits.argmax(dim=-1).cpu().numpy()
                intent_labels_np = intent_ids.cpu().numpy()
                intent_acc = accuracy_score(intent_labels_np, intent_preds)
                
                log.info(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                        f'speed: {log_num / (toc - tic):.2f} samples/s, '
                        f'loss={step_loss / log_interval:.4f}, intent_acc={intent_acc:.3f}')
                tic = time.time()
                step_loss = 0
                log_num = 0
        
        # Evaluate on dev set
        log.info('Evaluate on development set:')
        intent_acc, slot_f1 = evaluate(model, dev_input, vocab, intent2idx, label2idx)
        
        # Save best model
        if slot_f1 > best_score[1]:
            best_score = (intent_acc, slot_f1)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'intent2idx': intent2idx,
                'label2idx': label2idx
            }, os.path.join(model_dir, f'{model_name}.pt'))
            log.info(f'New best model saved with slot_f1={slot_f1:.4f}')
    
    epoch_toc = time.time()
    log.info(f'Training complete. Time: {epoch_toc - epoch_tic:.2f}s, '
            f'Speed: {total_num / (epoch_toc - epoch_tic):.2f} samples/s')
    log.info(f'Best dev scores: intent_acc={best_score[0]:.4f}, slot_f1={best_score[1]:.4f}')
    
    return model


def evaluate(model, eval_input, vocab, intent2idx, label2idx, model_path=None):
    """Evaluate model on a dataset."""
    # Load model if not provided
    if model is None:
        assert model_path is not None
        checkpoint = torch.load(model_path)
        vocab = checkpoint['vocab']
        intent2idx = checkpoint['intent2idx']
        label2idx = checkpoint['label2idx']
        
        model = ICSL(
            vocab_size=len(vocab),
            num_slot_labels=len(label2idx),
            num_intents=len(intent2idx)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Create reverse mappings
    idx2label = {v: k for k, v in label2idx.items()}
    idx2intent = {v: k for k, v in intent2idx.items()}
    
    # Load dataset
    eval_dataset = LSTMDataset(eval_input, vocab, label2idx, intent2idx)
    pad_label_id = label2idx[PAD]
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: lstm_collate_fn(b, pad_token_id=0, pad_label_id=pad_label_id)
    )
    
    # Collect predictions
    all_intent_preds = []
    all_intent_labels = []
    all_slot_preds = []
    all_slot_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            token_ids = batch['token_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            intent_ids = batch['intent_ids'].to(device)
            valid_lengths = batch['valid_lengths']
            words_list = batch['words']
            
            intent_logits, slot_logits = model(token_ids)
            
            # Intent predictions
            intent_preds = intent_logits.argmax(dim=-1).cpu().numpy()
            all_intent_preds.extend(intent_preds.tolist())
            all_intent_labels.extend(intent_ids.cpu().numpy().tolist())
            
            # Slot predictions
            slot_preds = slot_logits.argmax(dim=-1).cpu().numpy()
            slot_labels_np = label_ids[:, 1:].cpu().numpy()  # Skip [CLS]
            
            for i, (pred, label, length, words) in enumerate(
                zip(slot_preds, slot_labels_np, valid_lengths, words_list)
            ):
                # Convert to labels (exclude [CLS] and [SEP])
                length = len(words)  # Use original word count
                pred_labels = [idx2label.get(p, 'O') for p in pred[:length]]
                gold_labels = [idx2label.get(l, 'O') for l in label[:length]]
                
                all_slot_preds.append(pred_labels)
                all_slot_labels.append(gold_labels)
    
    # Compute intent accuracy
    intent_acc = accuracy_score(all_intent_labels, all_intent_preds)
    log.info(f"Intent Accuracy: {intent_acc:.4f}")
    
    # Compute slot F1 using CoNLL evaluation
    results = evaluate_conll(all_slot_labels, all_slot_preds)
    slot_f1 = results['f1']
    
    log.info(f"Slot F1: {slot_f1:.4f}")
    log.info(format_conll_output(results))
    
    # Write predictions to file for reference
    example_ids, utterances, labels, intents = load_tsv(eval_input)
    os.makedirs(os.path.dirname(conll_prediction_file) or '.', exist_ok=True)
    with open(conll_prediction_file, 'w', encoding='utf-8') as f:
        for i, (utterance, gold_labels) in enumerate(zip(utterances, labels)):
            if i < len(all_slot_preds):
                pred_labels = all_slot_preds[i]
                for j, (word, gold, pred) in enumerate(zip(utterance, gold_labels, pred_labels)):
                    f.write(f'{word} {gold} {pred}\n')
                f.write('\n')
    
    return intent_acc, slot_f1
