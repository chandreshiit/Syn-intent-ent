"""
Analyze the banking commands dataset for quality issues.
"""
import csv
from collections import Counter, defaultdict
import re

# Load dataset
data = []
with open('banking_commands_dataset_v2.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print('='*70)
print('ISSUE #1: EXACT DUPLICATES')
print('='*70)
all_commands = [r['command'].lower().strip() for r in data]
duplicates = [cmd for cmd, count in Counter(all_commands).items() if count > 1]
print(f'Exact duplicate commands: {len(duplicates)}')

# Show some duplicates with their intents
cmd_to_intents = defaultdict(list)
for r in data:
    cmd_to_intents[r['command'].lower().strip()].append(r['intent'])

cross_intent_dups = {cmd: intents for cmd, intents in cmd_to_intents.items() if len(set(intents)) > 1}
print(f'Commands appearing with MULTIPLE intents: {len(cross_intent_dups)}')
if cross_intent_dups:
    print('\nEXAMPLES (same command, different intents):')
    for cmd, intents in list(cross_intent_dups.items())[:10]:
        print(f'  "{cmd[:60]}..." -> {set(intents)}')

print()
print('='*70)
print('ISSUE #2: SEMANTIC CONFUSION (Wrong Intent Assignment)')
print('='*70)

# Check for semantic leakage - commands that mention wrong intent keywords
confusion_patterns = {
    'balance_enquiry': ['balance', 'how much money', 'check my balance', 'account balance'],
    'branch_address': ['branch', 'address', 'location', 'where is', 'nearest', 'directions'],
    'block': ['block my card', 'block it', 'block immediately'],
    'lost': ['lost my card', 'lost my', 'cannot find my card', 'misplaced'],
    'activate_card': ['activate', 'activation'],
    'card_issue': ['not working', 'declined', 'issue with', 'problem with'],
    'ifsc_code': ['ifsc', 'ifsc code'],
    'outstanding_balance': ['outstanding', 'owe', 'dues', 'credit card balance'],
    'generate_pin': ['pin', 'generate pin', 'new pin', 'atm pin'],
    'unauthorised_transaction': ['unauthorized', 'fraudulent', 'did not make', 'suspicious'],
    'loan_query': ['loan', 'interest rate', 'emi', 'eligibility'],
    'change_limit': ['limit', 'increase limit', 'change limit', 'withdrawal limit'],
    'dispatch_status': ['dispatch', 'status', 'when will i receive', 'been sent'],
    'past_transactions': ['transaction history', 'past transactions', 'recent', 'last.*transactions']
}

misclassified = defaultdict(list)
for r in data:
    cmd_lower = r['command'].lower()
    actual_intent = r['intent']
    
    for expected_intent, patterns in confusion_patterns.items():
        if expected_intent != actual_intent:
            for pattern in patterns:
                if re.search(pattern, cmd_lower):
                    misclassified[actual_intent].append((r['command'][:70], expected_intent, pattern))
                    break

total_suspicious = sum(len(v) for v in misclassified.values())
print(f'Total suspicious samples (keyword suggests different intent): {total_suspicious}')
print(f'Percentage: {total_suspicious/len(data)*100:.1f}%')

print('\nBreakdown by intent (commands with WRONG intent labels):')
for intent, issues in sorted(misclassified.items(), key=lambda x: -len(x[1])):
    print(f'\n{intent} ({len(issues)} suspicious samples):')
    for cmd, expected, pattern in issues[:5]:
        print(f'  "{cmd}..."')
        print(f'    -> Contains "{pattern}" (suggests {expected})')

print()
print('='*70)
print('ISSUE #3: SAMPLE SPECIFIC ANALYSIS FOR branch_address')
print('='*70)

# Deep dive into branch_address since we saw balance queries there
branch_data = [r for r in data if r['intent'] == 'branch_address']
print(f'Total branch_address samples: {len(branch_data)}')

# Count how many are actually about balance
balance_in_branch = 0
other_issues = defaultdict(int)
for r in branch_data:
    cmd = r['command'].lower()
    if 'balance' in cmd or 'how much money' in cmd:
        balance_in_branch += 1
    elif 'loan' in cmd:
        other_issues['loan_query'] += 1
    elif 'transaction' in cmd and 'history' not in cmd:
        other_issues['past_transactions'] += 1
        
print(f'Contains "balance/money" queries: {balance_in_branch} ({balance_in_branch/len(branch_data)*100:.1f}%)')
print(f'Other misclassifications: {dict(other_issues)}')

# Sample some balance queries labeled as branch_address
print('\nSample balance queries MISLABELED as branch_address:')
count = 0
for r in branch_data:
    cmd = r['command'].lower()
    if 'balance' in cmd or 'how much money' in cmd:
        print(f'  - {r["command"]}')
        count += 1
        if count >= 10:
            break

print()
print('='*70)
print('ISSUE #4: TEXT LENGTH & DIVERSITY ANALYSIS')
print('='*70)

# Length analysis
lengths = [len(r['command'].split()) for r in data]
print(f'Average command length: {sum(lengths)/len(lengths):.1f} words')
print(f'Min length: {min(lengths)} words')
print(f'Max length: {max(lengths)} words')

# Check for very short commands (might lack context)
short_cmds = [r for r in data if len(r['command'].split()) <= 3]
print(f'Very short commands (<=3 words): {len(short_cmds)} ({len(short_cmds)/len(data)*100:.1f}%)')

# Vocabulary analysis per intent
print('\nUnique first words per intent (shows diversity):')
for intent in sorted(set(r['intent'] for r in data)):
    intent_data = [r for r in data if r['intent'] == intent]
    first_words = set(r['command'].split()[0].lower() for r in intent_data if r['command'].split())
    print(f'  {intent}: {len(first_words)} unique first words')

print()
print('='*70)
print('SUMMARY & RECOMMENDATIONS')
print('='*70)
print(f'''
FINDINGS:
1. Dataset is BALANCED (850 samples per intent) - NOT the problem
2. {len(cross_intent_dups)} commands appear with MULTIPLE different intents - CRITICAL ISSUE
3. {total_suspicious} samples ({total_suspicious/len(data)*100:.1f}%) have semantic confusion - MAJOR ISSUE
4. The branch_address intent has {balance_in_branch} balance queries mislabeled

ROOT CAUSE: Same as MultiATIS issue - LLM generates semantically incorrect samples
because it was told the intent, not WHAT that intent should look like.

IMPACT ON MODEL:
- Model learns conflicting patterns (same text -> different labels)
- Confusion matrix will show cross-intent errors
- F1 and accuracy will be artificially low

RECOMMENDED FIX:
Use the SEED PATTERN approach (like we did for MultiATIS):
1. Define 10+ seed patterns per intent
2. Have LLM generate VARIATIONS of those patterns only
3. This ensures semantic consistency
''')
