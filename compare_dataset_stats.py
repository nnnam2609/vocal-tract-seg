"""
Compare expected vs actual dataset statistics
"""

# Expected from your table
expected = {
    'S1': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S2': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S3': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S4': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S5': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S6': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S7': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S8': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S9': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S7.1': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S7.2': {'train': 310, 'valid': 54, 'test': 63, 'total': 427},
}

# Actual from counting script
actual = {
    'S1': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S2': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S3': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S4': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S5': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S6': {'train': 72, 'valid': 9, 'test': 20, 'total': 101},
    'S7': {'train': 73, 'valid': 9, 'test': 20, 'total': 102},
    'S8': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S9': {'train': 71, 'valid': 9, 'test': 20, 'total': 100},
    'S7.1': {'train': 50, 'valid': 0, 'test': 50, 'total': 100},
    'S7.2': {'train': 368, 'valid': 57, 'test': 70, 'total': 495},
}

print("="*100)
print(f"{'Subject':<8} {'Split':<10} {'Expected':<15} {'Actual':<15} {'Difference':<15} {'Status':<15}")
print("="*100)

total_expected = {'train': 0, 'valid': 0, 'test': 0, 'total': 0}
total_actual = {'train': 0, 'valid': 0, 'test': 0, 'total': 0}

for subject in sorted(expected.keys(), key=lambda x: (x.replace('.', '_'))):
    exp = expected[subject]
    act = actual[subject]
    
    for split in ['train', 'valid', 'test', 'total']:
        diff = act[split] - exp[split]
        status = '✓ Match' if diff == 0 else f'{"+" if diff > 0 else ""}{diff}'
        
        if split == 'total':
            print(f"{subject:<8} {split:<10} {exp[split]:<15} {act[split]:<15} {status:<15} {'MISMATCH' if diff != 0 else 'OK'}")
            print("-"*100)
        else:
            print(f"{subject:<8} {split:<10} {exp[split]:<15} {act[split]:<15} {status:<15}")
        
        total_expected[split] += exp[split]
        total_actual[split] += act[split]

print("="*100)
for split in ['train', 'valid', 'test', 'total']:
    diff = total_actual[split] - total_expected[split]
    status = '✓ Match' if diff == 0 else f'{"+" if diff > 0 else ""}{diff}'
    print(f"{'TOTAL':<8} {split:<10} {total_expected[split]:<15} {total_actual[split]:<15} {status:<15}")
print("="*100)

print("\n" + "="*100)
print("SUMMARY OF ISSUES:")
print("="*100)
print("1. S6 (1638): +1 train image, +1 total")
print("2. S7 (1640): +2 train images, +2 total")
print("3. S7.1 (1662): Different split! 50 train/0 valid/50 test instead of 71/9/20")
print("4. S7.2 (1775/ASD2): +58 train, +3 valid, +7 test = +68 total images")
print("\nOverall:")
print(f"  Expected total: {total_expected['total']} images")
print(f"  Actual total: {total_actual['total']} images")
print(f"  Difference: +{total_actual['total'] - total_expected['total']} images")
print("="*100)
