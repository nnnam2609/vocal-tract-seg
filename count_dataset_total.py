"""
Count actual dataset statistics from converted YOLO dataset
"""
from pathlib import Path
from collections import defaultdict

# Mapping subject IDs to names
id_to_subject = {
    '1612': 'S1',
    '1617': 'S2', 
    '1618': 'S3',
    '1628': 'S4',
    '1635': 'S5',
    '1638': 'S6',
    '1640': 'S7',
    '1653': 'S8',
    '1659': 'S9',
    '1662': 'S7.1',
    '1775': 'S7.2'
}

# Database mapping
id_to_db = {
    '1612': 'ASD1', '1617': 'ASD1', '1618': 'ASD1', '1628': 'ASD1',
    '1635': 'ASD1', '1638': 'ASD1', '1640': 'ASD1', '1653': 'ASD1',
    '1659': 'ASD1', '1662': 'ASD1', '1775': 'ASD2'
}

def count_images():
    data_dir = Path('data_yolo/labels')
    counts = defaultdict(lambda: {'train': 0, 'valid': 0, 'test': 0})
    
    for split in ['train', 'valid', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        for label_file in split_dir.glob('*.txt'):
            # Extract subject ID from filename
            filename = label_file.name
            for subj_id in id_to_subject.keys():
                if subj_id in filename:
                    counts[subj_id][split] += 1
                    break
    
    return counts

# Count images
counts = count_images()

# Print results
print("\nAnalyzing dataset split...\n")
print("="*80)
print(f"{'Subject':<8} {'Database':<10} {'ID':<8} {'Train':<8} {'Valid':<8} {'Test':<8} {'Total':<8}")
print("="*80)

total_train = 0
total_valid = 0
total_test = 0

for subj_id in sorted(id_to_subject.keys(), key=lambda x: (id_to_subject[x].replace('.', '_'))):
    subject = id_to_subject[subj_id]
    db = id_to_db[subj_id]
    c = counts[subj_id]
    total = c['train'] + c['valid'] + c['test']
    
    total_train += c['train']
    total_valid += c['valid']
    total_test += c['test']
    
    print(f"{subject:<8} {db:<10} {subj_id:<8} {c['train']:<8} {c['valid']:<8} {c['test']:<8} {total:<8}")

print("="*80)
print(f"{'TOTAL':<27} {total_train:<8} {total_valid:<8} {total_test:<8} {total_train+total_valid+total_test:<8}")
print("="*80)
