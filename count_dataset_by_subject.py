"""
Count images per subject according to the current config split
"""
import yaml
import os
from pathlib import Path

# Subject mapping
SUBJECT_MAPPING = {
    'S1': '1612',
    'S2': '1617',
    'S3': '1618',
    'S4': '1628',
    'S5': '1635',
    'S6': '1638',
    'S7': '1640',
    'S8': '1653',
    'S9': '1659',
    'S7.1': '1662',
    'S7.2': '1775'
}

def count_images_in_sequence(datadir, subject_path, sequence, image_folder='NPY_MR', image_ext='npy'):
    """Count images with ROI annotations in a sequence"""
    seq_dir = os.path.join(datadir, subject_path, sequence)
    image_dir = os.path.join(seq_dir, image_folder)
    contours_dir = os.path.join(seq_dir, "contours")
    
    if not os.path.exists(image_dir) or not os.path.exists(contours_dir):
        return 0
    
    # Get all images
    image_files = [f for f in os.listdir(image_dir) if f.endswith(f'.{image_ext}')]
    
    # Count images that have at least one ROI annotation
    count = 0
    for img_file in image_files:
        instance_number = int(os.path.splitext(img_file)[0])
        # Check if any ROI file exists for this image
        roi_files = [f for f in os.listdir(contours_dir) 
                    if f.startswith(f'{instance_number:04d}_') and f.endswith('.roi')]
        if roi_files:
            count += 1
    
    return count

def analyze_config(config_path):
    """Analyze the dataset split from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    datadir = config['datadir']
    image_folder = config.get('image_folder', 'dicoms')
    image_ext = config.get('image_ext', 'dcm')
    
    # Reverse mapping for display
    reverse_mapping = {v: k for k, v in SUBJECT_MAPPING.items()}
    
    # Count per subject
    subject_counts = {}
    
    for split in ['train', 'valid', 'test']:
        split_key = f'{split}_sequences'
        if split_key not in config:
            continue
        
        for subject_path, sequences in config[split_key].items():
            # Extract subject ID from path
            subject_id = subject_path.split('/')[-1]
            subject_name = reverse_mapping.get(subject_id, f'Unknown-{subject_id}')
            
            # Determine database
            if 'ArtSpeech_Vocal_Tract_Segmentation' in subject_path:
                database = 'ASD1'
            elif 'ArtSpeech_Database_2' in subject_path:
                database = 'ASD2'
            else:
                database = 'Unknown'
            
            if subject_name not in subject_counts:
                subject_counts[subject_name] = {
                    'database': database,
                    'subject_id': subject_id,
                    'train': 0,
                    'valid': 0,
                    'test': 0
                }
            
            # Count images in each sequence
            for seq in sequences:
                count = count_images_in_sequence(datadir, subject_path, seq, image_folder, image_ext)
                subject_counts[subject_name][split] += count
    
    return subject_counts

def main():
    config_path = 'config/Nam_exp_01082026/yolo_seg_train.yaml'
    
    print("Analyzing dataset split...\n")
    
    subject_counts = analyze_config(config_path)
    
    # Print results
    print("="*80)
    print(f"{'Subject':<8} {'Database':<10} {'ID':<6} {'Train':<8} {'Valid':<8} {'Test':<8} {'Total':<8}")
    print("="*80)
    
    total_train = 0
    total_valid = 0
    total_test = 0
    
    # Sort by subject name
    for subject_name in sorted(subject_counts.keys(), key=lambda x: (x.replace('.', '_'))):
        info = subject_counts[subject_name]
        train = info['train']
        valid = info['valid']
        test = info['test']
        total = train + valid + test
        
        total_train += train
        total_valid += valid
        total_test += test
        
        print(f"{subject_name:<8} {info['database']:<10} {info['subject_id']:<6} "
              f"{train:<8} {valid:<8} {test:<8} {total:<8}")
    
    print("="*80)
    print(f"{'TOTAL':<25} {total_train:<8} {total_valid:<8} {total_test:<8} {total_train+total_valid+total_test:<8}")
    print("="*80)

if __name__ == '__main__':
    main()
