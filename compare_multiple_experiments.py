#!/usr/bin/env python3
"""
Multi-Experiment Comparison Tool
Compare results across multiple experiments for each subject
Generates comparison tables showing performance metrics across different experimental configurations
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentData:
    """Data class to store experiment information"""
    exp_id: str
    csv_path: str
    df: pd.DataFrame
    label: str = None


class MultiExperimentLoader:
    """Class to load and manage multiple experiment results"""
    
    def __init__(self, experiment_paths: Dict[str, str]):
        """
        Initialize and load multiple CSV files
        
        Parameters:
        -----------
        experiment_paths : Dict[str, str]
            Dictionary mapping experiment IDs to CSV file paths
            Example: {'32': 'results/32/test_results.csv', '35': 'results/35/test_results.csv'}
        """
        self.experiments = {}
        self.subjects = []
        self.sequences = []
        self.articulators = []
        
        self._load_all_experiments(experiment_paths)
        self._extract_common_metadata()
    
    def _load_all_experiments(self, experiment_paths: Dict[str, str]):
        """Load all experiment CSV files"""
        print(f"\n{'='*80}")
        print(f"📂 LOADING MULTIPLE EXPERIMENTS")
        print(f"{'='*80}\n")
        
        for exp_id, csv_path in experiment_paths.items():
            try:
                df = pd.read_csv(csv_path)
                self.experiments[exp_id] = ExperimentData(
                    exp_id=exp_id,
                    csv_path=csv_path,
                    df=df,
                    label=f"Exp {exp_id}"
                )
                print(f"✅ Loaded Exp {exp_id}: {csv_path}")
                print(f"   • Rows: {len(df)}")
                print(f"   • Subjects: {sorted(df['subject'].unique())}")
                print(f"   • Sequences: {sorted(df['sequence'].unique())}")
            except FileNotFoundError:
                print(f"❌ Error: File not found - {csv_path}")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error loading {csv_path}: {e}")
                sys.exit(1)
        
        print(f"\n{'='*80}\n")
    
    def _extract_common_metadata(self):
        """Extract common metadata across all experiments"""
        all_subjects = set()
        all_sequences = set()
        all_articulators = set()
        
        for exp_data in self.experiments.values():
            all_subjects.update(exp_data.df['subject'].unique())
            all_sequences.update(exp_data.df['sequence'].unique())
            all_articulators.update(exp_data.df['pred_class'].unique())
        
        self.subjects = sorted(all_subjects)
        self.sequences = sorted(all_sequences)
        self.articulators = sorted(all_articulators)
        
        print(f"📊 COMMON METADATA:")
        print(f"  • Subjects found: {self.subjects}")
        print(f"  • Sequences found: {self.sequences}")
        print(f"  • Articulators: {self.articulators}")
        print(f"{'='*80}\n")
    
    def get_subject_data(self, subject, exp_id: str) -> pd.DataFrame:
        """Get data for a specific subject from a specific experiment"""
        if exp_id not in self.experiments:
            return pd.DataFrame()
        return self.experiments[exp_id].df[
            self.experiments[exp_id].df['subject'].astype(str) == str(subject)
        ].copy()


class MultiExperimentComparisonEngine:
    """Engine to perform multi-experiment comparisons"""
    
    def __init__(self, loader: MultiExperimentLoader):
        """Initialize with multi-experiment loader"""
        self.loader = loader
    
    def compare_subject_across_experiments(self, subject) -> Dict:
        """
        Compare a subject's performance across all experiments
        
        Parameters:
        -----------
        subject : int or str
            Subject ID to compare
        
        Returns:
        --------
        Dict containing comparison results across experiments
        """
        results = {
            'subject': subject,
            'experiments': list(self.loader.experiments.keys()),
            'exp_labels': [exp.label for exp in self.loader.experiments.values()],
            'p2cp_by_articulator': {},
            'jaccard_by_articulator': {},
            'overall_p2cp': {},
            'overall_jaccard': {},
            'counts': {}
        }
        
        # Collect data for each experiment
        exp_data_list = []
        for exp_id in self.loader.experiments.keys():
            df = self.loader.get_subject_data(subject, exp_id)
            exp_data_list.append((exp_id, df))
            results['counts'][exp_id] = len(df)
        
        # Check if subject exists in any experiment
        if all(len(df) == 0 for _, df in exp_data_list):
            print(f"Warning: Subject {subject} not found in any experiment")
            return None
        
        # Compare per articulator
        for articulator in self.loader.articulators:
            results['p2cp_by_articulator'][articulator] = {}
            results['jaccard_by_articulator'][articulator] = {}
            
            for exp_id, df in exp_data_list:
                art_data = df[df['pred_class'] == articulator]
                
                # P2CP statistics
                p2cp_values = art_data['p2cp_rms'].dropna()
                if len(p2cp_values) > 0:
                    results['p2cp_by_articulator'][articulator][exp_id] = {
                        'mean': p2cp_values.mean(),
                        'std': p2cp_values.std(),
                        'count': len(p2cp_values)
                    }
                
                # Jaccard statistics
                jacc_values = art_data['jaccard_index'].dropna()
                if len(jacc_values) > 0:
                    results['jaccard_by_articulator'][articulator][exp_id] = {
                        'mean': jacc_values.mean(),
                        'std': jacc_values.std(),
                        'count': len(jacc_values)
                    }
        
        # Overall statistics across all articulators
        for exp_id, df in exp_data_list:
            # P2CP overall
            p2cp_all = df['p2cp_rms'].dropna()
            if len(p2cp_all) > 0:
                results['overall_p2cp'][exp_id] = {
                    'mean': p2cp_all.mean(),
                    'std': p2cp_all.std(),
                    'count': len(p2cp_all)
                }
            
            # Jaccard overall
            jacc_all = df['jaccard_index'].dropna()
            if len(jacc_all) > 0:
                results['overall_jaccard'][exp_id] = {
                    'mean': jacc_all.mean(),
                    'std': jacc_all.std(),
                    'count': len(jacc_all)
                }
        
        return results
    
    def compare_all_subjects(self) -> List[Dict]:
        """Compare all subjects across all experiments"""
        all_results = []
        
        for subject in self.loader.subjects:
            result = self.compare_subject_across_experiments(subject)
            if result:
                all_results.append(result)
        
        return all_results


class MultiExperimentTableFormatter:
    """Class to format and print multi-experiment comparison tables"""
    
    @staticmethod
    def print_text_table(results: Dict):
        """Print text format table for multi-experiment comparison"""
        subject = results['subject']
        exp_ids = results['experiments']
        exp_labels = results['exp_labels']
        
        print(f"\n{'='*120}")
        print(f"SUBJECT {subject} - COMPARISON ACROSS EXPERIMENTS")
        print(f"{'='*120}\n")
        
        # Build header dynamically based on number of experiments
        header_parts = [f"{'Articulator':<25}"]
        header_parts.append(f"{'P2CP_RMS (mm)':<{20*len(exp_ids)}}")
        header_parts.append(f"{'Jaccard Index':<{20*len(exp_ids)}}")
        print("".join(header_parts))
        
        # Sub-header with experiment labels
        subheader_parts = [f"{'':<25}"]
        for exp_label in exp_labels:
            subheader_parts.append(f"{exp_label:<20}")
        for exp_label in exp_labels:
            subheader_parts.append(f"{exp_label:<20}")
        print("".join(subheader_parts))
        
        print("-" * 120)
        
        # Per articulator rows
        for articulator in sorted(results['p2cp_by_articulator'].keys()):
            art_name = articulator.replace('-', ' ').title()
            row_parts = [f"{art_name:<25}"]
            
            # P2CP columns
            for exp_id in exp_ids:
                if exp_id in results['p2cp_by_articulator'][articulator]:
                    stats = results['p2cp_by_articulator'][articulator][exp_id]
                    value_str = f"{stats['mean']:.2f}±{stats['std']:.2f}"
                else:
                    value_str = "-"
                row_parts.append(f"{value_str:<20}")
            
            # Jaccard columns
            for exp_id in exp_ids:
                if exp_id in results['jaccard_by_articulator'][articulator]:
                    stats = results['jaccard_by_articulator'][articulator][exp_id]
                    value_str = f"{stats['mean']:.2f}±{stats['std']:.2f}"
                else:
                    value_str = "-"
                row_parts.append(f"{value_str:<20}")
            
            print("".join(row_parts))
        
        # Overall row
        print("-" * 120)
        row_parts = [f"{'OVERALL':<25}"]
        
        # P2CP overall
        for exp_id in exp_ids:
            if exp_id in results['overall_p2cp']:
                stats = results['overall_p2cp'][exp_id]
                value_str = f"{stats['mean']:.2f}±{stats['std']:.2f}"
            else:
                value_str = "-"
            row_parts.append(f"{value_str:<20}")
        
        # Jaccard overall
        for exp_id in exp_ids:
            if exp_id in results['overall_jaccard']:
                stats = results['overall_jaccard'][exp_id]
                value_str = f"{stats['mean']:.2f}±{stats['std']:.2f}"
            else:
                value_str = "-"
            row_parts.append(f"{value_str:<20}")
        
        print("".join(row_parts))
        
        # Summary
        print(f"\n{'='*120}")
        print(f"Sample counts per experiment:")
        for exp_id, exp_label in zip(exp_ids, exp_labels):
            count = results['counts'].get(exp_id, 0)
            print(f"  {exp_label}: {count} measurements")
        print(f"{'='*120}\n")
    
    @staticmethod
    def print_latex_table(results: Dict):
        """Print LaTeX format table for multi-experiment comparison"""
        subject = results['subject']
        exp_ids = results['experiments']
        exp_labels = results['exp_labels']
        
        # Build column specification
        n_exps = len(exp_ids)
        col_spec = "l" + "c" * n_exps + "c" * n_exps
        
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\small")
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print("\\hline")
        
        # Header
        header_parts = ["& \\multicolumn{" + str(n_exps) + "}{c}{P2CP$_{RMS}$ (mm)}"]
        header_parts.append("& \\multicolumn{" + str(n_exps) + "}{c}{Jaccard index} \\\\")
        print("".join(header_parts))
        
        # Sub-header
        subheader = "Articulator"
        for exp_label in exp_labels:
            subheader += f" & {exp_label}"
        for exp_label in exp_labels:
            subheader += f" & {exp_label}"
        subheader += " \\\\"
        print(subheader)
        print("\\hline")
        
        # Per articulator rows
        for articulator in sorted(results['p2cp_by_articulator'].keys()):
            art_name = articulator.replace('-', ' ').title()
            if 'Midline' in art_name:
                art_name = art_name.replace('Soft Palate Midline', 'Soft Palate\\newline Center line')
            
            row = art_name
            
            # P2CP columns
            for exp_id in exp_ids:
                if exp_id in results['p2cp_by_articulator'][articulator]:
                    stats = results['p2cp_by_articulator'][articulator][exp_id]
                    row += f" & {stats['mean']:.2f} $\\pm$ {stats['std']:.2f}"
                else:
                    row += " & -"
            
            # Jaccard columns
            for exp_id in exp_ids:
                if exp_id in results['jaccard_by_articulator'][articulator]:
                    stats = results['jaccard_by_articulator'][articulator][exp_id]
                    row += f" & {stats['mean']:.2f} $\\pm$ {stats['std']:.2f}"
                else:
                    row += " & -"
            
            row += " \\\\"
            print(row)
        
        # Overall row
        print("\\hline")
        row = "mean $\\pm$ std"
        
        # P2CP overall
        for exp_id in exp_ids:
            if exp_id in results['overall_p2cp']:
                stats = results['overall_p2cp'][exp_id]
                row += f" & {stats['mean']:.2f} $\\pm$ {stats['std']:.2f}"
            else:
                row += " & -"
        
        # Jaccard overall
        for exp_id in exp_ids:
            if exp_id in results['overall_jaccard']:
                stats = results['overall_jaccard'][exp_id]
                row += f" & {stats['mean']:.2f} $\\pm$ {stats['std']:.2f}"
            else:
                row += " & -"
        
        row += " \\\\"
        print(row)
        
        print("\\hline")
        print("\\end{tabular}")
        print(f"\\caption{{Subject {subject} comparison across experiments}}")
        print(f"\\label{{tab:subject_{subject}_multi_exp}}")
        print("\\end{table}")
    
    @staticmethod
    def print_markdown_table(results: Dict):
        """Print Markdown format table for multi-experiment comparison"""
        subject = results['subject']
        exp_ids = results['experiments']
        exp_labels = results['exp_labels']
        
        print(f"\n## Subject {subject} - Multi-Experiment Comparison\n")
        
        # Build header
        header = "| Articulator |"
        separator = "|-------------|"
        
        for exp_label in exp_labels:
            header += f" P2CP ({exp_label}) |"
            separator += "------------------|"
        for exp_label in exp_labels:
            header += f" Jaccard ({exp_label}) |"
            separator += "------------------|"
        
        print(header)
        print(separator)
        
        # Per articulator rows
        for articulator in sorted(results['p2cp_by_articulator'].keys()):
            art_name = articulator.replace('-', ' ').title()
            row = f"| {art_name} |"
            
            # P2CP columns
            for exp_id in exp_ids:
                if exp_id in results['p2cp_by_articulator'][articulator]:
                    stats = results['p2cp_by_articulator'][articulator][exp_id]
                    row += f" {stats['mean']:.2f} ± {stats['std']:.2f} |"
                else:
                    row += " - |"
            
            # Jaccard columns
            for exp_id in exp_ids:
                if exp_id in results['jaccard_by_articulator'][articulator]:
                    stats = results['jaccard_by_articulator'][articulator][exp_id]
                    row += f" {stats['mean']:.2f} ± {stats['std']:.2f} |"
                else:
                    row += " - |"
            
            print(row)
        
        # Overall row
        row = "| **OVERALL** |"
        
        # P2CP overall
        for exp_id in exp_ids:
            if exp_id in results['overall_p2cp']:
                stats = results['overall_p2cp'][exp_id]
                row += f" **{stats['mean']:.2f} ± {stats['std']:.2f}** |"
            else:
                row += " - |"
        
        # Jaccard overall
        for exp_id in exp_ids:
            if exp_id in results['overall_jaccard']:
                stats = results['overall_jaccard'][exp_id]
                row += f" **{stats['mean']:.2f} ± {stats['std']:.2f}** |"
            else:
                row += " - |"
        
        print(row)
        
        # Summary
        print(f"\n**Sample counts:**")
        for exp_id, exp_label in zip(exp_ids, exp_labels):
            count = results['counts'].get(exp_id, 0)
            print(f"- {exp_label}: {count} measurements")


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-Experiment Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare experiments 32, 35, 37, 38 (auto-detect paths in results/)
  python %(prog)s 32 35 37 38
  
  # Specify custom paths
  python %(prog)s --exp 32 results/32/test_results.csv --exp 35 results/35/test_results.csv
  
  # Compare specific subject across experiments
  python %(prog)s 32 35 37 38 --subject 1640
  
  # Generate LaTeX tables for all subjects
  python %(prog)s 32 35 37 38 -f latex
  
  # Save output to file
  python %(prog)s 32 35 37 38 -o comparison_results.txt
        """
    )
    
    parser.add_argument('exp_ids', nargs='*',
                        help='Experiment IDs to compare (e.g., 32 35 37 38)')
    parser.add_argument('--exp', '-e', action='append', nargs=2,
                        metavar=('EXP_ID', 'CSV_PATH'),
                        help='Specify experiment ID and CSV path manually')
    parser.add_argument('--results-dir', '-r',
                        default='results',
                        help='Base results directory (default: results/)')
    parser.add_argument('--subject', '-s',
                        help='Compare only specific subject')
    parser.add_argument('--format', '-f',
                        choices=['text', 'latex', 'markdown'],
                        default='text',
                        help='Output format (default: text)')
    parser.add_argument('--output', '-o',
                        help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Build experiment paths dictionary
    experiment_paths = {}
    
    # Method 1: Manual specification via --exp
    if args.exp:
        for exp_id, csv_path in args.exp:
            experiment_paths[exp_id] = csv_path
    
    # Method 2: Auto-detect from results directory
    elif args.exp_ids:
        results_dir = Path(args.results_dir)
        for exp_id in args.exp_ids:
            csv_path = results_dir / exp_id / 'test_results.csv'
            if csv_path.exists():
                experiment_paths[exp_id] = str(csv_path)
            else:
                print(f"Warning: {csv_path} not found")
    
    else:
        parser.print_help()
        sys.exit(1)
    
    if not experiment_paths:
        print("Error: No valid experiment paths found")
        sys.exit(1)
    
    # Redirect output if specified
    if args.output:
        sys.stdout = open(args.output, 'w')
    
    try:
        # Load all experiments
        loader = MultiExperimentLoader(experiment_paths)
        engine = MultiExperimentComparisonEngine(loader)
        formatter = MultiExperimentTableFormatter()
        
        # Compare subjects
        if args.subject:
            # Compare single subject
            try:
                subject_id = int(args.subject)
            except ValueError:
                subject_id = args.subject
            
            results = engine.compare_subject_across_experiments(subject_id)
            
            if results:
                if args.format == 'latex':
                    formatter.print_latex_table(results)
                elif args.format == 'markdown':
                    formatter.print_markdown_table(results)
                else:
                    formatter.print_text_table(results)
        else:
            # Compare all subjects
            all_results = engine.compare_all_subjects()
            
            for results in all_results:
                if args.format == 'latex':
                    formatter.print_latex_table(results)
                elif args.format == 'markdown':
                    formatter.print_markdown_table(results)
                else:
                    formatter.print_text_table(results)
    
    finally:
        if args.output:
            sys.stdout.close()
            print(f"\n✅ Comparison report saved to: {args.output}", file=sys.__stdout__)


if __name__ == '__main__':
    main()
