"""
Analysis Script for Interactive Tester Results
==============================================

Analyzes and visualizes results from the enhanced interactive tester.

Usage:
    python analyze_results.py results.csv
    python analyze_results.py baseline.csv plr.csv --compare
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def load_results(filepath):
    """Load results from CSV"""
    df = pd.read_csv(filepath)
    return df


def print_summary(df, model_name=None):
    """Print summary statistics"""
    name = model_name or df['Model'].iloc[0] if 'Model' in df.columns else "Model"
    
    print(f"\n{'='*80}")
    print(f"  SUMMARY: {name}")
    print(f"{'='*80}")
    
    if 'Scenario' in df.columns:
        # Generalization test format
        print(f"\nScenarios tested: {len(df)}")
        print(f"\nOverall Performance:")
        print(f"  Average Reward: {df['Avg Reward'].mean():.2f} ± {df['Avg Reward'].std():.2f}")
        print(f"  Average Crash Rate: {df['Crash Rate'].mean():.1f}%")
        
        print(f"\nBest Scenario: {df.loc[df['Avg Reward'].idxmax(), 'Scenario']}")
        print(f"  Reward: {df['Avg Reward'].max():.2f}")
        
        print(f"\nWorst Scenario: {df.loc[df['Avg Reward'].idxmin(), 'Scenario']}")
        print(f"  Reward: {df['Avg Reward'].min():.2f}")
        print(f"  Crash Rate: {df.loc[df['Avg Reward'].idxmin(), 'Crash Rate']:.1f}%")
        
    else:
        # Episode-by-episode format
        print(f"\nTotal Episodes: {len(df)}")
        print(f"Average Reward: {df['Reward'].mean():.2f} ± {df['Reward'].std():.2f}")
        print(f"Min Reward: {df['Reward'].min():.2f}")
        print(f"Max Reward: {df['Reward'].max():.2f}")
        
        crashes = df['Crashed'].sum() if 'Crashed' in df.columns else 0
        print(f"Crash Rate: {100 * crashes / len(df):.1f}%")
        print(f"Average Episode Length: {df['Length'].mean():.1f}")


def plot_single_model(df, output_dir='plots'):
    """Create plots for a single model's results"""
    Path(output_dir).mkdir(exist_ok=True)
    
    if 'Scenario' in df.columns:
        # Generalization test results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Reward by scenario
        scenarios = df['Scenario']
        axes[0, 0].bar(range(len(scenarios)), df['Avg Reward'], color='steelblue', alpha=0.7)
        axes[0, 0].errorbar(range(len(scenarios)), df['Avg Reward'], 
                           yerr=df['Std Reward'], fmt='none', color='black', capsize=5)
        axes[0, 0].set_xticks(range(len(scenarios)))
        axes[0, 0].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Reward Across Scenarios')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Crash rate by scenario
        axes[0, 1].bar(range(len(scenarios)), df['Crash Rate'], color='crimson', alpha=0.7)
        axes[0, 1].set_xticks(range(len(scenarios)))
        axes[0, 1].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Crash Rate (%)')
        axes[0, 1].set_title('Crash Rate Across Scenarios')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward vs crash rate scatter
        axes[1, 0].scatter(df['Crash Rate'], df['Avg Reward'], s=100, alpha=0.6)
        for i, scenario in enumerate(scenarios):
            axes[1, 0].annotate(scenario, 
                               (df['Crash Rate'].iloc[i], df['Avg Reward'].iloc[i]),
                               fontsize=8, alpha=0.7)
        axes[1, 0].set_xlabel('Crash Rate (%)')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].set_title('Reward vs Crash Rate Trade-off')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary table
        axes[1, 1].axis('off')
        summary_data = [
            ['Metric', 'Value'],
            ['Scenarios', str(len(df))],
            ['Avg Reward', f"{df['Avg Reward'].mean():.2f}"],
            ['Avg Crash Rate', f"{df['Crash Rate'].mean():.1f}%"],
            ['Best Scenario', df.loc[df['Avg Reward'].idxmax(), 'Scenario']],
            ['Worst Scenario', df.loc[df['Avg Reward'].idxmin(), 'Scenario']],
        ]
        table = axes[1, 1].table(cellText=summary_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Summary Statistics')
        
    else:
        # Episode-by-episode results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Reward over time
        axes[0, 0].plot(df['Episode'], df['Reward'], alpha=0.5, linewidth=0.5)
        # Add moving average
        window = min(20, len(df) // 10)
        if window > 1:
            ma = df['Reward'].rolling(window=window).mean()
            axes[0, 0].plot(df['Episode'], ma, color='red', linewidth=2, label=f'{window}-ep MA')
            axes[0, 0].legend()
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Reward Over Episodes')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reward distribution
        axes[0, 1].hist(df['Reward'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(df['Reward'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {df["Reward"].mean():.2f}')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode length over time
        axes[1, 0].plot(df['Episode'], df['Length'], alpha=0.5, linewidth=0.5)
        if window > 1:
            ma_length = df['Length'].rolling(window=window).mean()
            axes[1, 0].plot(df['Episode'], ma_length, color='green', linewidth=2, label=f'{window}-ep MA')
            axes[1, 0].legend()
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Length (steps)')
        axes[1, 0].set_title('Episode Length Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Crashes visualization
        if 'Crashed' in df.columns:
            crash_points = df[df['Crashed'] == True]
            survival_points = df[df['Crashed'] == False]
            
            axes[1, 1].scatter(survival_points['Episode'], survival_points['Reward'], 
                             color='green', alpha=0.5, s=20, label='Survived')
            axes[1, 1].scatter(crash_points['Episode'], crash_points['Reward'], 
                             color='red', alpha=0.5, s=20, label='Crashed')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].set_title('Outcomes: Crashes vs Survivals')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/single_model_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plots saved to {output_path}")
    plt.close()


def compare_models(dfs, names, output_dir='plots'):
    """Compare multiple models"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Check if generalization test format
    if 'Scenario' in dfs[0].columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        scenarios = dfs[0]['Scenario'].values
        x = np.arange(len(scenarios))
        width = 0.8 / len(dfs)
        
        # Reward comparison
        for i, (df, name) in enumerate(zip(dfs, names)):
            offset = (i - len(dfs)/2 + 0.5) * width
            axes[0, 0].bar(x + offset, df['Avg Reward'], width, label=name, alpha=0.7)
        
        axes[0, 0].set_xlabel('Scenario')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Reward Comparison Across Scenarios')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Crash rate comparison
        for i, (df, name) in enumerate(zip(dfs, names)):
            offset = (i - len(dfs)/2 + 0.5) * width
            axes[0, 1].bar(x + offset, df['Crash Rate'], width, label=name, alpha=0.7)
        
        axes[0, 1].set_xlabel('Scenario')
        axes[0, 1].set_ylabel('Crash Rate (%)')
        axes[0, 1].set_title('Crash Rate Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(scenarios, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overall average comparison
        avg_rewards = [df['Avg Reward'].mean() for df in dfs]
        avg_crashes = [df['Crash Rate'].mean() for df in dfs]
        
        x_pos = np.arange(len(names))
        axes[1, 0].bar(x_pos, avg_rewards, color='steelblue', alpha=0.7)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].set_title('Overall Average Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(x_pos, avg_crashes, color='crimson', alpha=0.7)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Crash Rate (%)')
        axes[1, 1].set_title('Overall Average Crash Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
    else:
        # Episode-by-episode comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward distribution comparison
        for df, name in zip(dfs, names):
            axes[0, 0].hist(df['Reward'], bins=30, alpha=0.5, label=name, edgecolor='black')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot comparison
        data_for_box = [df['Reward'].values for df in dfs]
        bp = axes[0, 1].boxplot(data_for_box, labels=names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Reward Distribution (Box Plot)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        summary_stats = []
        for name, df in zip(names, dfs):
            summary_stats.append([
                name,
                f"{df['Reward'].mean():.2f}",
                f"{df['Reward'].std():.2f}",
                f"{df['Reward'].min():.2f}",
                f"{df['Reward'].max():.2f}"
            ])
        
        axes[1, 0].axis('off')
        table_data = [['Model', 'Mean', 'Std', 'Min', 'Max']] + summary_stats
        table = axes[1, 0].table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[1, 0].set_title('Summary Statistics Comparison')
        
        # Crash rate comparison
        if all('Crashed' in df.columns for df in dfs):
            crash_rates = [100 * df['Crashed'].sum() / len(df) for df in dfs]
            axes[1, 1].bar(names, crash_rates, color='crimson', alpha=0.7)
            axes[1, 1].set_ylabel('Crash Rate (%)')
            axes[1, 1].set_title('Crash Rate Comparison')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plots saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze interactive tester results')
    parser.add_argument('files', nargs='+', help='CSV files to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--names', nargs='+', help='Names for models (optional)')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    dfs = [load_results(f) for f in args.files]
    names = args.names or [Path(f).stem for f in args.files]
    
    if len(names) != len(dfs):
        names = [f"Model {i+1}" for i in range(len(dfs))]
    
    # Print summaries
    for df, name in zip(dfs, names):
        print_summary(df, name)
    
    # Create plots
    if args.compare and len(dfs) > 1:
        print("\n📊 Creating comparison plots...")
        compare_models(dfs, names, args.output_dir)
    else:
        print("\n📊 Creating analysis plots...")
        for df, name in zip(dfs, names):
            plot_single_model(df, args.output_dir)
    
    print(f"\n{'='*80}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
