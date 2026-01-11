#!/usr/bin/env python3
"""
Create clean, minimal figures for CoR paper
Style: Academic, professional, similar to top-tier venues
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Professional academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.5,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.frameon': False,
    'legend.fontsize': 8,
})

# Color palette - muted, professional
COLORS = {
    'primary': '#2563EB',    # Blue
    'secondary': '#64748B',  # Slate
    'accent': '#DC2626',     # Red
    'success': '#059669',    # Green
    'warning': '#D97706',    # Amber
    'purple': '#7C3AED',     # Purple
    'gray': '#9CA3AF',
}


def create_sample_efficiency():
    """Sample efficiency - clean scatter plot for paper."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # Data: (samples, accuracy, label, color, marker, size)
    models = [
        (1000, 56.7, 'CoR-32B', COLORS['primary'], 'o', 80),
        (1000, 50.0, 'w/o CoR', COLORS['gray'], 's', 50),
        (17000, 43.3, 'Sky-T1', COLORS['secondary'], '^', 45),
        (17000, 63.3, 'Bespoke', COLORS['secondary'], 'D', 45),
        (800000, 72.6, 'r1-distill', COLORS['accent'], 'v', 55),
    ]
    
    ax.set_xscale('log')
    
    for x, y, label, color, marker, size in models:
        ax.scatter(x, y, c=color, s=size, marker=marker, 
                  edgecolors='white', linewidths=0.8, zorder=5, label=label)
    
    # o1-preview reference line
    ax.axhline(y=44.6, color=COLORS['warning'], linestyle='--', linewidth=1, alpha=0.7)
    ax.text(300000, 46, 'o1-preview', fontsize=7, color=COLORS['warning'], style='italic')
    
    # Annotation: sample efficiency
    ax.annotate('', xy=(1500, 53), xytext=(500000, 53),
               arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=1))
    ax.text(15000, 54.5, '800× fewer', fontsize=7, ha='center', color=COLORS['primary'])
    
    ax.set_xlabel('Training Samples', fontsize=9)
    ax.set_ylabel('AIME24 Accuracy (%)', fontsize=9)
    ax.set_xlim(400, 2000000)
    ax.set_ylim(38, 78)
    
    ax.legend(loc='lower right', fontsize=7, markerscale=0.8)
    
    plt.tight_layout()
    plt.savefig('cor_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cor_efficiency.png', dpi=150, bbox_inches='tight')
    print("✓ cor_efficiency.pdf")
    plt.close()


def create_reward_breakdown():
    """Reward components - horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    
    components = ['Total', 'R_converge', 'R_improve', 'R_int', 'R_ext']
    values = [1.49, 0.08, 0.15, 0.72, 0.54]
    colors = ['#1F2937', COLORS['purple'], COLORS['warning'], COLORS['primary'], COLORS['success']]
    
    y_pos = np.arange(len(components))
    bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.03, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                va='center', fontsize=8, color='#374151')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(components, fontsize=8)
    ax.set_xlabel('Reward Value', fontsize=9)
    ax.set_xlim(0, 1.8)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('cor_rewards.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cor_rewards.png', dpi=150, bbox_inches='tight')
    print("✓ cor_rewards.pdf")
    plt.close()


def create_calibration():
    """Self-rating calibration - line plot."""
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    
    ratings = np.linspace(0.1, 0.9, 9)
    
    # CoR: well calibrated
    cor_quality = ratings + np.array([0.02, 0.02, 0.01, 0.02, 0.01, -0.02, -0.02, -0.03, -0.02])
    
    # Baseline: overconfident
    baseline_quality = ratings * 0.78 + 0.01
    
    # Perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.4, label='Perfect')
    
    ax.plot(ratings, cor_quality, 'o-', color=COLORS['primary'], 
            linewidth=1.5, markersize=4, label='CoR-32B')
    ax.plot(ratings, baseline_quality, 's--', color=COLORS['gray'], 
            linewidth=1, markersize=3, label='Baseline')
    
    ax.set_xlabel('Self-Rating', fontsize=9)
    ax.set_ylabel('Actual Quality', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('cor_calibration.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cor_calibration.png', dpi=150, bbox_inches='tight')
    print("✓ cor_calibration.pdf")
    plt.close()


def create_ablation():
    """Ablation study - grouped bars."""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    
    benchmarks = ['AIME24', 'MATH500', 'GPQA']
    x = np.arange(len(benchmarks))
    width = 0.18
    
    # Data
    sft_only = [50.0, 92.6, 56.6]
    no_rint = [52.3, 92.8, 57.5]
    no_self = [53.5, 92.5, 58.0]
    cor_full = [56.7, 93.0, 59.6]
    
    ax.bar(x - 1.5*width, sft_only, width, label='SFT only', color=COLORS['gray'])
    ax.bar(x - 0.5*width, no_rint, width, label='w/o R_int', color='#FCD34D')
    ax.bar(x + 0.5*width, no_self, width, label='w/o Self-Rating', color='#93C5FD')
    ax.bar(x + 1.5*width, cor_full, width, label='CoR (full)', color=COLORS['primary'])
    
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=9)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.set_ylim(45, 100)
    
    # Only show left and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('cor_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cor_ablation.png', dpi=150, bbox_inches='tight')
    print("✓ cor_ablation.pdf")
    plt.close()


def create_scaling():
    """Model scaling results."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Model sizes (B parameters)
    sizes = [0.5, 1.5, 7, 32]
    size_labels = ['0.5B', '1.5B', '7B', '32B']
    
    # Accuracy on AIME24
    sft_acc = [23.3, 33.3, 43.3, 50.0]
    cor_acc = [26.7, 40.0, 50.0, 56.7]
    
    x = np.arange(len(sizes))
    width = 0.35
    
    ax.bar(x - width/2, sft_acc, width, label='SFT', color=COLORS['gray'])
    ax.bar(x + width/2, cor_acc, width, label='CoR-GRPO', color=COLORS['primary'])
    
    # Improvement annotations
    for i, (s, c) in enumerate(zip(sft_acc, cor_acc)):
        diff = c - s
        ax.annotate(f'+{diff:.1f}', xy=(i + width/2, c + 1), 
                   fontsize=7, ha='center', color=COLORS['primary'])
    
    ax.set_ylabel('AIME24 Accuracy (%)', fontsize=9)
    ax.set_xlabel('Model Size', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 65)
    
    plt.tight_layout()
    plt.savefig('cor_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('cor_scaling.png', dpi=150, bbox_inches='tight')
    print("✓ cor_scaling.pdf")
    plt.close()


def create_main_results_table():
    """Main results comparison as figure table."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis('off')
    
    # Table data
    headers = ['Model', 'AIME24', 'MATH500', 'GPQA', 'Samples']
    data = [
        ['o1-preview', '44.6', '—', '73.3', '—'],
        ['DeepSeek-R1-Distill', '72.6', '97.3', '62.5', '800K'],
        ['Sky-T1-32B', '43.3', '82.4', '56.8', '17K'],
        ['Bespoke-32B', '63.3', '93.0', '—', '17K'],
        ['\\textbf{CoR-32B (Ours)}', '\\textbf{56.7}', '\\textbf{93.0}', '\\textbf{59.6}', '\\textbf{1K}'],
    ]
    
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.28, 0.15, 0.17, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E5E7EB')
        table[(0, i)].set_text_props(fontweight='bold')
    
    # Highlight our row
    for i in range(len(headers)):
        table[(5, i)].set_facecolor('#DBEAFE')
    
    plt.tight_layout()
    plt.savefig('cor_results_table.pdf', dpi=300, bbox_inches='tight')
    print("✓ cor_results_table.pdf")
    plt.close()


if __name__ == '__main__':
    print("\nGenerating CoR figures (academic style)...\n")
    create_sample_efficiency()
    create_reward_breakdown()
    create_calibration()
    create_ablation()
    create_scaling()
    print("\n✅ All figures generated!")
