"""
METAR Weather Report Analysis
Visualizes weather condition statistics from METAR reports.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

OUT_DIR = "analysis_out"

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# METAR Data
TOTAL_REPORTS = 2805
DATA = {
    "Visibility < 1000m": 275,
    "Ceiling < 100ft": 162,
    "Visibility < 1000m &\nCeiling < 100ft": 35,
    "CAVOK": 628,
}

COLORS = {
    'visibility': '#FF6B6B',    
    'ceiling': '#FFA94D',     
    'both': '#CC5DE8',   
    'cavok': '#51CF66',       
    'other': '#74C0FC',       
    'background': '#FFF',   
    'text': '#343A40',       
    'accent': '#495057',         
}

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def add_value_labels(ax, bars, values, total, offset=8):
    """Add beautiful value labels on top of bars."""
    for bar, val in zip(bars, values):
        height = bar.get_height()
        percentage = (val / total) * 100
        ax.annotate(
            f'{val:,}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=14, fontweight='bold',
            color=COLORS['text']
        )
        ax.annotate(
            f'{percentage:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset + 18),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10,
            color=COLORS['accent'],
            style='italic'
        )


def plot_metar_bar_chart(out_dir: str):
    """Create a beautiful bar chart showing METAR weather condition counts."""
    print_section("PLOT: METAR WEATHER CONDITIONS BAR CHART")

    categories = ["Visibility\n< 1000m", "Ceiling\n< 100ft", 
                  "Vis < 1000m &\nCeil < 100ft", "CAVOK"]
    values = [275, 162, 35, 628]
    colors = [COLORS['visibility'], COLORS['ceiling'], COLORS['both'], COLORS['cavok']]
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
 
    bars = ax.bar(categories, values, color=colors, width=0.65,
                  edgecolor='white', linewidth=2, zorder=3)
    
    for bar, color in zip(bars, colors):
        bar.set_alpha(0.9)
    
    add_value_labels(ax, bars, values, TOTAL_REPORTS)
    
    ax.set_ylabel('Number of Reports', fontsize=13, fontweight='medium', 
                  color=COLORS['text'], labelpad=10)
    ax.set_xlabel('Weather Condition', fontsize=13, fontweight='medium', 
                  color=COLORS['text'], labelpad=10)
    
    ax.set_title('METAR Weather Report Analysis', fontsize=18, fontweight='bold', 
                 color=COLORS['text'], pad=25)
    ax.text(0.5, 1.02, f'Total Reports: {TOTAL_REPORTS:,}', transform=ax.transAxes,
            ha='center', fontsize=12, color=COLORS['accent'], style='italic')
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='both', labelsize=11, colors=COLORS['text'])
    ax.set_ylim(0, max(values) * 1.25)
    
    ax.spines['left'].set_color(COLORS['accent'])
    ax.spines['bottom'].set_color(COLORS['accent'])
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "metar_conditions_bar.png")
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print(f"Saved: {path}")


def plot_metar_donut_chart(out_dir: str):
    """Create a beautiful donut chart showing METAR weather condition distribution."""
    print_section("PLOT: METAR WEATHER CONDITIONS DONUT CHART")
    
    vis_only = DATA["Visibility < 1000m"] - DATA["Visibility < 1000m &\nCeiling < 100ft"]
    ceil_only = DATA["Ceiling < 100ft"] - DATA["Visibility < 1000m &\nCeiling < 100ft"]
    both = DATA["Visibility < 1000m &\nCeiling < 100ft"]
    cavok = DATA["CAVOK"]
    other = TOTAL_REPORTS - (vis_only + ceil_only + both + cavok)
    
    sizes = [vis_only, ceil_only, both, cavok, other]
    labels = ['Visibility Only', 'Ceiling Only', 'Both Conditions', 'CAVOK', 'Other']
    colors = [COLORS['visibility'], COLORS['ceiling'], COLORS['both'], 
              COLORS['cavok'], COLORS['other']]
    
    fig, ax = plt.subplots(figsize=(11, 11), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,
        colors=colors,
        autopct='',
        startangle=90,
        pctdistance=0.75,
        wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 3}
    )
    
    centre_circle = plt.Circle((0, 0), 0.35, fc=COLORS['background'])
    ax.add_artist(centre_circle)
    
    ax.text(0, 0.08, f'{TOTAL_REPORTS:,}', ha='center', va='center',
            fontsize=32, fontweight='bold', color=COLORS['text'])
    ax.text(0, -0.12, 'Total Reports', ha='center', va='center',
            fontsize=14, color=COLORS['accent'])
    
    legend_labels = [f'{label}\n{size:,} ({size/TOTAL_REPORTS*100:.1f}%)' 
                     for label, size in zip(labels, sizes)]
    
    legend = ax.legend(wedges, legend_labels, loc='center left', 
                       bbox_to_anchor=(1.05, 0.5), fontsize=11,
                       frameon=True, fancybox=True, shadow=False,
                       edgecolor=COLORS['accent'], facecolor='white')
    legend.get_frame().set_alpha(0.9)
    
    ax.set_title('Weather Conditions Distribution', fontsize=18, fontweight='bold', 
                 color=COLORS['text'], pad=20)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "metar_conditions_donut.png")
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print(f"Saved: {path}")


def plot_metar_summary(out_dir: str):
    """Create an elegant combined summary visualization."""
    print_section("PLOT: METAR COMBINED SUMMARY")
    
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS['background'])
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.6], hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(COLORS['background'])
    
    categories = ["Visibility\n< 1000m", "Ceiling\n< 100ft", 
                  "Both\nConditions", "CAVOK"]
    values = [275, 162, 35, 628]
    colors = [COLORS['visibility'], COLORS['ceiling'], COLORS['both'], COLORS['cavok']]
    
    bars = ax1.bar(categories, values, color=colors, width=0.6,
                   edgecolor='white', linewidth=2, zorder=3)
    
    for bar in bars:
        bar.set_alpha(0.9)
    
    add_value_labels(ax1, bars, values, TOTAL_REPORTS, offset=6)
    
    ax1.set_ylabel('Number of Reports', fontsize=12, fontweight='medium', 
                   color=COLORS['text'], labelpad=10)
    ax1.set_title('Weather Conditions by Category', fontsize=14, fontweight='bold', 
                  color=COLORS['text'], pad=15)
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='both', labelsize=10, colors=COLORS['text'])
    ax1.set_ylim(0, max(values) * 1.3)
    ax1.spines['left'].set_color(COLORS['accent'])
    ax1.spines['bottom'].set_color(COLORS['accent'])
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(COLORS['background'])
    
    vis_only = 275 - 35
    ceil_only = 162 - 35
    both = 35
    cavok = 628
    other = TOTAL_REPORTS - (vis_only + ceil_only + both + cavok)
    
    sizes = [vis_only, ceil_only, both, cavok, other]
    colors_pie = [COLORS['visibility'], COLORS['ceiling'], COLORS['both'], 
                  COLORS['cavok'], COLORS['other']]
    
    wedges, _ = ax2.pie(sizes, colors=colors_pie, startangle=90,
                        wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2})
    
    centre_circle = plt.Circle((0, 0), 0.35, fc=COLORS['background'])
    ax2.add_artist(centre_circle)
    
    ax2.text(0, 0.05, f'{TOTAL_REPORTS:,}', ha='center', va='center',
             fontsize=20, fontweight='bold', color=COLORS['text'])
    ax2.text(0, -0.15, 'Total', ha='center', va='center',
             fontsize=11, color=COLORS['accent'])
    
    ax2.set_title('Overall Distribution', fontsize=14, fontweight='bold', 
                  color=COLORS['text'], pad=15)
    
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor(COLORS['background'])
    
    bar_data = [
        (vis_only, COLORS['visibility'], f'Visibility Only ({vis_only:,})'),
        (ceil_only, COLORS['ceiling'], f'Ceiling Only ({ceil_only:,})'),
        (both, COLORS['both'], f'Both ({both:,})'),
        (cavok, COLORS['cavok'], f'CAVOK ({cavok:,})'),
        (other, COLORS['other'], f'Other ({other:,})'),
    ]
    
    left = 0
    for width, color, label in bar_data:
        bar = ax3.barh([''], width, left=left, color=color, edgecolor='white', 
                       linewidth=2, label=label, height=0.5, alpha=0.9)
        if width > 150:
            ax3.text(left + width/2, 0, f'{width:,}', ha='center', va='center', 
                     fontsize=11, fontweight='bold', color='white',
                     path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        left += width
    
    ax3.set_xlim(0, TOTAL_REPORTS * 1.02)
    ax3.set_xlabel('Number of Reports', fontsize=12, fontweight='medium', 
                   color=COLORS['text'], labelpad=10)
    ax3.set_title('Complete Report Breakdown (Non-overlapping)', fontsize=14, 
                  fontweight='bold', color=COLORS['text'], pad=15)
    
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5, 
               fontsize=10, frameon=True, fancybox=True,
               edgecolor=COLORS['accent'], facecolor='white')
    
    ax3.tick_params(axis='x', labelsize=10, colors=COLORS['text'])
    ax3.tick_params(axis='y', left=False, labelleft=False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_color(COLORS['accent'])
    
    fig.suptitle('METAR Weather Report Analysis', fontsize=22, fontweight='bold', 
                 color=COLORS['text'], y=0.98)
    
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.15, hspace=0.4, wspace=0.25)
    
    path = os.path.join(out_dir, "metar_summary.png")
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print(f"Saved: {path}")


def print_statistics():
    """Print summary statistics."""
    print_section("METAR STATISTICS SUMMARY")
    
    print(f"Total METAR Reports: {TOTAL_REPORTS:,}")
    print()
    
    for condition, count in DATA.items():
        percentage = (count / TOTAL_REPORTS) * 100
        condition_clean = condition.replace('\n', ' ')
        print(f"  {condition_clean}: {count:,} ({percentage:.2f}%)")
    
    vis_only = DATA["Visibility < 1000m"] - DATA["Visibility < 1000m &\nCeiling < 100ft"]
    ceil_only = DATA["Ceiling < 100ft"] - DATA["Visibility < 1000m &\nCeiling < 100ft"]
    both = DATA["Visibility < 1000m &\nCeiling < 100ft"]
    other = TOTAL_REPORTS - (vis_only + ceil_only + both + DATA["CAVOK"])
    
    print()
    print("Non-overlapping breakdown:")
    print(f"  Visibility Only: {vis_only}")
    print(f"  Ceiling Only: {ceil_only}")
    print(f"  Both Conditions: {both}")
    print(f"  CAVOK: {DATA['CAVOK']}")
    print(f"  Other Conditions: {other}")


def main():
    ensure_out_dir(OUT_DIR)
    
    print_statistics()
    
    plot_metar_bar_chart(OUT_DIR)
    plot_metar_donut_chart(OUT_DIR)
    plot_metar_summary(OUT_DIR)
    
    print_section("DONE")
    print(f"All outputs written to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
