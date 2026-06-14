"""Publication-quality global station distribution map."""
import pandas as pd, geopandas as gpd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import numpy as np
import os, warnings; warnings.filterwarnings('ignore')

SHP = ('/gpfs/home5/pkhanal/miniforge3/envs/terramind/lib/python3.11/'
       'site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp')
df    = pd.read_csv('csvs/station_splits.csv')
world = gpd.read_file(SHP)

# ── colour / marker config ────────────────────────────────────────────────────
SPLIT_CFG = {
    'train': dict(color='#2166ac', marker='o', s=28,  zorder=6, label='Train  (n=683)'),
    'val':   dict(color='#f4a620', marker='s', s=60,  zorder=7, label='Val  (n=91)'),
    'oos':   dict(color='#1a9850', marker='^', s=40,  zorder=6, label='OOS  (n=219)'),
}
SNOTEL_COLOR = '#d62728'

def draw_world(ax, xlim, ylim, lw=0.4):
    world.plot(ax=ax, color='#f0ede6', edgecolor='#aaaaaa', linewidth=lw)
    ax.set_facecolor('#d6e8f5')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, linewidth=0.3, alpha=0.45, color='white', linestyle='-')
    ax.tick_params(labelsize=8)

def scatter_splits(ax, sub_df):
    for split, cfg in SPLIT_CFG.items():
        sub = sub_df[sub_df['split'] == split]
        ax.scatter(sub['longitude'], sub['latitude'],
                   c=cfg['color'], s=cfg['s'], marker=cfg['marker'],
                   alpha=0.82, linewidths=0.3, edgecolors='white', zorder=cfg['zorder'])

def scatter_snotel_rings(ax, sub_df, s=55, lw=1.3):
    sn = sub_df[sub_df['network'] == 'SNOTEL']
    ax.scatter(sn['longitude'], sn['latitude'],
               facecolors='none', edgecolors=SNOTEL_COLOR,
               s=s, linewidths=lw, zorder=8)

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
gs  = GridSpec(2, 2, figure=fig,
               height_ratios=[2.2, 1],
               hspace=0.08, wspace=0.06,
               left=0.04, right=0.97, top=0.91, bottom=0.06)

ax_world = fig.add_subplot(gs[0, :])   # full-width top
ax_na    = fig.add_subplot(gs[1, 0])   # bottom-left: North America zoom
ax_eu    = fig.add_subplot(gs[1, 1])   # bottom-right: Europe zoom

# ── WORLD panel ───────────────────────────────────────────────────────────────
draw_world(ax_world, (-180, 180), (-60, 85), lw=0.4)
scatter_splits(ax_world, df)
scatter_snotel_rings(ax_world, df, s=50, lw=1.2)

# draw zoom-box outlines on world map
for (x0,x1,y0,y1), color in [
    ((-170,-55,20,75), '#555555'),   # NA box
    ((-15, 50, 30, 72), '#555555'),  # EU box
]:
    ax_world.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle='square,pad=0', fill=False,
        edgecolor=color, linewidth=1.4, linestyle='--', zorder=9))

ax_world.set_xlabel('Longitude', fontsize=10)
ax_world.set_ylabel('Latitude', fontsize=10)
ax_world.set_title('Global distribution of soil moisture monitoring stations by data split',
                   fontsize=14, fontweight='bold', pad=6)

# station counts annotation
total = len(df)
snotel_n = (df['network']=='SNOTEL').sum()
ax_world.text(0.01, 0.03,
    f'Total: {total} stations  |  SNOTEL: {snotel_n} ({snotel_n/total*100:.0f}%)',
    transform=ax_world.transAxes, fontsize=9.5,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ── NORTH AMERICA panel ───────────────────────────────────────────────────────
draw_world(ax_na, (-170, -55), (20, 75), lw=0.5)
scatter_splits(ax_na, df)
scatter_snotel_rings(ax_na, df, s=60, lw=1.3)
ax_na.set_title('North America', fontsize=11, fontweight='bold', pad=4)
ax_na.set_xlabel('Longitude', fontsize=9); ax_na.set_ylabel('Latitude', fontsize=9)

# annotate SNOTEL cluster
ax_na.annotate('SNOTEL\ncluster', xy=(-118, 44), xytext=(-145, 55),
               fontsize=8.5, color=SNOTEL_COLOR, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=SNOTEL_COLOR, lw=1.2))

# ── EUROPE panel ──────────────────────────────────────────────────────────────
draw_world(ax_eu, (-15, 50), (30, 72), lw=0.5)
scatter_splits(ax_eu, df)
scatter_snotel_rings(ax_eu, df, s=60, lw=1.3)
ax_eu.set_title('Europe', fontsize=11, fontweight='bold', pad=4)
ax_eu.set_xlabel('Longitude', fontsize=9); ax_eu.set_ylabel('Latitude', fontsize=9)

# ── LEGEND (attached to world panel) ─────────────────────────────────────────
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#2166ac',
           markersize=11, label='Train  (n=683)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#f4a620',
           markersize=11, label='Val  (n=91)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#1a9850',
           markersize=11, label='OOS  (n=219)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=SNOTEL_COLOR, markeredgewidth=1.5, markersize=11,
           label=f'SNOTEL  (n={snotel_n}, any split)'),
]
ax_world.legend(handles=legend_elements, loc='lower left',
                fontsize=11, framealpha=0.92, borderpad=0.9,
                edgecolor='#cccccc', fancybox=False)

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs('demo_output', exist_ok=True)
out = 'demo_output/station_splits_map.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
