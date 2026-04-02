"""
CarbonTatva AI — GDAM/DAM/RTM Feature Analysis
================================================
Generates all 12 analysis plots and the full HTML report.

Setup:
    python3 -m venv venv
    source venv/bin/activate
    pip install pandas openpyxl matplotlib seaborn scipy scikit-learn numpy
    python analysis.py

Output:
    figures/fig1_price_overview.png ... fig12_supply_side.png
    carbontatva_feature_analysis_v2.html
"""

import os
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_FILE   = 'final_merged_data.xlsx'
OUTPUT_DIR  = 'figures'
REPORT_FILE = 'carbontatva_feature_analysis_v2.html'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# COLOUR / STYLE
# ─────────────────────────────────────────────
C  = {'GDAM': '#e74c3c', 'DAM': '#2980b9', 'RTM': '#27ae60'}
BG = '#0f1117'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    '#1a1d27',
    'axes.edgecolor':    '#3a3d4d',
    'text.color':        '#e0e0e0',
    'xtick.color':       '#aaa',
    'ytick.color':       '#aaa',
    'grid.color':        '#2a2d3a',
    'grid.linewidth':    0.5,
    'axes.labelcolor':   '#ccc',
    'font.family':       'DejaVu Sans',
})

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
print("Loading data...")
sheets = pd.read_excel(DATA_FILE, sheet_name=None)


def prep(df, name):
    df = df.copy()
    d = pd.to_datetime(df['Delivery Date'], dayfirst=True)
    t = df['Time Period'].str.split('-').str[0]
    df['datetime'] = pd.to_datetime(d.astype(str) + ' ' + t)
    df = df.sort_values('datetime').reset_index(drop=True)
    df['hour']       = df['datetime'].dt.hour
    df['slot']       = df['hour'] * 4 + df['datetime'].dt.minute // 15
    df['dow']        = df['datetime'].dt.dayofweek
    df['month']      = df['datetime'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    p = df['Price (Rs./MWh)']
    df['price_lag1']   = p.shift(1)
    df['price_lag4']   = p.shift(4)
    df['price_lag96']  = p.shift(96)
    df['price_lag672'] = p.shift(672)
    df['ma3d']  = p.shift(1).rolling(288).mean()
    df['ma7d']  = p.shift(1).rolling(672).mean()
    df['ma14d'] = p.shift(1).rolling(1344).mean()
    df['imbalance'] = df['Cleared Buy (MW)'] - df['Cleared Sell (MW)']
    return df


gdam = prep(sheets['GDAM'], 'GDAM')
dam  = prep(sheets['DAM'],  'DAM')
rtm  = prep(sheets['RTM'],  'RTM')
all_dfs = {'GDAM': gdam, 'DAM': dam, 'RTM': rtm}

FEAT_COLS = [
    'temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
    'precipitation (mm)', 'apparent_temperature (°C)', 'cloud_cover (%)',
    'wind_speed_100m (km/h)', 'sunshine_duration (s)', 'shortwave_radiation (W/m²)'
]

print("Data loaded. Generating figures...\n")

# ─────────────────────────────────────────────
# FIG 1 — Price distributions + time-series
# ─────────────────────────────────────────────
print("Fig 1: Price overview...")
fig, axes = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle('Figure 1 — Price Overview: Distributions & Time Series',
             fontsize=16, color='white', y=1.01)

for i, (name, df) in enumerate(all_dfs.items()):
    ax = axes[i][0]
    ax.hist(df['Price (Rs./MWh)'].clip(0, 15000), bins=80,
            color=C[name], alpha=0.8, edgecolor='none')
    ax.axvline(df['Price (Rs./MWh)'].median(), color='yellow', lw=1.5,
               ls='--', label=f"Median={df['Price (Rs./MWh)'].median():.0f}")
    ax.set_title(f'{name} — Price Distribution', color='white')
    ax.set_xlabel('Price (Rs./MWh)')
    ax.legend(fontsize=9)

    ax2 = axes[i][1]
    daily = df.set_index('datetime')['Price (Rs./MWh)'].resample('D').mean()
    ax2.plot(daily.index, daily.values, color=C[name], lw=0.8, alpha=0.9)
    roll = daily.rolling(30).mean()
    ax2.plot(roll.index, roll.values, color='white', lw=1.5, label='30-day MA')
    ax2.set_title(f'{name} — Daily Mean Price', color='white')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_price_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 2 — Hourly + Day-of-Week seasonality
# ─────────────────────────────────────────────
print("Fig 2: Temporal patterns...")
fig, axes = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle('Figure 2 — Temporal Patterns: Hourly & Day-of-Week Price Profiles',
             fontsize=16, color='white')
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

for i, (name, df) in enumerate(all_dfs.items()):
    ax = axes[i][0]
    hourly = df.groupby('hour')['Price (Rs./MWh)'].agg(['mean', 'std'])
    ax.fill_between(hourly.index,
                    hourly['mean'] - hourly['std'],
                    hourly['mean'] + hourly['std'],
                    color=C[name], alpha=0.25)
    ax.plot(hourly.index, hourly['mean'], color=C[name], lw=2)
    ax.set_title(f'{name} — Hourly Price Profile (±1σ)', color='white')
    ax.set_xlabel('Hour of Day')
    ax.set_xticks(range(0, 24, 2))

    ax2 = axes[i][1]
    dow = df.groupby('dow')['Price (Rs./MWh)'].mean()
    ax2.bar(days, dow.values, color=C[name], alpha=0.85)
    ax2.set_title(f'{name} — Day-of-Week Average Price', color='white')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Avg Price (Rs./MWh)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_temporal.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 3 — Hour × Month heatmap
# ─────────────────────────────────────────────
print("Fig 3: Seasonality heatmap...")
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle('Figure 3 — Monthly Seasonality Heatmap (Hour × Month)',
             fontsize=16, color='white')

for i, (name, df) in enumerate(all_dfs.items()):
    pivot = df.pivot_table(values='Price (Rs./MWh)',
                           index='hour', columns='month', aggfunc='mean')
    sns.heatmap(pivot, ax=axes[i], cmap='RdYlGn_r', linewidths=0,
                cbar_kws={'label': 'Avg Price (Rs./MWh)', 'shrink': 0.7})
    axes[i].set_title(f'{name}', color='white', fontsize=13)
    axes[i].set_xlabel('Month')
    axes[i].set_ylabel('Hour')
    axes[i].tick_params(colors='#ccc')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_seasonality_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 4 — Full correlation matrix
# ─────────────────────────────────────────────
print("Fig 4: Correlation matrix...")
fig, axes = plt.subplots(1, 3, figsize=(22, 9))
fig.suptitle('Figure 4 — Feature Correlation Matrix (Weather + Market → Price)',
             fontsize=16, color='white')

for i, (name, df) in enumerate(all_dfs.items()):
    cols_for_corr = FEAT_COLS + ['Cleared Buy (MW)', 'Cleared Sell (MW)', 'Price (Rs./MWh)']
    corr = df[cols_for_corr].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, ax=axes[i], mask=mask, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                linewidths=0.3, cbar_kws={'shrink': 0.6})
    axes[i].set_title(f'{name}', color='white', fontsize=13)
    axes[i].tick_params(axis='x', rotation=45, labelsize=7)
    axes[i].tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 5 — Weather scatter plots vs price
# ─────────────────────────────────────────────
print("Fig 5: Scatter plots...")
top_feats = ['temperature_2m (°C)', 'shortwave_radiation (W/m²)',
             'wind_speed_100m (km/h)', 'relative_humidity_2m (%)']
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
fig.suptitle('Figure 5 — Top Weather Features vs Price (Scatter + Trend)',
             fontsize=15, color='white')

for row, (name, df) in enumerate(all_dfs.items()):
    sample = df.sample(min(5000, len(df)), random_state=42)
    for col, feat in enumerate(top_feats):
        ax = axes[row][col]
        ax.scatter(sample[feat], sample['Price (Rs./MWh)'],
                   s=2, alpha=0.3, color=C[name])
        x = sample[feat].values
        y = sample['Price (Rs./MWh)'].values
        ok = np.isfinite(x) & np.isfinite(y)
        m, b, r, p_val, _ = stats.linregress(x[ok], y[ok])
        xl = np.array([x[ok].min(), x[ok].max()])
        ax.plot(xl, m * xl + b, color='yellow', lw=1.5, label=f'r={r:.2f}')
        ax.set_xlabel(feat.split(' ')[0], fontsize=9)
        if col == 0:
            ax.set_ylabel(f'{name}\nPrice', fontsize=8)
        ax.set_title(feat.split('(')[0].strip(), fontsize=9, color='white')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 6 — Cross-correlation ±24h
# ─────────────────────────────────────────────
print("Fig 6: Cross-correlation...")
lag_feats = ['temperature_2m (°C)', 'shortwave_radiation (W/m²)',
             'wind_speed_100m (km/h)', 'cloud_cover (%)']
MAX_LAG = 96

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
fig.suptitle('Figure 6 — Cross-Correlation: Weather Features vs Price (Lag ±24h)',
             fontsize=14, color='white')

for row, (name, df) in enumerate(all_dfs.items()):
    p_sig = (df['Price (Rs./MWh)'] - df['Price (Rs./MWh)'].mean()).fillna(0).values
    for col, feat in enumerate(lag_feats):
        ax = axes[row][col]
        f_sig = (df[feat] - df[feat].mean()).fillna(0).values
        cc = correlate(p_sig, f_sig, mode='full')
        cc = cc / (np.std(p_sig) * np.std(f_sig) * len(p_sig))
        mid = len(cc) // 2
        lags = np.arange(-MAX_LAG, MAX_LAG + 1)
        cc_slice = cc[mid - MAX_LAG: mid + MAX_LAG + 1]
        ax.bar(lags * 0.25, cc_slice, width=0.2, color=C[name], alpha=0.7)
        ax.axvline(0, color='yellow', lw=1, ls='--')
        best_lag = lags[np.argmax(np.abs(cc_slice))] * 0.25
        ax.axvline(best_lag, color='#ff9f43', lw=1.2, ls=':',
                   label=f'peak={best_lag:.1f}h')
        ax.set_xlabel('Lag (hours)', fontsize=8)
        if col == 0:
            ax.set_ylabel(name, fontsize=9)
        if row == 0:
            ax.set_title(feat.split('(')[0].strip(), fontsize=9, color='white')
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_crosscorr.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 7 — Price autocorrelation 0–7 days
# ─────────────────────────────────────────────
print("Fig 7: Autocorrelation...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Figure 7 — Price Autocorrelation (0–7 Days)', fontsize=15, color='white')

for i, (name, df) in enumerate(all_dfs.items()):
    p = df['Price (Rs./MWh)'].ffill().values
    p_norm = (p - p.mean()) / p.std()
    lags_h = range(0, 169)
    ac = []
    for lh in lags_h:
        ls = lh * 4
        if ls == 0:
            ac.append(1.0)
        elif ls < len(p_norm):
            ac.append(np.corrcoef(p_norm[:-ls], p_norm[ls:])[0, 1])
        else:
            ac.append(np.nan)

    ax = axes[i]
    ax.fill_between(list(lags_h), ac, color=C[name], alpha=0.6)
    ax.axhline(0, color='white', lw=0.8)
    for peak_h in [24, 48, 96, 168]:
        ax.axvline(peak_h, color='yellow', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('Lag (hours)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'{name}', color='white')
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
    ax.set_xticklabels(['0', '24h', '48h', '3d', '4d', '5d', '6d', '7d'])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig7_autocorr.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 8 — AR lag feature correlation bar chart
# ─────────────────────────────────────────────
print("Fig 8: AR lag correlations...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Figure 8 — Autoregressive Lag Features vs Price (Pearson r)',
             fontsize=14, color='white')
lag_labels = ['lag-1\n(15min)', 'lag-4\n(1hr)', 'lag-96\n(1day)',
              'lag-672\n(1week)', 'MA-3d', 'MA-7d', 'MA-14d']
lag_cols   = ['price_lag1', 'price_lag4', 'price_lag96', 'price_lag672',
              'ma3d', 'ma7d', 'ma14d']

for i, (name, df) in enumerate(all_dfs.items()):
    df2 = df.dropna(subset=lag_cols)
    corrs = [df2['Price (Rs./MWh)'].corr(df2[f]) for f in lag_cols]
    bars = axes[i].bar(lag_labels, corrs, color=C[name], alpha=0.85)
    axes[i].axhline(0, color='white', lw=0.8)
    for bar, v in zip(bars, corrs):
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.005 * np.sign(v), f'{v:.3f}',
                     ha='center',
                     va='bottom' if v >= 0 else 'top',
                     fontsize=8, color='white')
    axes[i].set_title(f'{name}', color='white')
    axes[i].set_ylabel('Pearson r')
    axes[i].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig8_ar_corr.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 9 — Volume & buy-sell imbalance
# ─────────────────────────────────────────────
print("Fig 9: Volume analysis...")
fig, axes = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle('Figure 9 — Market Depth: Cleared Volume vs Price by Hour',
             fontsize=14, color='white')

for i, (name, df) in enumerate(all_dfs.items()):
    hourly_price = df.groupby('hour')['Price (Rs./MWh)'].mean()
    hourly_buy   = df.groupby('hour')['Cleared Buy (MW)'].mean()
    hourly_sell  = df.groupby('hour')['Cleared Sell (MW)'].mean()

    ax1 = axes[i][0]
    ax2 = ax1.twinx()
    ax1.fill_between(hourly_buy.index,  hourly_buy,  alpha=0.5, color='#00d2ff', label='Buy MW')
    ax1.fill_between(hourly_sell.index, hourly_sell, alpha=0.5, color='#fd79a8', label='Sell MW')
    ax2.plot(hourly_price.index, hourly_price, color='yellow', lw=2, label='Avg Price')
    ax1.set_title(f'{name} — Volume vs Price', color='white')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Volume (MW)', color='#aaa')
    ax2.set_ylabel('Price (Rs./MWh)', color='yellow')
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc='upper left')

    ax3 = axes[i][1]
    samp = df.sample(min(3000, len(df)), random_state=42)
    ax3.scatter(samp['imbalance'], samp['Price (Rs./MWh)'],
                s=2, alpha=0.3, color=C[name])
    ok = np.isfinite(samp['imbalance']) & np.isfinite(samp['Price (Rs./MWh)'])
    m, b, r, _, _ = stats.linregress(samp['imbalance'][ok],
                                      samp['Price (Rs./MWh)'][ok])
    xl = np.array([samp['imbalance'].min(), samp['imbalance'].max()])
    ax3.plot(xl, m * xl + b, color='yellow', lw=2, label=f'r={r:.2f}')
    ax3.set_xlabel('Buy-Sell Imbalance (MW)')
    ax3.set_ylabel('Price (Rs./MWh)')
    ax3.set_title(f'{name} — Imbalance vs Price', color='white')
    ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig9_volume.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 10 — Feature importance (Random Forest)
# ─────────────────────────────────────────────
print("Fig 10: Feature importance (RF — slowest step, ~2-3 min)...")
rf_feats = ['hour', 'dow', 'month', 'is_weekend',
            'temperature_2m (°C)', 'relative_humidity_2m (%)', 'rain (mm)',
            'cloud_cover (%)', 'wind_speed_100m (km/h)',
            'sunshine_duration (s)', 'shortwave_radiation (W/m²)',
            'Cleared Buy (MW)', 'Cleared Sell (MW)']

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Figure 10 — Feature Importance (Random Forest Proxy)',
             fontsize=14, color='white')

for i, (name, df) in enumerate(all_dfs.items()):
    sample = (df[rf_feats + ['Price (Rs./MWh)']]
              .dropna()
              .sample(min(20000, len(df)), random_state=42))
    X = sample[rf_feats].values
    y = sample['Price (Rs./MWh)'].values
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]

    def short(s):
        for suffix in [' (°C)', ' (%)', ' (km/h)', ' (s)', ' (W/m²)', ' (mm)']:
            s = s.replace(suffix, '')
        return s

    labels = [short(rf_feats[j]) for j in idx]
    ax = axes[i]
    ax.barh(labels[::-1], imp[idx][::-1], color=C[name], alpha=0.85)
    ax.set_title(f'{name}', color='white', fontsize=12)
    ax.set_xlabel('Feature Importance')
    for bar, v in zip(ax.patches, imp[idx][::-1]):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=7.5, color='white')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 11 — Weekend vs Weekday distributions
# ─────────────────────────────────────────────
print("Fig 11: Weekend effect...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Figure 11 — Weekend vs Weekday Price Distribution',
             fontsize=14, color='white')

for i, (name, df) in enumerate(all_dfs.items()):
    ax = axes[i]
    wd = df[df['is_weekend'] == 0]['Price (Rs./MWh)'].clip(0, 15000)
    we = df[df['is_weekend'] == 1]['Price (Rs./MWh)'].clip(0, 15000)
    ax.hist(wd, bins=60, alpha=0.6, color='#3498db',
            label=f'Weekday (μ={wd.mean():.0f})', density=True)
    ax.hist(we, bins=60, alpha=0.6, color='#e74c3c',
            label=f'Weekend (μ={we.mean():.0f})', density=True)
    ax.set_title(f'{name}', color='white')
    ax.set_xlabel('Price (Rs./MWh)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig11_weekend.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# FIG 12 — Solar & Wind hourly profiles vs price
# ─────────────────────────────────────────────
print("Fig 12: Supply-side features...")
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Figure 12 — Supply-Side Features: Solar & Wind Hourly Profiles vs Price',
             fontsize=14, color='white')

for i, (name, df) in enumerate(all_dfs.items()):
    hourly_price = df.groupby('hour')['Price (Rs./MWh)'].mean()
    hourly_solar = df.groupby('hour')['shortwave_radiation (W/m²)'].mean()
    hourly_wind  = df.groupby('hour')['wind_speed_100m (km/h)'].mean()

    ax1 = axes[0][i]
    ax2 = ax1.twinx()
    ax1.fill_between(hourly_solar.index, hourly_solar,
                     alpha=0.7, color='#f9ca24', label='Solar Radiation')
    ax2.plot(hourly_price.index, hourly_price,
             color='white', lw=2, label='Avg Price')
    ax1.set_title(f'{name} — Solar vs Price', color='white')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Radiation (W/m²)', color='#f9ca24')
    ax2.set_ylabel('Price (Rs./MWh)', color='white')
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    ax3 = axes[1][i]
    ax4 = ax3.twinx()
    ax3.fill_between(hourly_wind.index, hourly_wind,
                     alpha=0.7, color='#00cec9', label='Wind Speed')
    ax4.plot(hourly_price.index, hourly_price,
             color='white', lw=2, label='Avg Price')
    ax3.set_title(f'{name} — Wind vs Price', color='white')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Wind Speed (km/h)', color='#00cec9')
    ax4.set_ylabel('Price (Rs./MWh)', color='white')
    lines1, lab1 = ax3.get_legend_handles_labels()
    lines2, lab2 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig12_supply_side.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nAll 12 figures saved to: {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# 2. EMBED FIGURES & BUILD HTML REPORT
# ─────────────────────────────────────────────
print("\nBuilding HTML report...")


def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


b64 = {
    'fig1':  img_to_b64(f'{OUTPUT_DIR}/fig1_price_overview.png'),
    'fig2':  img_to_b64(f'{OUTPUT_DIR}/fig2_temporal.png'),
    'fig3':  img_to_b64(f'{OUTPUT_DIR}/fig3_seasonality_heatmap.png'),
    'fig4':  img_to_b64(f'{OUTPUT_DIR}/fig4_correlation.png'),
    'fig5':  img_to_b64(f'{OUTPUT_DIR}/fig5_scatter.png'),
    'fig6':  img_to_b64(f'{OUTPUT_DIR}/fig6_crosscorr.png'),
    'fig7':  img_to_b64(f'{OUTPUT_DIR}/fig7_autocorr.png'),
    'fig8':  img_to_b64(f'{OUTPUT_DIR}/fig8_ar_corr.png'),
    'fig9':  img_to_b64(f'{OUTPUT_DIR}/fig9_volume.png'),
    'fig10': img_to_b64(f'{OUTPUT_DIR}/fig10_feature_importance.png'),
    'fig11': img_to_b64(f'{OUTPUT_DIR}/fig11_weekend.png'),
    'fig12': img_to_b64(f'{OUTPUT_DIR}/fig12_supply_side.png'),
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CarbonTatva AI — GDAM/DAM/RTM Feature Analysis Report v2</title>
<style>
  :root{{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;--accent:#e74c3c;--blue:#2980b9;--green:#27ae60;--text:#e0e0e0;--muted:#aaa;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',Arial,sans-serif;line-height:1.75;}}
  .cover{{background:linear-gradient(135deg,#0f1117 0%,#1a1d27 50%,#0f1117 100%);border-bottom:3px solid var(--accent);padding:60px 80px;}}
  .cover h1{{font-size:2.4rem;color:#fff;letter-spacing:1px;margin-bottom:8px;}}
  .cover .sub{{color:var(--accent);font-size:1.1rem;margin-bottom:20px;font-weight:600;}}
  .cover p{{color:var(--muted);font-size:0.95rem;max-width:700px;}}
  .cover .badges{{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap;}}
  .badge{{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:4px 14px;font-size:0.8rem;color:var(--muted);}}
  .badge.red{{border-color:var(--accent);color:var(--accent);}}
  .badge.blue{{border-color:var(--blue);color:var(--blue);}}
  .badge.green{{border-color:var(--green);color:var(--green);}}
  .container{{max-width:1200px;margin:0 auto;padding:40px 60px;}}
  .toc{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:30px;margin:30px 0;}}
  .toc h2{{color:#fff;margin-bottom:16px;font-size:1.2rem;}}
  .toc ol{{padding-left:20px;}}
  .toc li{{margin:8px 0;color:var(--muted);font-size:0.95rem;}}
  .toc li a{{color:#7f8ff4;text-decoration:none;}}
  .toc li a:hover{{color:#fff;}}
  .section{{margin:50px 0;}}
  .section-title{{display:flex;align-items:center;gap:12px;margin-bottom:24px;border-bottom:2px solid var(--border);padding-bottom:12px;}}
  .section-num{{background:var(--accent);color:#fff;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1rem;flex-shrink:0;}}
  .section-num.blue{{background:var(--blue);}}
  .section-num.green{{background:var(--green);}}
  h2{{color:#fff;font-size:1.5rem;}}
  h3{{color:#fff;font-size:1.1rem;margin:20px 0 10px;}}
  p{{color:var(--muted);margin:10px 0;font-size:0.95rem;}}
  .plot{{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin:24px 0;}}
  .plot img{{width:100%;display:block;}}
  .plot .caption{{padding:12px 16px;font-size:0.82rem;color:var(--muted);border-top:1px solid var(--border);background:#13161f;}}

  /* ── NEW: plain-english explanation box ── */
  .explain{{background:#12172a;border:1px solid #2e3a5c;border-left:4px solid #7f8ff4;border-radius:0 10px 10px 0;padding:20px 24px;margin:6px 0 20px;}}
  .explain .ex-head{{display:flex;align-items:center;gap:8px;font-size:0.78rem;color:#7f8ff4;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:10px;}}
  .explain .ex-body{{color:#c8d3e8;font-size:0.93rem;line-height:1.8;}}
  .explain .ex-body strong{{color:#fff;}}
  .explain .ex-bullets{{margin-top:10px;padding-left:18px;}}
  .explain .ex-bullets li{{color:#b0bcd8;font-size:0.9rem;margin:5px 0;}}

  .insight-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:20px 0;}}
  .insight-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;}}
  .insight-card .ic-label{{font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}}
  .insight-card .ic-value{{font-size:1.6rem;font-weight:700;color:#fff;}}
  .insight-card .ic-desc{{font-size:0.82rem;color:var(--muted);margin-top:6px;}}
  .insight-card.red{{border-color:var(--accent);}}
  .insight-card.blue{{border-color:var(--blue);}}
  .insight-card.green{{border-color:var(--green);}}
  table{{width:100%;border-collapse:collapse;margin:20px 0;font-size:0.88rem;}}
  th{{background:#1e2233;color:#fff;padding:10px 14px;text-align:left;border-bottom:2px solid var(--border);}}
  td{{padding:9px 14px;border-bottom:1px solid var(--border);color:var(--muted);}}
  tr:nth-child(even){{background:#161929;}}
  .rank-badge{{display:inline-block;background:#2a2d3a;border-radius:4px;padding:2px 8px;font-size:0.78rem;font-weight:700;}}
  .rank-badge.t1{{background:#c0392b;color:#fff;}}
  .rank-badge.t2{{background:#2471a3;color:#fff;}}
  .rank-badge.t3{{background:#1e8449;color:#fff;}}

  /* ── final plan ── */
  .final-plan{{background:linear-gradient(135deg,#1a1d27,#161929);border:2px solid #7f8ff4;border-radius:16px;padding:36px;margin:30px 0;}}
  .final-plan h2{{color:#7f8ff4;margin-bottom:20px;}}
  .feat-card{{background:#0f1117;border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin:14px 0;}}
  .feat-card .fc-header{{display:flex;align-items:center;gap:12px;margin-bottom:10px;}}
  .feat-card .fc-num{{background:#7f8ff4;color:#fff;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;flex-shrink:0;}}
  .feat-card .fc-num.t1{{background:#c0392b;}}
  .feat-card .fc-num.t2{{background:#2471a3;}}
  .feat-card .fc-num.t3{{background:#1e8449;}}
  .feat-card .fc-name{{color:#fff;font-size:1rem;font-weight:600;}}
  .feat-card .fc-tag{{font-size:0.75rem;background:#2a2d3a;padding:2px 8px;border-radius:4px;color:#aaa;margin-left:auto;}}
  .feat-card .fc-why{{color:#b0bcd8;font-size:0.88rem;margin:6px 0 4px;}}
  .feat-card .fc-how{{background:#1a1d27;border-radius:6px;padding:10px 14px;font-size:0.85rem;color:#7f8ff4;margin-top:8px;font-family:monospace;}}
  .feat-card .fc-example{{color:#aaa;font-size:0.83rem;margin-top:6px;font-style:italic;}}
  .phase{{background:var(--card);border-left:4px solid var(--accent);border-radius:0 8px 8px 0;padding:18px 20px;margin:16px 0;}}
  .phase.blue{{border-color:var(--blue);}}
  .phase.green{{border-color:var(--green);}}
  .phase h3{{color:#fff;font-size:1rem;margin-bottom:8px;}}
  .phase p{{color:var(--muted);font-size:0.88rem;margin:6px 0;}}
  .phase ul{{color:var(--muted);font-size:0.88rem;padding-left:18px;}}
  .phase ul li{{margin:5px 0;}}
  .callout{{background:#1c1f2e;border:1px solid #7f8ff4;border-radius:8px;padding:16px 20px;margin:16px 0;font-size:0.88rem;color:#c8d3ff;}}
  .callout strong{{color:#fff;}}
  .warn{{background:#1f1510;border:1px solid #e67e22;border-radius:8px;padding:14px 18px;margin:12px 0;font-size:0.87rem;color:#f0c080;}}
  footer{{border-top:1px solid var(--border);padding:30px 60px;text-align:center;color:var(--muted);font-size:0.82rem;margin-top:60px;}}
</style>
</head>
<body>

<!-- COVER -->
<div class="cover">
  <div style="font-size:0.8rem;color:var(--accent);letter-spacing:3px;text-transform:uppercase;margin-bottom:12px;">CarbonTatva AI · Internal Research</div>
  <h1>GDAM / DAM / RTM Price Forecasting</h1>
  <div class="sub">Feature Analysis, Correlation Study & Modelling Roadmap</div>
  <p>Comprehensive exploratory data analysis of 144,000+ 15-minute electricity market records (Dec 2022 – Feb 2026) across three Indian power exchanges — covering temporal patterns, weather-feature correlations, lagged effects, supply-side dynamics and a fully elaborated feature-engineering plan with plain-language explanations for every graph.</p>
  <div class="badges">
    <span class="badge red">GDAM · μ=₹5,422/MWh</span>
    <span class="badge blue">DAM · μ=₹4,924/MWh</span>
    <span class="badge green">RTM · μ=₹4,669/MWh</span>
    <span class="badge">~4 Years · 15-min resolution</span>
    <span class="badge">12 Analysis Plots</span>
  </div>
</div>

<div class="container">

<div class="toc">
  <h2>📋 Table of Contents</h2>
  <ol>
    <li><a href="#s1">Dataset Overview & Key Statistics</a></li>
    <li><a href="#s2">Price Distribution & Long-Term Trends (Fig 1)</a></li>
    <li><a href="#s3">Temporal Patterns — Hourly, Weekly & Monthly Seasonality (Fig 2, 3)</a></li>
    <li><a href="#s4">Weather Feature Correlations (Fig 4, 5)</a></li>
    <li><a href="#s5">Lagged Cross-Correlation Analysis (Fig 6)</a></li>
    <li><a href="#s6">Price Autocorrelation & Autoregressive Structure (Fig 7, 8)</a></li>
    <li><a href="#s7">Market Depth — Volume & Imbalance (Fig 9)</a></li>
    <li><a href="#s8">Supply-Side: Solar & Wind Effects (Fig 12)</a></li>
    <li><a href="#s9">Feature Importance — Random Forest Proxy (Fig 10, 11)</a></li>
    <li><a href="#s10">Final Feature Engineering Plan & Modelling Roadmap</a></li>
  </ol>
</div>

<!-- ═══════════════════ S1 ═══════════════════ -->
<div class="section" id="s1">
  <div class="section-title">
    <div class="section-num">1</div>
    <h2>Dataset Overview & Key Statistics</h2>
  </div>
  <p>We have 15-minute electricity price data from three different Indian power market segments. Each record pairs market clearing data (buy/sell volumes and price) with weather measurements at the same timestamp.</p>
  <div class="explain">
    <div class="ex-head">🧭 What is each market?</div>
    <div class="ex-body">
      <strong>GDAM (Green Day-Ahead Market)</strong> — trades electricity from renewable sources only. Buyers pay a premium for "green" power. Prices are consistently highest here.<br/><br/>
      <strong>DAM (Day-Ahead Market)</strong> — the standard electricity auction where buyers bid for power to be delivered tomorrow. This is the most liquid and commonly studied market.<br/><br/>
      <strong>RTM (Real-Time Market)</strong> — intraday trading that clears every 30 minutes for near-immediate delivery. It reacts fastest to supply shocks (cloud cover, grid faults) and is the hardest to forecast.
    </div>
  </div>
  <div class="insight-grid">
    <div class="insight-card red"><div class="ic-label">GDAM — Avg Price</div><div class="ic-value">₹5,422</div><div class="ic-desc">Highest avg; premium for green certificates. CV = 0.53</div></div>
    <div class="insight-card blue"><div class="ic-label">DAM — Avg Price</div><div class="ic-value">₹4,924</div><div class="ic-desc">Standard day-ahead auction. CV = 0.60 — most volatile</div></div>
    <div class="insight-card green"><div class="ic-label">RTM — Avg Price</div><div class="ic-value">₹4,669</div><div class="ic-desc">Real-time intraday. CV = 0.61 — hardest to forecast</div></div>
    <div class="insight-card"><div class="ic-label">Universal Peak Hour</div><div class="ic-value">19:00</div><div class="ic-desc">Evening peak — solar is gone, residential load is highest</div></div>
    <div class="insight-card"><div class="ic-label">Universal Trough</div><div class="ic-value">13:00</div><div class="ic-desc">Solar noon → max renewable injection → lowest price</div></div>
    <div class="insight-card"><div class="ic-label">Max Observed Price</div><div class="ic-value">₹20,000</div><div class="ic-desc">Price cap all three markets — grid-stress spike events present</div></div>
  </div>
  <table>
    <tr><th>Market</th><th>Records</th><th>Mean Price</th><th>Std Dev</th><th>CV</th><th>Weekday Avg</th><th>Weekend Avg</th><th>Weekend Δ</th></tr>
    <tr><td>GDAM</td><td>144,864</td><td>₹5,422</td><td>₹2,871</td><td>0.53</td><td>₹5,530</td><td>₹5,150</td><td><span class="rank-badge t1">−6.9%</span></td></tr>
    <tr><td>DAM</td><td>144,864</td><td>₹4,924</td><td>₹2,974</td><td>0.60</td><td>₹5,072</td><td>₹4,553</td><td><span class="rank-badge t2">−10.2%</span></td></tr>
    <tr><td>RTM</td><td>144,116</td><td>₹4,669</td><td>₹2,847</td><td>0.61</td><td>₹4,858</td><td>₹4,196</td><td><span class="rank-badge t3">−13.6%</span></td></tr>
  </table>
</div>

<!-- ═══════════════════ S2 ═══════════════════ -->
<div class="section" id="s2">
  <div class="section-title">
    <div class="section-num blue">2</div>
    <h2>Price Distribution & Long-Term Trends</h2>
  </div>
  <div class="plot">
    <img src="data:image/png;base64,{fig1}" alt="Price Overview"/>
    <div class="caption"><strong>Fig 1.</strong> Left column: Price frequency histograms with median (yellow dashed line). Right column: Daily mean price + 30-day rolling average (white line) from Dec 2022 to Feb 2026.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      <strong>Left panels (histograms):</strong> The x-axis is the price in Rs/MWh and the y-axis is how often that price appears. A tall bar means that price is very common. You can see a big cluster of bars on the left side (low prices are most frequent) and then a long thin tail stretching to ₹20,000 on the right — this means extreme price spikes are rare but do happen.<br/><br/>
      <strong>Right panels (time series):</strong> Each dot on the coloured line is the average price on one day. The white line is a 30-day smoothed version — think of it as showing the "trend" without the noise. You can clearly see waves going up and down — these are seasonal cycles.
    </div>
    <ul class="ex-bullets">
      <li><strong>Why does this matter for us?</strong> The right-skewed distribution (long tail) tells us we cannot use a simple linear model — we need models that handle extreme values. We should consider <em>log-transforming</em> the price before training.</li>
      <li><strong>The seasonal waves</strong> confirm that <em>month of year</em> must be a feature — summer and winter peaks, monsoon troughs are real and recurring.</li>
      <li><strong>All three markets follow the same seasonal pattern</strong>, which means we can use the same feature set across markets — just retrain the model separately for each.</li>
    </ul>
  </div>
</div>

<!-- ═══════════════════ S3 ═══════════════════ -->
<div class="section" id="s3">
  <div class="section-title">
    <div class="section-num green">3</div>
    <h2>Temporal Patterns — Hourly, Weekly & Monthly Seasonality</h2>
  </div>
  <div class="plot">
    <img src="data:image/png;base64,{fig2}" alt="Temporal Patterns"/>
    <div class="caption"><strong>Fig 2.</strong> Left: Average price by hour of day, with the shaded band showing ±1 standard deviation (spread). Right: Average price by day of week (Mon=0, Sun=6).</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      <strong>Left (Hourly profile):</strong> The solid line is the average price at each hour over the entire dataset. The shaded area around it shows how much prices vary at that hour — a wide band means that hour is unpredictable, a narrow band means it is consistent. Look for the two bumps (morning peak ~9h, big evening peak ~19h) and the deep trough at ~13h.<br/><br/>
      <strong>Right (Day of week):</strong> Simply the average price on each day. Taller bar = more expensive day on average.
    </div>
    <ul class="ex-bullets">
      <li><strong>Evening spike at 19h is universal</strong> — this is when solar generation stops but household electricity demand is at its highest (cooking, lights, AC). Our model must capture this sharp transition. The <em>hour feature with sin/cos encoding</em> does exactly this.</li>
      <li><strong>Solar trough at 13h</strong> — midday prices are the lowest because solar panels are generating at full capacity, flooding supply. This directly connects solar radiation to price.</li>
      <li><strong>Weekdays are more expensive than weekends</strong> — factories and offices are running. This gap is 7–14% across markets. We must include an <em>is_weekend</em> binary flag.</li>
      <li><strong>Wide variance band in evening hours</strong> means the model will find 18–21h the hardest to predict — this is where we need the most features.</li>
    </ul>
  </div>

  <div class="plot">
    <img src="data:image/png;base64,{fig3}" alt="Seasonality Heatmap"/>
    <div class="caption"><strong>Fig 3.</strong> Heatmap: each cell is the average price for a given (Hour, Month) combination. Dark red = expensive. Dark green = cheap. This is the most information-dense temporal plot.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      <strong>Rows</strong> = Hour of day (0 at top, 23 at bottom). <strong>Columns</strong> = Month of year (1=Jan, 12=Dec). Each coloured cell shows the average price for that exact hour+month combination. Look for dark red clusters (expensive: evening + summer/winter) and dark green clusters (cheap: midday + monsoon months).
    </div>
    <ul class="ex-bullets">
      <li><strong>The green stripe across the middle rows (hours 10–15)</strong> in most months confirms the solar trough — midday is cheap all year round.</li>
      <li><strong>The bright red top-right corner (hours 18–22, months May–June)</strong> = peak summer evening load — the most expensive time-slot in the whole dataset. Our model must learn this interaction between hour and month.</li>
      <li><strong>Monsoon months (July–September)</strong> are visibly cooler (greener) across all hours — high hydro + wind generation brings prices down.</li>
      <li><strong>What to do with this:</strong> Include both <em>hour</em> and <em>month</em> as features, AND consider their interaction term <code>hour × month</code> or allow a tree/NN model to learn it automatically.</li>
    </ul>
  </div>
</div>

<!-- ═══════════════════ S4 ═══════════════════ -->
<div class="section" id="s4">
  <div class="section-title">
    <div class="section-num" style="background:#8e44ad">4</div>
    <h2>Weather Feature Correlations</h2>
  </div>
  <div class="plot">
    <img src="data:image/png;base64,{fig4}" alt="Correlation Matrix"/>
    <div class="caption"><strong>Fig 4.</strong> Lower-triangle Pearson correlation matrix for all weather features + market features. Each cell shows a number from −1 (perfect inverse relationship) to +1 (perfect direct relationship). Blue = negative, Red = positive.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      Think of this as a "relationship table." Each row and column is one feature. The number in the cell tells you how strongly two features move together. <strong>A value near ±1 means they are almost the same thing</strong> (redundant — drop one). A value near 0 means they are unrelated.
    </div>
    <ul class="ex-bullets">
      <li><strong>apparent_temperature ↔ temperature: r ≈ 0.97</strong> — nearly identical. Drop <code>apparent_temperature</code>, keep <code>temperature_2m</code>.</li>
      <li><strong>sunshine_duration ↔ shortwave_radiation: r ≈ 0.93</strong> — two measures of the same thing (sun). Keep radiation (it's in physical units W/m²), drop sunshine_duration.</li>
      <li><strong>rain ↔ precipitation: r ≈ 0.99</strong> — almost exactly the same column. Keep one.</li>
      <li><strong>shortwave_radiation ↔ Price: r ≈ −0.30 to −0.36</strong> — meaningful negative correlation. More solar → lower price. This is the strongest weather signal.</li>
    </ul>
  </div>

  <div class="plot">
    <img src="data:image/png;base64,{fig5}" alt="Scatter plots"/>
    <div class="caption"><strong>Fig 5.</strong> Scatter plots of 4 key weather features vs price. Each dot is one 15-min slot. The yellow trend line is a linear fit; the "r" value in the legend is the Pearson correlation.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      Each scatter plot puts one weather variable on the x-axis and price on y-axis. If the dots cluster around an upward-sloping line, the feature and price move together. A downward slope means they are inversely related. A cloud of random dots means no linear relationship.
    </div>
    <ul class="ex-bullets">
      <li><strong>Solar radiation (top right)</strong> — clearly downward slope (r ≈ −0.30 to −0.36). High radiation → lots of solar power pumped into the grid → prices go down. This is one of the most important weather features to include.</li>
      <li><strong>Temperature (top left)</strong> — almost flat (r ≈ 0). This does NOT mean temperature is useless — it means the relationship is <em>non-linear</em> (U-shaped: extreme cold AND extreme hot both increase price). We should add a <code>temperature²</code> feature to capture this.</li>
      <li><strong>Wind speed (bottom left)</strong> — weak downward trend. More wind → more wind generation → lower prices, but the effect is noisy. Still worth including.</li>
      <li><strong>Humidity (bottom right)</strong> — near-zero correlation. Weakest direct signal, but it acts as a monsoon proxy. Keep for now as Tier 3.</li>
    </ul>
  </div>

  <table>
    <tr><th>Feature</th><th>GDAM r</th><th>DAM r</th><th>RTM r</th><th>Direction</th><th>Importance</th></tr>
    <tr><td>shortwave_radiation</td><td>−0.357</td><td>−0.304</td><td>−0.234</td><td>More solar → lower price</td><td><span class="rank-badge t1">HIGH</span></td></tr>
    <tr><td>temperature_2m</td><td>−0.031</td><td>+0.011</td><td>+0.029</td><td>Non-linear (U-shape)</td><td><span class="rank-badge t1">HIGH (add temp²)</span></td></tr>
    <tr><td>cloud_cover</td><td>−0.053</td><td>−0.065</td><td>−0.098</td><td>Weak negative</td><td><span class="rank-badge t2">MEDIUM</span></td></tr>
    <tr><td>wind_speed_100m</td><td>−0.004</td><td>−0.013</td><td>−0.069</td><td>Weak negative</td><td><span class="rank-badge t2">MEDIUM</span></td></tr>
    <tr><td>relative_humidity</td><td>+0.033</td><td>−0.016</td><td>−0.074</td><td>Mixed / monsoon proxy</td><td><span class="rank-badge t3">LOW–MEDIUM</span></td></tr>
    <tr><td>rain / precipitation</td><td>~0</td><td>~0</td><td>~0</td><td>Negligible direct effect</td><td><span class="rank-badge">LOW</span></td></tr>
  </table>
</div>

<!-- ═══════════════════ S5 ═══════════════════ -->
<div class="section" id="s5">
  <div class="section-title">
    <div class="section-num" style="background:#e67e22">5</div>
    <h2>Lagged Cross-Correlation Analysis</h2>
  </div>
  <div class="plot">
    <img src="data:image/png;base64,{fig6}" alt="Cross Correlation"/>
    <div class="caption"><strong>Fig 6.</strong> Cross-correlation between each weather feature and price at lags from −24h to +24h. The orange dashed vertical line marks the lag at which correlation is strongest. X-axis is lag in hours; positive lag means the feature leads price.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      Imagine you shift one signal forward or backward in time and check how well it matches the other. The x-axis shows the time shift ("lag") in hours. <strong>If the peak is at lag = 0</strong>, both signals are most related at the same time. <strong>If the peak is at lag = +5h</strong>, then the feature today predicts price 5 hours from now — that feature is a <em>leading indicator</em>.
    </div>
    <ul class="ex-bullets">
      <li><strong>Solar radiation peak lag ≈ 0 to +6h</strong> — radiation now is most correlated with price in the next few hours. Since weather forecasts give us radiation 24–48h ahead, we can use <em>future forecast radiation</em> as a known input to our model. This is a huge advantage.</li>
      <li><strong>Temperature lag ≈ 0, but broad plateau</strong> — temperature effect on demand is immediate but sustained. Use current + 24h-ahead temperature forecast.</li>
      <li><strong>Wind speed lag ≈ 0</strong> — wind effect is concurrent. Use wind forecast from open-meteo as exogenous input.</li>
      <li><strong>Practical takeaway:</strong> All four key weather features have their peak influence at or near lag = 0. This confirms we should feed <em>weather forecasts for the target time slot</em> (not just current weather) into our model. Open-Meteo API provides free 48h forecasts.</li>
    </ul>
  </div>
</div>

<!-- ═══════════════════ S6 ═══════════════════ -->
<div class="section" id="s6">
  <div class="section-title">
    <div class="section-num">6</div>
    <h2>Price Autocorrelation & Autoregressive Structure</h2>
  </div>
  <div class="plot">
    <img src="data:image/png;base64,{fig7}" alt="Autocorrelation"/>
    <div class="caption"><strong>Fig 7.</strong> Autocorrelation Function (ACF) of price from lag 0 to lag 168 hours (7 days). Yellow dashed lines mark the 24h, 48h, 96h, and 168h marks. The y-axis is Pearson r — how much today's price resembles the price at that lag.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      The ACF answers one question: <em>"If I know the price right now, how useful is the price from X hours ago for predicting the current price?"</em> A high value means past prices are still very informative. The curve naturally decays (falls off) as the lag grows, but look out for <strong>bumps/peaks at regular intervals</strong> — those reveal periodicity.
    </div>
    <ul class="ex-bullets">
      <li><strong>The curve stays very high (≈ 0.7–0.9) even at 168h (1 week)</strong> — electricity prices have very long memory. Last week's price at the same time is still a good predictor of today's price.</li>
      <li><strong>The bumps at exactly 24h, 48h, 72h… intervals</strong> confirm the <em>daily cycle</em> — price at 8pm today closely resembles price at 8pm yesterday. This is why <code>P(t−96)</code> (same slot, 1 day ago) is such a powerful feature. Each "day" = 96 slots of 15 minutes.</li>
      <li><strong>RTM decays faster</strong> than GDAM/DAM — real-time prices are more responsive to current conditions, so the past matters a little less. This means RTM needs stronger weather and volume features to compensate.</li>
    </ul>
  </div>

  <div class="plot">
    <img src="data:image/png;base64,{fig8}" alt="AR Lag Correlations"/>
    <div class="caption"><strong>Fig 8.</strong> Bar chart of Pearson correlation between specific lag features and the current price. Each bar represents how much one past price value explains the current price.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      This is a simplified summary of the ACF above. Each bar is the correlation between price right now and the price at a specific past moment: 15 minutes ago (lag-1), 1 hour ago (lag-4), 1 day ago (lag-96), and 1 week ago (lag-672). Taller bar = more predictive power.
    </div>
    <ul class="ex-bullets">
      <li><strong>Lag-1 (15 min ago): r ≈ 0.96–0.97</strong> — the single most powerful feature in the entire dataset. If price was ₹5,000 fifteen minutes ago, it is very likely still near ₹5,000 now.</li>
      <li><strong>Lag-96 (yesterday, same slot): r ≈ 0.76–0.88</strong> — almost as powerful. "Yesterday 8pm" is almost as good a predictor of "today 8pm" as just the last 15-minute reading. This captures the daily routine of the grid.</li>
      <li><strong>Lag-672 (last week, same slot): r ≈ 0.60–0.75</strong> — still very useful. Weekly industrial/commercial patterns repeat reliably.</li>
      <li><strong>Moving averages (MA-3d, MA-7d)</strong> are at r ≈ 0.75–0.85 — they capture the underlying price level/trend without the noise of individual spikes.</li>
    </ul>
  </div>

  <table>
    <tr><th>Lag Feature</th><th>GDAM r</th><th>DAM r</th><th>RTM r</th><th>Meaning</th><th>Include?</th></tr>
    <tr><td>P(t−1) — 15 min ago</td><td>0.967</td><td>0.975</td><td>0.959</td><td>Price is highly persistent over short time</td><td><span class="rank-badge t1">YES — Critical</span></td></tr>
    <tr><td>P(t−4) — 1 hour ago</td><td>0.851</td><td>0.854</td><td>0.826</td><td>Within-hour dynamics still strong</td><td><span class="rank-badge t1">YES</span></td></tr>
    <tr><td>P(t−96) — 1 day ago</td><td>0.863</td><td>0.878</td><td>0.760</td><td>Same hour yesterday — daily seasonality</td><td><span class="rank-badge t1">YES — Critical</span></td></tr>
    <tr><td>P(t−672) — 1 week ago</td><td>0.723</td><td>0.748</td><td>0.601</td><td>Same hour last week — weekly seasonality</td><td><span class="rank-badge t2">YES</span></td></tr>
    <tr><td>MA-3d, MA-7d</td><td>~0.85</td><td>~0.87</td><td>~0.78</td><td>Price trend / level context</td><td><span class="rank-badge t2">YES</span></td></tr>
  </table>
</div>

<!-- ═══════════════════ S7 ═══════════════════ -->
<div class="section" id="s7">
  <div class="section-title">
    <div class="section-num blue">7</div>
    <h2>Market Depth — Cleared Volume & Imbalance</h2>
  </div>
  <div class="plot">
    <img src="data:image/png;base64,{fig9}" alt="Volume Analysis"/>
    <div class="caption"><strong>Fig 9.</strong> Left: Cleared buy (cyan) and sell (pink) volume by hour overlaid with average price (yellow). Right: Scatter of buy-sell imbalance vs price, with trend line and correlation r.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      <strong>Left panel:</strong> Two filled area charts stacked together, with price as a line on the right axis. It shows at what hours the market is most active (tall areas = lots of volume traded) and whether price peaks align with volume changes.<br/><br/>
      <strong>Right panel:</strong> "Buy-Sell Imbalance" = how much more buying than selling is happening at each 15-min slot. If lots of buyers and few sellers → price goes up. If lots of sellers and few buyers → price goes down. The scatter shows this intuition and the trend line confirms it.
    </div>
    <ul class="ex-bullets">
      <li><strong>Negative imbalance (more sell than buy) → lower price</strong> — confirmed by the downward trend. Excess renewable supply in the afternoon shows up clearly.</li>
      <li><strong>DAM volumes are much larger</strong> (mean Buy = 1,558 MW vs GDAM's 47 MW) — DAM is the main bulk market. Volume features matter more for DAM.</li>
      <li><strong>Leakage warning:</strong> At forecasting time for RTM, we don't know the cleared volume yet — it hasn't been auctioned. So for RTM, we can only use <em>lagged</em> volumes (e.g., volume from the previous slot). For DAM, the D-1 auction result is known before delivery, so volume is a valid feature.</li>
    </ul>
  </div>
</div>

<!-- ═══════════════════ S8 ═══════════════════ -->
<div class="section" id="s8">
  <div class="section-title">
    <div class="section-num green">8</div>
    <h2>Supply-Side: Solar & Wind Effects</h2>
  </div>
  <div class="plot">
    <img src="data:image/png;base64,{fig10}" alt="Supply Side"/>
    <div class="caption"><strong>Fig 12.</strong> Top row: Solar shortwave radiation (yellow filled) vs average price (white line) by hour of day. Bottom row: Wind speed at 100m (teal filled) vs price by hour.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      Two overlapping area/line plots for each market. The filled area is the weather variable (solar or wind); the white line is the price. The two y-axes scale independently — we are looking for the <em>shape</em> of the two curves relative to each other, not absolute values.
    </div>
    <ul class="ex-bullets">
      <li><strong>Solar vs Price (top row) — clear mirror image:</strong> When solar radiation rises (8am–1pm), price falls. When radiation falls (afternoon to evening), price rises sharply. This is the famous <em>Solar Duck Curve</em> — the grid dips in the middle of the day from solar oversupply, then ramps steeply in the evening.</li>
      <li><strong>Wind vs Price (bottom row) — weaker but present:</strong> Wind speed peaks in the afternoon in many regions. This provides some additional generation at exactly the time when solar starts to fall, slightly buffering the evening price spike. But the relationship is noisier than solar.</li>
      <li><strong>Why this matters for Andhra Pradesh specifically:</strong> AP has very high solar penetration (one of India's largest solar parks). The duck curve effect is therefore particularly strong here. Including a real-time solar radiation forecast from open-meteo for AP's location will be a very significant feature for our DISCOM clients.</li>
      <li><strong>Feature engineering idea:</strong> Create an interaction feature: <code>solar_radiation × hour_sin</code> to explicitly capture the solar-duck-curve shape that the model needs to learn.</li>
    </ul>
  </div>
</div>

<!-- ═══════════════════ S9 ═══════════════════ -->
<div class="section" id="s9">
  <div class="section-title">
    <div class="section-num" style="background:#8e44ad">9</div>
    <h2>Feature Importance — Random Forest Proxy</h2>
  </div>
  <div class="plot">
    <img src="data:image/png;base64,{fig11}" alt="Feature Importance"/>
    <div class="caption"><strong>Fig 10.</strong> Horizontal bar chart of feature importance scores from a 100-tree Random Forest trained on 20,000 samples per market. Longer bar = more useful feature for predicting price.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      A Random Forest is a collection of decision trees. For each feature, it measures how much better the model's predictions become when that feature is available. The "importance" score (x-axis) is expressed as a fraction — all bars add up to 1. A bar of 0.3 means that feature alone is responsible for 30% of the model's ability to predict correctly.
    </div>
    <ul class="ex-bullets">
      <li><strong>Hour of day dominates</strong> (typically 0.20–0.35) — confirms what we saw in Fig 2. The intraday cycle is the strongest signal in the data.</li>
      <li><strong>Month and Cleared Buy follow</strong> — seasonal patterns and current market activity are the second tier of importance.</li>
      <li><strong>Solar radiation appears in the top 5</strong> for all three markets, validating its inclusion as a must-have weather feature.</li>
      <li><strong>Temperature, wind, humidity are lower</strong> — not useless, but secondary. The Random Forest cannot capture the non-linear temperature effect (U-shape) easily with raw values, which is why it ranks it lower than it deserves. Our quadratic transform will fix this.</li>
    </ul>
  </div>

  <div class="plot">
    <img src="data:image/png;base64,{fig12}" alt="Weekend Effect"/>
    <div class="caption"><strong>Fig 11.</strong> Overlapping price histograms for weekdays (blue) vs weekends (red), shown as probability densities (so total area = 1 for each). The shift between the two peaks shows how much cheaper weekends are.</div>
  </div>
  <div class="explain">
    <div class="ex-head">📖 How to read this graph</div>
    <div class="ex-body">
      Two overlapping bell-curve-like distributions. If the red (weekend) hump is clearly to the left of the blue (weekday) hump, weekends are cheaper on average. The gap between the peaks is the "weekend discount."
    </div>
    <ul class="ex-bullets">
      <li><strong>All three markets show a clear leftward shift for weekends</strong> — RTM has the biggest shift (−13.6%), confirming real-time markets are most sensitive to industrial demand patterns.</li>
      <li><strong>Simple feature to add:</strong> A single binary column <code>is_weekend = 1 if Sat/Sun else 0</code> captures this entire effect.</li>
      <li><strong>Enhancement:</strong> Later, add a <em>holiday flag</em> (national/AP state holidays) — holidays behave like Sundays even mid-week and adding this to the feature set will reduce forecast errors on those days.</li>
    </ul>
  </div>
</div>

<!-- ═══════════════════ S10 ═══════════════════ -->
<div class="section" id="s10">
  <div class="final-plan">
    <h2>🗺️ Final Feature Engineering Plan & Modelling Roadmap</h2>
    <p style="color:#c8d3ff;margin-bottom:24px;">This is the definitive, elaborated guide on every feature — what it is, why we need it, how to compute it, and what it tells the model. Organised by priority tier.</p>

    <!-- ── TIER 1 ── -->
    <h3 style="color:#c0392b;margin:24px 0 8px;">🔴 TIER 1 — Must-Have Features (core model)</h3>
    <p style="color:#aaa;font-size:0.88rem;margin-bottom:16px;">These features alone can build a strong baseline model. Never train without them.</p>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t1">1</div>
        <div class="fc-name">Hour of Day — Cyclical Encoding</div>
        <div class="fc-tag">Temporal</div>
      </div>
      <div class="fc-why">The single most important temporal feature. Price has a strong, recurring pattern every 24 hours (morning ramp, solar trough at 1pm, evening spike at 7pm). Raw hour 0–23 is a bad encoding because the model thinks hour 23 and hour 0 are far apart, when in reality they are consecutive.</div>
      <div class="fc-how">hour_sin = sin(2π × hour / 24)<br/>hour_cos = cos(2π × hour / 24)</div>
      <div class="fc-example">Example: hour=0 → sin=0, cos=1 &nbsp;|&nbsp; hour=12 → sin=0, cos=−1 &nbsp;|&nbsp; hour=6 → sin=1, cos=0</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t1">2</div>
        <div class="fc-name">Month — Cyclical Encoding</div>
        <div class="fc-tag">Temporal</div>
      </div>
      <div class="fc-why">Captures annual seasonality: summer heat waves (April–June), monsoon cooling (July–Sept), winter demand (Dec–Feb). Same cyclical encoding logic — Dec and Jan must be "close together" in feature space.</div>
      <div class="fc-how">month_sin = sin(2π × month / 12)<br/>month_cos = cos(2π × month / 12)</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t1">3</div>
        <div class="fc-name">Day of Week — Cyclical Encoding + is_weekend flag</div>
        <div class="fc-tag">Temporal</div>
      </div>
      <div class="fc-why">Weekday vs weekend demand is 7–14% different. The cyclical encoding helps the model understand the weekly cycle (Mon→Tue→…→Sun→Mon). The binary is_weekend provides an explicit strong signal for the demand drop.</div>
      <div class="fc-how">dow_sin = sin(2π × dow / 7) &nbsp;&nbsp; (0=Mon, 6=Sun)<br/>dow_cos = cos(2π × dow / 7)<br/>is_weekend = 1 if dow ≥ 5 else 0</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t1">4</div>
        <div class="fc-name">P(t−1) — Price 15 minutes ago</div>
        <div class="fc-tag">Autoregressive</div>
      </div>
      <div class="fc-why">The single strongest numerical predictor in the dataset (r = 0.96–0.97). Price is "sticky" — it does not jump randomly from slot to slot. If price was ₹4,000 fifteen minutes ago, there is very high probability it is still near ₹4,000 now. This captures short-term market inertia.</div>
      <div class="fc-how">lag1 = price.shift(1)  &nbsp;&nbsp;# shift by 1 row in sorted DataFrame</div>
      <div class="fc-example">At 19:15, use the 19:00 price as a feature</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t1">5</div>
        <div class="fc-name">P(t−4) — Price 1 hour ago</div>
        <div class="fc-tag">Autoregressive</div>
      </div>
      <div class="fc-why">Captures within-hour trends. If price has been rising for the past hour, the model should "know" it is in a rising phase. Combined with lag-1, this gives the model a short-term trajectory signal.</div>
      <div class="fc-how">lag4 = price.shift(4)  &nbsp;&nbsp;# 4 slots × 15 min = 1 hour</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t1">6</div>
        <div class="fc-name">P(t−96) — Same Slot Yesterday</div>
        <div class="fc-tag">Autoregressive</div>
      </div>
      <div class="fc-why">The single most powerful longer-horizon feature (r = 0.76–0.88). "Yesterday 7pm price" is an excellent predictor of "today 7pm price." This captures the daily seasonal cycle directly in a single number — the model does not need to re-learn that 7pm is always expensive.</div>
      <div class="fc-how">lag96 = price.shift(96)  &nbsp;&nbsp;# 96 slots × 15 min = 24 hours = 1 day</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t1">7</div>
        <div class="fc-name">P(t−672) — Same Slot Last Week</div>
        <div class="fc-tag">Autoregressive</div>
      </div>
      <div class="fc-why">Captures the weekly cycle. Tuesday 8pm this week behaves like Tuesday 8pm last week. This is especially valuable for DAM (r = 0.748). For RTM, it is weaker but still useful as a directional anchor.</div>
      <div class="fc-how">lag672 = price.shift(672)  &nbsp;&nbsp;# 672 slots = 7 days</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t1">8</div>
        <div class="fc-name">Shortwave Radiation (W/m²) — Solar Generation Proxy</div>
        <div class="fc-tag">Supply-side Weather</div>
      </div>
      <div class="fc-why">The strongest weather feature (r = −0.30 to −0.36). High radiation means solar panels in AP are producing lots of electricity → more supply → lower price. During night hours this is always 0 (no solar), automatically providing an implicit day/night signal as well. Since weather forecasts give us radiation 24–48h ahead, we can use the <em>forecasted</em> value at time t as a known-future exogenous input.</div>
      <div class="fc-how">Use raw W/m² value.<br/>Optional: add interaction → radiation_x_hour = radiation × hour_sin</div>
      <div class="fc-example">At 13:00, radiation ≈ 800 W/m² → price tends to drop to ₹2,800. At 20:00, radiation = 0 → price jumps to ₹7,000+</div>
    </div>

    <!-- ── TIER 2 ── -->
    <h3 style="color:#2471a3;margin:28px 0 8px;">🔵 TIER 2 — Strong Supporting Features</h3>
    <p style="color:#aaa;font-size:0.88rem;margin-bottom:16px;">Add these to the Tier 1 baseline. They provide measurable performance lifts, especially for longer-horizon forecasts and spike events.</p>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t2">9</div>
        <div class="fc-name">Moving Average 3-day &amp; 7-day</div>
        <div class="fc-tag">Autoregressive</div>
      </div>
      <div class="fc-why">A rolling mean over the past 3 days (288 slots) and 7 days (672 slots) of price captures the current "price level" — is the market in a generally expensive period or a cheap period? This de-noises the lag features and helps the model know the background trend, not just the last spike.</div>
      <div class="fc-how">ma3d = price.shift(1).rolling(288).mean()<br/>ma7d = price.shift(1).rolling(672).mean()<br/>(shift(1) ensures no future data leaks in)</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t2">10</div>
        <div class="fc-name">Temperature (°C) + Temperature² (quadratic term)</div>
        <div class="fc-tag">Demand-side Weather</div>
      </div>
      <div class="fc-why">Temperature has a U-shaped effect on electricity demand: very cold winter (heaters) and very hot summer (ACs) both increase demand and price. The linear correlation is near-zero because the two ends of the curve cancel out. Adding <code>temp²</code> explicitly gives the model the ability to capture this curve — "the further from a comfortable 22°C, the more expensive".</div>
      <div class="fc-how">temp = df['temperature_2m (°C)']<br/>temp_sq = temp ** 2<br/># Or: temp_deviation = (temp - 22) ** 2  # distance from comfort zone</div>
      <div class="fc-example">Temp = 10°C (cold) → high demand, high price. Temp = 22°C (pleasant) → low demand, low price. Temp = 42°C (extreme heat) → high AC load, high price.</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t2">11</div>
        <div class="fc-name">Wind Speed at 100m (km/h)</div>
        <div class="fc-tag">Supply-side Weather</div>
      </div>
      <div class="fc-why">Wind turbines in India operate at 80–120m hub height. The 100m wind speed directly approximates how much wind energy is being generated. More wind → more supply → lower price, especially visible in RTM (r = −0.069). The effect is modest but consistent, and wind forecasts are freely available 48h ahead.</div>
      <div class="fc-how">Use raw km/h value. Available from open-meteo: wind_speed_100m parameter.</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t2">12</div>
        <div class="fc-name">Cloud Cover (%)</div>
        <div class="fc-tag">Supply-side Weather</div>
      </div>
      <div class="fc-why">Cloud cover directly reduces solar irradiance in real time. It acts as a faster-changing proxy for solar generation than the radiation measurement itself — sudden cloud cover during an otherwise sunny day can cause a short price spike that radiation (measured hourly) might miss at 15-min resolution.</div>
      <div class="fc-how">Use raw 0–100% value. Note: correlated with shortwave radiation, but not redundant — captures intra-hour solar variability.</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t2">13</div>
        <div class="fc-name">Buy-Sell Imbalance (MW)</div>
        <div class="fc-tag">Market Depth</div>
      </div>
      <div class="fc-why">Captures real-time supply-demand balance in the market. Excess buy demand (positive imbalance) pushes price up; excess supply (negative imbalance) pushes price down. This is a direct market microstructure signal that weather and temporal features cannot capture on their own.</div>
      <div class="fc-how">imbalance = cleared_buy_MW - cleared_sell_MW<br/># For RTM: use imbalance from t-1 (previous slot) to avoid leakage<br/># For DAM: previous-day clearing results are available before delivery</div>
    </div>

    <!-- ── TIER 3 ── -->
    <h3 style="color:#1e8449;margin:28px 0 8px;">🟢 TIER 3 — Enhancement Features (add after baseline is solid)</h3>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t3">14</div>
        <div class="fc-name">Relative Humidity (%)</div>
        <div class="fc-tag">Weather / Monsoon Proxy</div>
      </div>
      <div class="fc-why">Low direct correlation with price, but high humidity months (July–Sept) are exactly the monsoon months when hydro and wind generation are high and prices are low. So humidity acts as an indirect seasonal proxy that adds information on top of the month feature — especially useful for capturing unusual monsoon years.</div>
      <div class="fc-how">Use raw % value. No special transformation needed.</div>
    </div>

    <div class="feat-card">
      <div class="fc-header">
        <div class="fc-num t3">15</div>
        <div class="fc-name">Holiday Flag</div>
        <div class="fc-tag">Temporal / Calendar</div>
      </div>
      <div class="fc-why">National holidays (Diwali, Holi, Independence Day etc.) and AP state holidays cause a sudden drop in industrial demand, making price behave like a Sunday even mid-week. Without this flag, the model will consistently over-forecast on those days because it expects weekday-level demand.</div>
      <div class="fc-how">from pandas.tseries.holiday import AbstractHolidayCalendar<br/># Or use: pip install holidays → holidays.India()<br/>is_holiday = date.apply(lambda d: d in india_holidays)</div>
    </div>

    <!-- ── WHAT TO DROP ── -->
    <div class="warn" style="margin:24px 0;">
      <strong>⚠️ Features to DROP (and why):</strong><br/><br/>
      <strong>apparent_temperature</strong> — Computed from temperature + humidity. Pearson r with temperature = 0.97. Keeping both inflates variance explained artificially. Drop this, keep raw temperature.<br/><br/>
      <strong>sunshine_duration (s)</strong> — Pearson r with shortwave_radiation = 0.93. It's just counting the hours of sunshine, which is essentially the same info as radiation intensity. Drop this, keep shortwave_radiation (it's in physical units and more precise).<br/><br/>
      <strong>precipitation (mm)</strong> — Nearly identical to rain (mm), r ≈ 0.99. Keep one (whichever has fewer nulls), drop the other. Both have near-zero direct correlation with price anyway, so this is Tier 3 at most.
    </div>

    <!-- ── MODELLING ROADMAP ── -->
    <h3 style="color:#7f8ff4;margin:30px 0 12px;">🚀 Modelling Roadmap</h3>

    <div class="phase">
      <h3>Phase 1 — Baseline & Validation Framework (Week 1)</h3>
      <p>Before any deep learning, establish strong baselines and a robust evaluation setup.</p>
      <ul>
        <li><strong>Persistence baseline:</strong> Predict P(t) = P(t−96). This gives you an r ≈ 0.86 baseline for free — any ML model must beat this to be worth deploying.</li>
        <li><strong>LightGBM / XGBoost</strong> with all Tier 1+2 features. Tree models handle non-linearities (temperature U-shape) natively, train in minutes, and are interpretable. This is your "strong baseline."</li>
        <li><strong>Train/val/test split (time-based, no shuffling):</strong> Train = Dec 2022–Dec 2024, Validation = Jan–Sep 2025, Test = Oct 2025–Feb 2026. Never shuffle — future data must never leak into training.</li>
        <li><strong>Metrics to track:</strong> MAE (mean absolute error in Rs/MWh), RMSE, MAPE (%), and separately track spike accuracy (errors when price &gt; ₹10,000).</li>
      </ul>
    </div>

    <div class="phase blue">
      <h3>Phase 2 — Sequence Models: LSTM / CNN-LSTM (Weeks 2–3)</h3>
      <p>Sequence models learn patterns across a window of past time steps, capturing dynamics that tabular models miss.</p>
      <ul>
        <li><strong>Input window:</strong> 96 slots (24 hours of history). At each step, feed [price, weather features, temporal encodings]. Output: next 96 slots (full day-ahead forecast).</li>
        <li><strong>LSTM/GRU:</strong> Vanilla starting point. The LSTM's memory cell is well-suited to the long persistence we observed (ACF staying high at 7 days).</li>
        <li><strong>CNN-LSTM hybrid:</strong> A 1D convolutional layer first extracts local patterns (e.g., the daily peak shape) → feeds into LSTM for long-range temporal dependencies. This often outperforms pure LSTM in electricity forecasting papers.</li>
        <li><strong>Dual-stream architecture (recommended):</strong> Separate encoder for exogenous inputs (weather forecasts known ahead of time) and past price sequence, fused at the bottleneck before the output layer. This mirrors the Temporal Fusion Transformer idea at lower complexity.</li>
        <li><strong>Loss function:</strong> Huber loss (less sensitive to ₹20,000 cap spikes than MSE, more sensitive than MAE).</li>
      </ul>
    </div>

    <div class="phase green">
      <h3>Phase 3 — Transformer / Attention Models (Weeks 3–4)</h3>
      <ul>
        <li><strong>Temporal Fusion Transformer (TFT)</strong> — the gold standard for multi-horizon time-series forecasting with exogenous inputs. It natively handles: (a) known future inputs like weather forecasts, (b) past observables like price history, (c) static covariates like which market. Published by Google, open-source via pytorch-forecasting library.</li>
        <li><strong>PatchTST / iTransformer</strong> — newer architectures (2023–2024) that treat time-series as patches and apply self-attention across features. Can outperform TFT on some benchmarks with less tuning.</li>
        <li><strong>Key design decision:</strong> Use timestamp-based positional encoding (actual datetime values) rather than position indices — this handles the missing-data gap between GDAM and the other markets gracefully.</li>
        <li><strong>Multi-task learning:</strong> Train one model on DAM + GDAM + RTM jointly, with a "market type" embedding token. This allows knowledge transfer — patterns learned from DAM's larger dataset improve GDAM predictions.</li>
      </ul>
    </div>

    <div class="phase" style="border-color:#8e44ad;">
      <h3>Phase 4 — Production, Probabilistic Forecasting & Research (Month 2+)</h3>
      <ul>
        <li><strong>Probabilistic output:</strong> Instead of predicting a single price, predict a range (10th, 50th, 90th percentile). Clients can use the uncertainty bands for risk-aware bidding decisions. Use quantile regression loss or conformalized predictions.</li>
        <li><strong>Price spike classifier:</strong> A separate binary classifier that predicts whether P(t) &gt; ₹10,000. Train on class-weighted data (spikes are rare, ~2–3% of data). This can trigger early warnings for procurement teams.</li>
        <li><strong>DISCOM load data integration:</strong> If actual metered demand from DISCOMs is available, it is the strongest demand-side signal possible. Even D-1 load forecasts from SLDC would be transformative for model accuracy.</li>
        <li><strong>Retraining cadence:</strong> Retrain weekly with a rolling window (drop data older than 2 years). Market behavior shifts over time — a model trained only on 2022 will underperform on 2026.</li>
        <li><strong>Publication angle:</strong> "Joint multi-market electricity price forecasting for Indian renewable energy certificates using weather-integrated sequence models" — the multi-market + weather + Indian context is a novel combination for a conference paper (e.g., IEEE PES, Applied Energy journal).</li>
      </ul>
    </div>

    <div class="callout" style="margin-top:24px;">
      <strong>🔑 5 Critical Implementation Rules:</strong><br/><br/>
      1. <strong>No data leakage</strong> — never use P(t) volume/price features in RTM at inference time. Only use lag-1 volumes. Build a leak-check function that verifies every feature column was computable before time t.<br/><br/>
      2. <strong>Normalise per market</strong> — GDAM (mean ₹5,422), DAM (₹4,924), RTM (₹4,669) have different scales. Z-score or min-max normalise separately for each market before training.<br/><br/>
      3. <strong>Handle the ₹20,000 cap</strong> — prices hitting exactly ₹20,000 are truncation artifacts (real price would be even higher). Consider <code>log(price + 1)</code> transformation or Winsorize at 99th percentile before training.<br/><br/>
      4. <strong>Weather forecast source</strong> — open-meteo.com provides free, high-resolution (hourly, 48h ahead) forecasts for any lat/long in India. For AP, use coordinates of the relevant solar zone (e.g., Kurnool: 15.83°N, 78.04°E). Fetch daily via API cron job.<br/><br/>
      5. <strong>Retraining cadence</strong> — retrain the model weekly with a rolling 2-year window. Indian electricity markets evolve (new solar capacity additions change the duck curve each year). A stale model will degrade fast.
    </div>
  </div>
</div>

</div><!-- end container -->
<footer>CarbonTatva AI · Feature Analysis Report v2 · March 2026 · Data: GDAM/DAM/RTM (IEX), Dec 2022 – Feb 2026 · Weather: Open-Meteo</footer>
</body>
</html>"""

HTML = HTML_TEMPLATE.format(
    fig1=b64['fig1'],   fig2=b64['fig2'],   fig3=b64['fig3'],
    fig4=b64['fig4'],   fig5=b64['fig5'],   fig6=b64['fig6'],
    fig7=b64['fig7'],   fig8=b64['fig8'],   fig9=b64['fig9'],
    fig10=b64['fig10'], fig11=b64['fig11'], fig12=b64['fig12'],
)

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"\n✅ Done! Report saved to: {REPORT_FILE}")
print(f"   Open it:  open {REPORT_FILE}")
