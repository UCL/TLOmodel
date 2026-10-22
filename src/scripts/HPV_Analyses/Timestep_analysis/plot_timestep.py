"""Separate figure-only script. Run after analysis_timestep.py."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('./outputs/time_step_convergence')
df = pd.read_csv(ROOT / 'timestep_series_long.csv', parse_dates=['date'])
metrics = ['overall_any', 'age_15_19', 'age_20_24', 'age_25_34',
           'age_35_44', 'age_45_54', 'age_55plus', 'P1', 'P2', 'P3', 'newinf_total']
fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
for ax, metric in zip(axes.flat, metrics):
    subset = df[df.metric == metric]
    for step in ['1_month', '2_weeks', '1_week']:
        part = subset[subset.step == step].sort_values('date')
        ax.plot(part.date, part.value, label=step, linewidth=1.25)
    ax.set_title(metric)
    ax.set_ylabel('Monthly events' if metric.startswith('newinf') else 'Prevalence')
    ax.grid(alpha=0.2)
for ax in list(axes.flat)[len(metrics):]:
    ax.axis('off')
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
fig.suptitle('HPV time-step comparison: monthly, 14-day, 7-day updates', y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(ROOT / 'timestep_comparison.png', dpi=250, bbox_inches='tight')
plt.close(fig)
print(ROOT / 'timestep_comparison.png')
