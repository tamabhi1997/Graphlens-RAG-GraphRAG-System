import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#ffffff')

# ── LEFT: Feature weights ──────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('#f8fafc')

features = [
    'answer_length', 'num_citations', 'citation_coverage',
    'best_similarity', 'mean_similarity', 'worst_similarity',
    'rerank_score', 'num_sources', 'expansion_ratio',
    'similarity_gap', 'expanded_chunks', 'graph_fired'
]
weights = [0.737, 0.711, 0.711, 0.406, 0.28, 0.16,
           -0.18, -0.24, -0.33, -0.41, -0.53, -0.72]

colors = ['#16a34a' if w > 0 else '#ea580c' for w in weights]
y_pos = np.arange(len(features))

bars = ax1.barh(y_pos, weights, color=colors, height=0.6, alpha=0.85)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(features, color='#334155', fontsize=9)
ax1.set_xlabel('Logistic Regression Coefficient', color='#1e293b', fontsize=9)
ax1.axvline(x=0, color='#cbd5e1', linewidth=0.8)
ax1.set_xlim(-1.0, 1.0)
ax1.tick_params(colors='#64748b', labelsize=8)
ax1.spines[:].set_color('#e2e8f0')
ax1.set_title('Feature Weights', color='#0f172a', fontsize=11, pad=10)

for bar, w in zip(bars, weights):
    x = w + 0.03 if w > 0 else w - 0.03
    ha = 'left' if w > 0 else 'right'
    ax1.text(x, bar.get_y() + bar.get_height()/2,
             f'{w:+.3f}', va='center', ha=ha,
             color='#334155', fontsize=7.5)

# ── RIGHT: Confusion matrix + metrics ─────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#ffffff')
ax2.set_aspect('equal')

labels = [['TP\n23', 'FP\n2'], ['FN\n2', 'TN\n23']]
cell_colors = [['#dcfce7', '#fff7ed'], ['#fff7ed', '#dcfce7']]
text_colors = [['#15803d', '#c2410c'], ['#c2410c', '#15803d']]

for i in range(2):
    for j in range(2):
        rect = mpatches.FancyBboxPatch(
            (j * 1.1, (1 - i) * 1.1), 1.0, 1.0,
            boxstyle="round,pad=0.05",
            facecolor=cell_colors[i][j],
            edgecolor='#e2e8f0', linewidth=1.5
        )
        ax2.add_patch(rect)
        ax2.text(j * 1.1 + 0.5, (1 - i) * 1.1 + 0.5,
                 labels[i][j], ha='center', va='center',
                 color=text_colors[i][j], fontsize=13, fontweight='bold')

ax2.set_xlim(-0.1, 2.3)
ax2.set_ylim(-0.8, 2.4)
ax2.axis('off')

ax2.text(0.5, 2.25, 'Predicted Positive', ha='center', color='#475569', fontsize=9)
ax2.text(1.6, 2.25, 'Predicted Negative', ha='center', color='#475569', fontsize=9)
ax2.text(-0.08, 1.6, 'Actual\nPositive', ha='center', color='#475569',
         fontsize=9, rotation=90, va='center')
ax2.text(-0.08, 0.5, 'Actual\nNegative', ha='center', color='#475569',
         fontsize=9, rotation=90, va='center')

ax2.set_title('Confusion Matrix  (n=50)', color='#0f172a', fontsize=11, pad=10)

metrics = [
    ('AUC', '0.984 ± 0.020', '#2563eb'),
    ('Accuracy', '0.92', '#16a34a'),
    ('Precision', '0.92', '#7c3aed'),
    ('Recall', '0.92', '#ea580c'),
]
for idx, (label, val, color) in enumerate(metrics):
    ax2.text(2.22, 1.9 - idx * 0.38, label, ha='left',
             color='#64748b', fontsize=9)
    ax2.text(2.22, 1.72 - idx * 0.38, val, ha='left',
             color=color, fontsize=11, fontweight='bold')

plt.suptitle('Reliability Model Performance', color='#0f172a',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('evaluation/reliability_model_slide.png', dpi=200,
            bbox_inches='tight', facecolor='#ffffff')
plt.savefig('evaluation/reliability_model_slide.pdf', dpi=200,
            bbox_inches='tight', facecolor='#ffffff')
print("Saved evaluation/reliability_model_slide.png + .pdf")