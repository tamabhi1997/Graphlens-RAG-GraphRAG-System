import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('~/Downloads/reliability_training_data_SAYALI.csv')

answered = df[df['refused'] == False]
refused  = df[df['refused'] == True]

fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(answered['best_similarity'], bins=10, alpha=0.75, 
        label=f'Answered (n={len(answered)})', color='steelblue')
ax.hist(refused['best_similarity'], bins=10, alpha=0.75, 
        label=f'Refused (n={len(refused)})', color='salmon')
ax.axvline(x=0.28, color='red', linestyle='--', linewidth=1.2, label='Refusal threshold (0.28)')
ax.set_xlabel('Best Similarity Score')
ax.set_ylabel('Count')
ax.set_title('Similarity Score Distribution: Answered vs. Refused Queries')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('evaluation/similarity_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('evaluation/similarity_distribution.png', dpi=300, bbox_inches='tight')
print("Saved")