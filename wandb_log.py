import wandb
import pandas as pd

pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)

api = wandb.Api()
run = api.run("jkjadeahn-university/mediskin/276dl1yl")

history_df = run.history(samples=10000) 

# best epoch 찾기
best_epoch = history_df['epoch'][history_df['val_loss'].idxmin()]

cols = ['epoch', 'val_loss', 'train_loss', 'accuracy',
        'mean_f1_score', 'mean_recall', 'mean_precision','mean_specificity',
        'mean_auc', 'mean_auprc', 'ece', 'brier_score']

row = history_df[history_df['epoch'] == best_epoch][cols]

print(f"best epoch: {best_epoch}")
print(row.to_string(index=False))