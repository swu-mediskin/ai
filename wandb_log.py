import wandb
import pandas as pd

pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)

api = wandb.Api()
run = api.run("jkjadeahn-university/mediskin/wg3v5hxn")

history_df = run.history(samples=10000) 

# best epoch 찾기
best_epoch = history_df['epoch'][history_df['mean_f1_score'].idxmax()]

cols = ['epoch', 'val_loss', 'train_loss', 'accuracy',
        'mean_f1_score', 'mean_recall', 'mean_precision','mean_specificity',
        'mean_auc', 'mean_auprc', 'ece', 'brier_score']

row = history_df[history_df['epoch'] == best_epoch][cols]

print(f"best epoch: {best_epoch}")
print(row.to_string(index=False))

# 클래스별 지표 (best epoch 기준)
classes = ['NEV', 'BCC', 'SCC', 'MEL']
class_metrics = ['recall', 'precision', 'f1_score', 'auc', 'auprc']

best_row = history_df[history_df['epoch'] == best_epoch].iloc[0]
class_table = pd.DataFrame(
    {m: [best_row.get(f'label_metrics/{c}_{m}') for c in classes] for m in class_metrics},
    index=classes,
)
class_table.index.name = 'class'

print()
print("클래스별 지표 (best epoch 기준)")
print(class_table.to_string())