from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'raw_placement_data.csv'
OUT = ROOT / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'branch',
    'college_tier',
    'cgpa',
    'backlogs',
    'coding_skill_score',
    'communication_skill_score',
    'internships_count',
    'projects_count'
]
TARGET = 'placement_status'

def load():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    df = df[FEATURES + [TARGET]].dropna()
    if not pd.api.types.is_numeric_dtype(df['college_tier']):
        df['college_tier'] = (
            df['college_tier']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )
    X = pd.get_dummies(df[FEATURES], columns=['branch'], dtype=float)
    y = df[TARGET]
    if not pd.api.types.is_numeric_dtype(y):
        labels = sorted(y.astype(str).unique())
        if len(labels) != 2:
            raise ValueError('placement_status must be binary')
        y = y.astype(str).map({labels[0]: 0, labels[1]: 1})
    return X, y

def run():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    for name, criterion in [('gini', 'gini'), ('entropy', 'entropy')]:
        m = DecisionTreeClassifier(criterion=criterion, random_state=42, max_depth=5).fit(Xtr, ytr)
        print(name, 'test accuracy:', round(accuracy_score(yte, m.predict(Xte)), 4))
        plt.figure(figsize=(18, 10))
        plot_tree(
            m,
            feature_names=X.columns.tolist(),
            class_names=[str(x) for x in sorted(y.unique())],
            filled=True,
            max_depth=4,
            fontsize=7
        )
        plt.title(f'Placement Decision Tree - {name.title()}')
        plt.tight_layout()
        plt.savefig(OUT / f'decision_tree_{name}.png', dpi=150)
        plt.close()

if __name__ == '__main__':
    run()