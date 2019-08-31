import numpy as np
from sklearn.model_selection import StratifiedKFold as skf
from sklearn.model_selection import train_test_split as tts


def Cellwise_train_test_split(X, y, cells, test_size=0.33, random_state=71):
    X_train, X_test, y_train, y_test = [], [], [], []
    for cell in np.unique(cells):
        X_cell = X[cells == cell]
        y_cell = y[cells == cell]
        _X_train, _X_test, _y_train, _y_test = tts(
            X_cell, y_cell, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def CellwiseStratifiedKFold(
        X_df, y, n_splits=5, shuffle=False, random_state=71):
    cells = X_df.experiment.apply(lambda x: x.split('-')[0])
    cell_folds = []
    for cell in np.unique(cells):
        cell_df = X_df[cells == cell]
        cell_y = y[cells == cell]
        if cell_y.value_counts().min() < n_splits:
            cell_fold = skf(
                n_splits=int(cell_y.value_counts().min()),
                shuffle=shuffle,
                random_state=random_state).split(
                cell_df,
                cell_y)
        else:
            cell_fold = skf(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state).split(
                cell_df,
                cell_y)
        cell_folds.append(cell_fold)

    fold = [[[], []] for i in range(n_splits)]
    for cell_fold in cell_folds:
        for i, (trn_idx, val_idx) in enumerate(cell_fold):
            if i > 2:
                break
            fold[i][0].append(trn_idx)
            fold[i][1].append(val_idx)
    for i, _ in enumerate(fold):
        if len(fold[i][0]) > 0:
            fold[i][0] = np.concatenate(fold[i][0])
            fold[i][1] = np.concatenate(fold[i][1])

    return fold
