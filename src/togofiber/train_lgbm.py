import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


def auc_score(preds: np.ndarray, data: lgb.Dataset):

    labels = (data.get_label().ravel().round() > 0.5).astype(int)
    # print("LGBM:", labels.shape, preds.shape)

    preds = preds.ravel()

    try:
        score = roc_auc_score(
            y_true=labels,
            y_score=preds,
        )
    except ValueError:
        score = 0.0

    return "AUC", score, True


def train_test_split(X, y, folds, fold):
    print(fold, X.shape, y.shape, folds.shape)
    bools = folds == fold
    is_in_all_vals = folds == -100

    test_idx = np.where(bools | is_in_all_vals)[0]
    test_data = (X.iloc[test_idx], y[test_idx])

    # train_idx = np.where(~bools | is_in_all_vals)[0]
    train_idx = np.where(~bools & ~is_in_all_vals)[0]
    train_data = (X.iloc[train_idx], y[train_idx])

    return (test_idx, test_data), (train_idx, train_data)


class LRScheduler:
    def __init__(self, init_lr: float = None):
        self.init_lr = init_lr or 0.1

    def __call__(self, iter):
        return self.init_lr * (0.99**iter)


def train_one(
    X,
    y,
    folds,
    params,
    fold,
    lr_scheduler=None,
    init_lr=None,
    cat_feats=None,
    early_stopping_rounds=30,
):
    lr_scheduler = lr_scheduler or LRScheduler(init_lr)
    (test_idx, test_data), (train_idx, train_data) = train_test_split(
        X=X, y=y, folds=folds, fold=fold
    )

    lgb_train_data = lgb.Dataset(train_data[0], label=train_data[1])
    lgb_test_data = lgb.Dataset(test_data[0], label=test_data[1])

    tlabel = test_data[1]
    # tlabel = np.array([class_map[t] for t in tlabel], dtype=np.float32)

    clf = lgb.train(
        params,
        lgb_train_data,
        valid_sets=[lgb_train_data, lgb_test_data],
        feval=auc_score,
        learning_rates=lr_scheduler,
        verbose_eval=10,
        early_stopping_rounds=early_stopping_rounds,
        **({"categorical_feature": cat_feats} if cat_feats else {}),
    )

    test_preds = clf.predict(
        test_data[0],
    )

    return clf, test_idx, test_preds, tlabel


def train_kfold(
    X,
    y,
    folds,
    params,
    suffix=None,
    cat_feats=None,
    early_stopping_rounds=30,
):

    suffix = suffix or ""
    clfs = []
    fmax = folds.max() + 1
    all_idx, oof, oof_label = [], [], []

    for fold in tqdm(list(range(fmax))):

        clf, idx, test_preds, label = train_one(
            X=X,
            y=y,
            folds=folds,
            fold=fold,
            params=params,
            cat_feats=cat_feats,
            early_stopping_rounds=early_stopping_rounds,
        )

        clfs.append(clf)
        all_idx.append(idx)
        oof.append(test_preds)
        oof_label.append(label)

    all_idx = np.concatenate(all_idx)
    oof = np.concatenate(oof)
    oof_label = np.concatenate(oof_label)
    return clfs, all_idx, oof, oof_label
