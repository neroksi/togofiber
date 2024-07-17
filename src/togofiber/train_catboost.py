import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .train_lgbm import train_test_split


class AUCScore:
    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        y_pred = np.array(approxes, dtype=np.float32)
        y_true = (np.array(target, dtype=np.float32) > 0.5).astype(int)

        # print("catb shape:", y_pred.shape, y_true.shape)
        # print("catb sample:", y_pred[:5], y_true[:5])

        score = roc_auc_score(y_true.ravel(), y_pred.ravel())

        return score, 1

    def get_final_error(self, error, weight):
        return error


def train_one(
    X,
    y,
    folds,
    params,
    fold,
    cat_feats=None,
    early_stopping_rounds=30,
):
    (test_idx, test_data), (train_idx, train_data) = train_test_split(
        X=X, y=y, folds=folds, fold=fold
    )

    catboost_train_data = Pool(
        train_data[0], label=train_data[1], cat_features=cat_feats
    )
    catboost_test_data = Pool(test_data[0], label=test_data[1], cat_features=cat_feats)

    tlabel = test_data[1]

    model = CatBoostClassifier(eval_metric=AUCScore(), **params)

    clf = model.fit(
        catboost_train_data,
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True,
        eval_set=catboost_test_data,
        logging_level="Verbose",
        metric_period=10,
    )

    test_preds = clf.predict(test_data[0], prediction_type="RawFormulaVal")

    return clf, test_idx, test_preds, tlabel


def train_kfold(
    X,
    y,
    folds,
    params,
    cat_feats=None,
    early_stopping_rounds=30,
):
    if params["loss_function"].lower() == "logloss":
        y = (y > 0.5).astype(int)

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

        print(
            "auc-fold{}: {:.3f}".format(
                fold,
                roc_auc_score(label.reshape(-1), test_preds.reshape(-1)),
            )
        )

        clfs.append(clf)
        all_idx.append(idx)
        oof.append(test_preds)
        oof_label.append(label)

    all_idx = np.concatenate(all_idx)
    oof = np.concatenate(oof)
    oof_label = np.concatenate(oof_label)
    return clfs, all_idx, oof, oof_label
