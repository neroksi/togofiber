import pickle
from copy import deepcopy
from datetime import datetime
from time import time

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from tqdm.auto import tqdm

from .config import CFG
from .train_catboost import train_kfold as catb_train_kfold
from .train_lgbm import train_kfold as lgbm_train_kfold
from .utils import print_duration


def utcnow():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def save_model(model_name, model_info, filename=None):
    filename = filename or f"{model_name}_{utcnow()}.bin"
    model_info["model"]["model_filename"] = filename

    with open(CFG.MODEL_ROOT / filename, "wb") as f:
        pickle.dump(model_info, f)


def train_and_save(data: dict, model_params: dict, model_trainer: callable, save=True):
    t0 = time()

    model_params = deepcopy(model_params)

    X_train = data["X_train"]
    Y_train = np.asarray(data["Y_train"], dtype=float)
    folds = np.asarray(data["folds"], dtype=int)

    info = {
        "data": {
            "scaler": data["scaler"],
            "dim_reducer": data["dim_reducer"],
            "cols": data["cols"],
            "kwargs": data["kwargs"],
        },
        "model": {"folds": folds, "model_params": model_params, "classifier": None},
    }

    print(X_train.shape, Y_train.shape)
    print(Y_train[:5])

    clfs, all_idx, oof, oof_label = model_trainer(
        X=X_train,
        y=Y_train,
        folds=folds,
        params=model_params["params"],
        # cat_feats=None,
        cat_feats=data["cols"]["cat_cols"] if data["kwargs"]["keep_cat_cols"] else None,
        early_stopping_rounds=30,
    )

    report = classification_report(
        oof_label,
        oof >= 0.5,
        digits=3,
    )
    print(report)

    auc = roc_auc_score(oof_label, oof)
    print("auc:", auc)

    info["model"]["classifier"] = {
        "config": CFG.as_dict(),
        "clfs": clfs,
        "all_idx": all_idx,
        "oof": oof,
        "oof_label": oof_label,
        "metrics": {
            "report": report,
            "auc": auc,
        },
    }

    if save:
        filename = None
        # filename = model_params.get("slug")
        # filename = f"{filename}.bin" if filename else None
        save_model(model_params["name"], info, filename=filename)

    print_duration(time() - t0, "MODEL TRAINING")
    return info


def catb_train_and_save(data: dict, save=True):
    return train_and_save(
        data, model_params=CFG.CATB_PARAMS, model_trainer=catb_train_kfold, save=save
    )


def lgbm_train_and_save(data: dict, save=True):
    return train_and_save(
        data, model_params=CFG.LGBM_PARAMS, model_trainer=lgbm_train_kfold, save=save
    )


def predict(df, data, model_info):
    test_preds = 0.0

    clfs = model_info["model"]["classifier"]["clfs"]

    for clf in tqdm(clfs):
        if hasattr(clf, "predict_proba"):
            tp = clf.predict_proba(data["X_test"])
        else:
            tp = clf.predict(data["X_test"])

        tp = tp[:, 1] if tp.ndim == 2 else tp

        test_preds += tp

    test_preds /= len(clfs)

    sub = df.loc[data["X_test"].index, ["ID"]].copy()
    sub["Target"] = test_preds

    return sub
