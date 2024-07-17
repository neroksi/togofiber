from time import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors

from .config import CFG
from .dataset import read_psl
from .utils import print_duration


def inv_distance(x, exp=0.5):
    x **= exp
    x = 1 / (1e-8 + x)
    x /= x.sum(axis=1, keepdims=True)
    return x


def distance(x, temp=10.0):
    x = inv_distance(x, exp=0.5)
    x = softmax(temp * x, axis=1)
    return x


def get_avg_feats(feats, dist, index, knn_feats_cols, step=10):
    k = feats.shape[1]
    nf = feats.shape[2]

    assert step is None or step <= k
    cols = []
    steps = [k] if step is None else list(range(step, k + 1, step))

    print(f"k: {k} nf:{nf} nsteps: {len(steps)}")

    if steps[-1] < k:
        steps.append(k)

    r = np.zeros((len(feats), len(steps) * nf), dtype=np.float32)
    for ii, i in enumerate(steps):
        d = dist[:, :i]
        # d = dist[:, :i] / (1e-8 + np.sum(dist[:, :i], axis=1, keepdims=True))

        if not CFG.KNN_FEATS_PARAMS["use_target"]:
            avg = np.sum(d * feats[:, :i], axis=1)
        else:
            is_not_null = np.logical_not(np.isnan(feats[:, :i]))
            d = d * is_not_null
            d /= np.sum(d, axis=1, keepdims=True)
            avg = np.sum(
                d * is_not_null * np.nan_to_num(feats[:, :i], nan=0.5, copy=True),
                axis=1,
            )

        cols += [f"__neigh_avg{ii:02d}__{col}" for col in knn_feats_cols]
        r[:, ii * nf : (ii + 1) * nf] = avg

    r = pd.DataFrame(r, columns=cols, index=index)

    return r


def get_neigh_feats(X, X_pred, knn, knn_cols, knn_feats_cols, is_train=False):

    dist, neigh = knn.kneighbors(X_pred[knn_cols])

    tp = X[knn_feats_cols].values
    shape = (len(neigh), neigh.shape[1], tp.shape[1])
    tp = tp[neigh.ravel()].reshape(shape)

    dist = inv_distance(dist)[:, :, None]

    tp = get_avg_feats(
        feats=tp,
        dist=dist,
        index=X_pred.index,
        step=None,
        # step=20,
        knn_feats_cols=knn_feats_cols,
    )

    return tp


def get_psl(df, data):
    if not CFG.KNN_FEATS_PARAMS["use_psl_as_test_target"]:
        return np.full(
            len(data["X_test"]), fill_value=np.nan, dtype=data["Y_train"].dtype
        )
    assert (
        CFG.TEST_PSL_FILE_PATH
    ), "`use_psl_as_test_target` is True but `TEST_PSL_FILE_PATH` is not set !"
    return read_psl(df=df, data=data)


def train_and_extract_knn_feats(df, data):
    t0 = time()

    # knn_cols = data["cols"]["dim_reduction_cols"]
    knn_cols = data["X_train"].columns.tolist()

    knn_feats_cols = data["cols"]["simple_num_cols"]
    if CFG.KNN_FEATS_PARAMS["use_target"]:
        knn_feats_cols.append("Target")

    X = pd.concat([data["X_train"], data["X_test"]], axis=0).loc[df.index]

    if CFG.KNN_FEATS_PARAMS["use_target"]:
        idx = np.concatenate([data["X_train"].index, data["X_test"].index])
        X.loc[idx, "Target"] = np.concatenate(
            [data["Y_train"], get_psl(df=df, data=data)]
        )

    knn = NearestNeighbors(**CFG.KNN_FEATS_PARAMS["knn_params"])
    knn = knn.fit(X[knn_cols])

    tp = get_neigh_feats(
        X,
        data["X_train"],
        is_train=True,
        knn=knn,
        knn_cols=knn_cols,
        knn_feats_cols=knn_feats_cols,
    )
    data["X_train"] = pd.concat([data["X_train"], tp], axis=1)

    tp = get_neigh_feats(
        X,
        data["X_test"],
        is_train=False,
        knn=knn,
        knn_cols=knn_cols,
        knn_feats_cols=knn_feats_cols,
    )
    data["X_test"] = pd.concat([data["X_test"], tp], axis=1)

    print("KNN_FEATS:", data["X_train"].shape, data["X_test"].shape)

    data["cols"]["knn_cols"] = knn_cols
    data["cols"]["knn_feats_cols"] = knn_feats_cols
    data["knn_feats"] = knn

    print_duration(time() - t0, desc="Train & Extract KNN Feats")

    return df, data
