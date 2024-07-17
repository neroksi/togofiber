import json
from time import time

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler

from .config import CFG
from .utils import print_duration


def add_training_fold(df, nfolds=5, strat_col="cluster_bucket", seed=42):
    np.random.seed(seed)

    df["fold"] = -1

    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

    for fold, (_, val_set) in enumerate(skf.split(np.arange(len(df)), y=df[strat_col])):
        df.loc[df.index[val_set], "fold"] = fold

    return df


def read_train_df(train_csv_path=None, test_csv_path=None, usecols=None, debug=False):
    T0 = time()

    train_csv_path = train_csv_path or CFG.TRAIN_CSV_PATH
    test_csv_path = test_csv_path or CFG.TEST_CSV_PATH

    df = pd.read_csv(
        train_csv_path,
        encoding="utf-8",
        usecols=usecols,
        nrows=400 if debug else None,
    )
    df["Split"] = "Train"

    df_test = pd.read_csv(
        test_csv_path,
        nrows=200 if debug else None,
        encoding="utf-8",
        usecols=(
            [col for col in usecols if col != "Accès internet"] if usecols else None
        ),
    )
    df_test["Split"] = "Test"

    df = pd.concat([df, df_test], axis=0, sort=False)
    df.reset_index(inplace=True, drop=True)

    df.rename(columns={"Accès internet": "Target", " ": " .0"}, inplace=True)

    print_duration(time() - T0, "df read duration:")

    return df


def load_folds(df, fold_dict_json_path=None, bucket_dict_json_path=None):
    fold_dict_json_path = fold_dict_json_path or CFG.FOLD_DICT_JSON_PATH
    bucket_dict_json_path = bucket_dict_json_path or CFG.BUCKET_DICT_JSON_PATH

    with open(fold_dict_json_path) as f:
        fold_dict = json.load(f)

    with open(bucket_dict_json_path) as f:
        cluster_dict = json.load(f)

    df["cluster_bucket"] = df["ID"].map(cluster_dict)
    df["fold"] = df["ID"].map(fold_dict).fillna(-1).astype(int)

    return df


def make_buckets(df, X=None, use_agg_cl=True, normalize=False):
    sat_cols = get_sat_cols(df)
    if use_agg_cl:
        cl = AgglomerativeClustering(n_clusters=300, linkage="complete")
    else:
        cl = KMeans(
            n_clusters=300,
            max_iter=1000,
            n_init=10,
            random_state=42,
            verbose=0,
            tol=5e-4,
        )

    if X is None:
        X = df[sat_cols].fillna(0)

    if normalize:
        X /= np.linalg.norm(X, axis=1, ord=2, keepdims=True)

    cl = cl.fit(X)

    cluster_dict = dict(zip(df["ID"], cl.labels_.tolist()))
    return cluster_dict


def make_dim_reducer(
    df, n_components=400, n_iter=100, random_state=42, drop_sat_cols=False, **kwargs
):
    T0 = time()

    dim_reducer = TruncatedSVD(
        n_components=n_components, n_iter=n_iter, random_state=random_state, **kwargs
    )

    sat_cols = get_sat_cols(df)
    df_red = dim_reducer.fit_transform(df[sat_cols].fillna(0))

    if drop_sat_cols:
        df.drop(sat_cols, axis=1, inplace=True)

    dim_reduction_cols = [f"__svd__{i:03d}" for i in range(dim_reducer.n_components)]

    df_red = pd.DataFrame(df_red, columns=dim_reduction_cols, index=df.index)
    df = pd.concat([df, df_red], axis=1, sort=False)

    print_duration(time() - T0, desc="dim_reducer duration")

    return dim_reduction_cols, dim_reducer, df


def get_sat_cols(df):
    return df.columns[df.columns.str.startswith(" .")].tolist()


def get_num_cols(df):
    return [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in CFG.EXCLUDE_COLS
    ]


def get_cat_cols(df):
    return [
        col
        for col in df.select_dtypes(include="object").columns
        if col not in CFG.EXCLUDE_COLS
    ]


def preprocess_data(
    df,
    sat_cols=None,
    num_cols=None,
    cat_cols=None,
    add_label_encoding=False,
    keep_cat_cols=False,
    reduce_dim=False,
    drop_sat_cols=False,
):
    T0 = time()

    sat_cols = sat_cols or get_sat_cols(df)
    num_cols = num_cols or get_num_cols(df)
    cat_cols = cat_cols or get_cat_cols(df)

    assert len(set(num_cols).intersection(sat_cols)) == len(sat_cols)

    all_cols = set(num_cols + cat_cols)
    if drop_sat_cols:
        all_cols = all_cols.difference(sat_cols)

    if reduce_dim:
        dim_reduction_cols, dim_reducer, df = make_dim_reducer(
            df, drop_sat_cols=drop_sat_cols
        )
    else:
        dim_reduction_cols = []
        dim_reducer = None

    all_cols = all_cols.union(dim_reduction_cols)

    all_cols = sorted(all_cols)

    cat_encoders = {col: LabelEncoder() for col in cat_cols}

    cat_to_numerical_map = {}

    stat_funcs = [
        "mean",
        "median",
    ]

    X = df[all_cols + ["Target"]].copy(deep=False)

    tp_list = [X]

    cat_embed_cols = []

    for icol in range(len(cat_cols) - 1, -1, -1):
        col = cat_cols[icol]

        X[col] = X[col].fillna("__NA__").astype(str)
        cat_encoder = cat_encoders[col].fit(X[col])
        X[col] = cat_encoder.transform(X[col])

        if add_label_encoding:
            tp = X[[col, "TAILLE_MENAGE", "Target"]].groupby(col).agg(stat_funcs)
            tp.columns = [f"__cat__{col}__{c}" for c in map("__".join, tp.columns)]
            cat_to_numerical_map[col] = tp
            cat_embed_cols.extend(tp.columns)

        if len(cat_encoder.classes_) <= 2:
            cat_cols.pop(icol)
            num_cols.append(col)
            X[col] = X[col].astype(np.float32)
        else:
            tp_list.append(
                pd.get_dummies(
                    X[col],
                    dummy_na=False,
                    prefix=f"__dummy__{col}",
                    prefix_sep="__",
                    dtype=np.int32,
                    drop_first=False,
                )
            )

            # if col.lower() == "connexion":
            #     tp_list[-1] *= 10

    if add_label_encoding:
        for col in cat_encoders:
            tp = cat_to_numerical_map[col].loc[X[col].values]
            tp.index = df.index
            tp_list.append(tp)

    X = pd.concat(tp_list, axis=1, sort=False, ignore_index=False)
    X.drop(["Target"], axis=1, inplace=True)

    if not keep_cat_cols:
        X.drop(cat_cols, axis=1, inplace=True)

    simple_num_cols = [
        col
        for col in X.columns
        if (col not in cat_cols)
        and (col not in sat_cols)
        and (col not in cat_embed_cols)
        and (col not in dim_reduction_cols)
    ]

    all_num_cols = sorted(
        set(X.columns).difference(cat_cols).difference(CFG.EXCLUDE_COLS)
    )
    assert len(cat_embed_cols) + len(simple_num_cols) + (
        0 if drop_sat_cols else len(sat_cols)
    ) + len(dim_reduction_cols) == len(all_num_cols), {
        "simple_num_cols": len(simple_num_cols),
        "sat_cols": len(sat_cols),
        "cat_embed_cols": len(cat_embed_cols),
        "dim_reduction_cols": len(dim_reduction_cols),
        "all_num_cols": len(all_num_cols),
    }

    res = {
        "X": X,
        "cat_encoders": cat_encoders,
        "dim_reducer": dim_reducer,
        "kwargs": {
            "add_label_encoding": add_label_encoding,
            "keep_cat_cols": keep_cat_cols,
            "reduce_dim": reduce_dim,
            "drop_sat_cols": drop_sat_cols,
        },
        "cols": {
            "sat_cols": sat_cols,
            "cat_cols": cat_cols,
            # "num_cols": num_cols,
            "cat_embed_cols": cat_embed_cols,
            "simple_num_cols": simple_num_cols,
            "dim_reduction_cols": dim_reduction_cols,
            "all_num_cols": all_num_cols,
        },
    }

    print_duration(time() - T0, desc="preprocess_data duration")

    return res


def finalize_data_prep(
    df,
    add_label_encoding=False,
    keep_cat_cols=False,
    reduce_dim=False,
    drop_sat_cols=False,
    normalize=True,
):
    T0 = time()

    data = preprocess_data(
        df,
        add_label_encoding=add_label_encoding,
        keep_cat_cols=keep_cat_cols,
        reduce_dim=reduce_dim,
        drop_sat_cols=drop_sat_cols,
    )

    data["kwargs"]["normalize"] = normalize

    X = data.pop("X")

    assert (X.index.values == df.index.values).all()

    sat_cols = data["cols"]["sat_cols"]
    # cat_cols = data["cols"]["cat_cols"]
    all_num_cols = data["cols"]["all_num_cols"]
    cat_embed_cols = data["cols"]["cat_embed_cols"]
    simple_num_cols = data["cols"]["simple_num_cols"]
    dim_reduction_cols = data["cols"]["dim_reduction_cols"]

    scaler = RobustScaler()

    if normalize and not drop_sat_cols:
        X[sat_cols] /= np.linalg.norm(X[sat_cols].values, axis=1, ord=2, keepdims=True)

    if normalize:
        X[simple_num_cols] /= np.linalg.norm(
            X[simple_num_cols].values, axis=1, ord=2, keepdims=True
        )

    if normalize and add_label_encoding:
        X[cat_embed_cols] /= np.linalg.norm(
            X[cat_embed_cols].values, axis=1, ord=2, keepdims=True
        )

    if normalize and reduce_dim:
        X[dim_reduction_cols] /= np.linalg.norm(
            X[dim_reduction_cols].values, axis=1, ord=2, keepdims=True
        )

    scaler = scaler.fit(X[all_num_cols])
    X[all_num_cols] = scaler.transform(X[all_num_cols].fillna(0))

    train_cols = X.columns
    # train_cols = [col for col in X.columns if col not in cat_cols]
    is_train_bools = df["Split"] == "Train"
    X_train = X.loc[is_train_bools, train_cols]
    X_test = X.loc[~is_train_bools, train_cols]
    folds = df.loc[is_train_bools, "fold"].values

    Y_train = df.loc[is_train_bools, "Target"].fillna(0).values  # .astype(int)

    print("shapes:", X.shape, X_test.shape, (X_train.shape, Y_train.shape), folds.shape)

    data["cols"]["train_cols"] = train_cols
    data["X_train"] = X_train
    data["X_test"] = X_test
    data["Y_train"] = Y_train
    data["folds"] = folds
    data["scaler"] = scaler

    print_duration(time() - T0, desc="finalize_data_prep duration")

    return data


def rescale_data(data):
    data["kwargs"]["CFG.COL_WEIGHTS"] = CFG.COL_WEIGHTS

    for col, col_w in CFG.COL_WEIGHTS.items():
        cols = data["cols"][col]
        if len(cols) and len(set(data["X_train"].columns).intersection(cols)):
            data["X_train"][cols] *= col_w
            data["X_test"][cols] *= col_w

    return data


def read_psl(df, data):
    df_sub = pd.read_csv(CFG.TEST_PSL_FILE_PATH)
    test_ids = df.loc[data["X_test"].index, "ID"]
    target = df_sub["Target"]
    # target = 1*(df_sub["Target"] > 0.5)
    psl = test_ids.map(dict(zip(df_sub["ID"], target))).values

    data["Y_test_psl"] = psl

    return psl


def extend_data_for_psl(df, data):
    data["X_train"] = pd.concat([data["X_train"], data["X_test"]], axis=0)
    data["Y_train"] = np.concatenate([data["Y_train"], read_psl(df=df, data=data)])
    data["folds"] = np.concatenate([data["folds"], [-1] * len(data["X_test"])])
    return df, data


def prepare_data(debug=False, rescale=True):
    df = read_train_df(debug=debug)
    df = load_folds(df)
    data = finalize_data_prep(
        df,
        add_label_encoding=False,
        keep_cat_cols=False,
        reduce_dim=True,
        drop_sat_cols=True,
        normalize=True,
    )

    if rescale:
        data = rescale_data(data)

    return df, data


def build_clusters_and_folds(debug=False, save=False):
    """
    Build a kfold stratified validation based on custom computed clusters.

    !!Carefull!! : would take a bit of time to run for the whole data set (~30 mins).
    Hopefully, this will just be run once during all the project development.

    Parameters
    ----------
    debug: bool (default=False)
        Wether one is in debug mode (fast testing)
    save: bool (default=False)
        Wether to save resulting folder_dict and cluster_dict

    Returns
    -------
    cluster_dict : dict
        The dict mapping each id to its cluster
    fold_dict: dict
        The dict mapping each id to its fold
    """
    t0 = time()

    # df, data = prepare_data(debug=debug, rescale=False)

    df = read_train_df(debug=debug)
    # df = load_folds(df)
    df["fold"] = -1
    data = finalize_data_prep(
        df,
        add_label_encoding=False,
        keep_cat_cols=False,
        reduce_dim=False,
        drop_sat_cols=False,
        normalize=True,
    )

    X = pd.concat([data["X_train"], data["X_test"]], axis=0).loc[df.index]
    print(f"X.shape: {X.shape}")

    cluster_dict = make_buckets(df=df, X=X, normalize=False, use_agg_cl=True)

    df["cluster_bucket"] = df["ID"].map(cluster_dict)
    df["fold"] = add_training_fold(df[["cluster_bucket"]].copy())["fold"]

    print("Folds Value Counts:\n", df["fold"].value_counts())

    fold_dict = dict(zip(df["ID"], df["fold"]))

    print(f"len(cluster_dict): {len(cluster_dict)}    len(fold_dict): {len(fold_dict)}")

    if save:
        with open(CFG.DATA_ROOT / "fold_dict.json", "w") as f:
            json.dump(fold_dict, f)

        with open(CFG.DATA_ROOT / "cluster_dict.json", "w") as f:
            json.dump(cluster_dict, f)

    print_duration(time() - t0, desc="Folds & Clusters Duration")

    return cluster_dict, fold_dict
