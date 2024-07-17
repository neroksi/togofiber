from pathlib import Path
from typing import ClassVar


class CFG:
    DEBUG: bool = False

    MODEL_ROOT: Path = Path(__file__).parents[2] / "models"
    DATA_ROOT: Path = Path(__file__).parents[2] / "data"
    SUB_SAVE_ROOT: Path = DATA_ROOT
    # SUB_FILENAME: str = None
    SUB_FILENAME = "submission.csv"
    TRAIN_CSV_PATH = DATA_ROOT / "Train_Full.csv"
    TEST_CSV_PATH = DATA_ROOT / "Test.csv"
    FOLD_DICT_JSON_PATH: Path = DATA_ROOT / "fold_dict_skf_aggclust_240517.json"
    BUCKET_DICT_JSON_PATH: Path = DATA_ROOT / "cluster_dict_aggclust_240517.json"

    EXCLUDE_COLS: ClassVar[list] = [
        "ID",
        "Target",
        "fold",
        "BoxLabel",
        "Split",
        "cluster_bucket",
    ]

    USE_ORDER_IN_BAGGING: bool = False

    COL_WEIGHTS: ClassVar[dict] = {
        "sat_cols": 1,
        "cat_embed_cols": 1,
        "dim_reduction_cols": 2,
        "simple_num_cols": 20,
    }

    DO_PSL: bool = False
    TEST_PSL_FILE_PATH: Path = None
    # TEST_PSL_FILE_PATH: Path = (
    #     DATA_ROOT / "psl/sub_lgbm_2024-06-22T17-40-46Z++catb_2024-06-22T17-44-42Z.csv"
    # )

    KNN_FEATS_PARAMS: ClassVar[dict] = {
        "use_target": False,
        "use_psl_as_test_target": False,
        "knn_params": {
            # "n_neighbors": 121,
            # "n_neighbors": 201,
            "n_neighbors": 81,
        },
    }

    LGBM_PARAMS: ClassVar[dict] = {
        "name": "lgbm",
        "params": {
            "num_leaves": 64,
            "min_data_in_leaf": 128,
            "objective": "binary",
            "boosting": "gbdt",
            "num_iterations": 200,
            "learning_rate": 0.075,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
            "max_depth": 16,
            # "early_stopping_round": 100,
            # "lambda_l1": 10.0,
            # "lambda_l2": 0.75,
        },
    }

    CATB_PARAMS: ClassVar[dict] = {
        "name": "catb",
        "params": {
            "iterations": 250,
            "depth": 8,
            "learning_rate": 0.075,
            "loss_function": "Logloss",
            "verbose": True,
            "min_data_in_leaf": 128,
            # "max_leaves": 32,
            # "colsample_bylevel": 0.75,
        },
    }

    @classmethod
    def as_dict(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and (k != "as_dict")
        }
