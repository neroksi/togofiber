import numpy as np
import pandas as pd

from .config import CFG


def check_weights(sub_paths):
    S = sum([p["w"] for p in sub_paths])
    assert abs(1 - S) < 1e-5, S


def get_bag_prediction(sub_paths, use_order=False):
    check_weights(sub_paths)

    df = None
    for d in sub_paths:
        tp = pd.read_csv(d["path"]).set_index("ID", drop=True)

        if use_order:
            tp.loc[tp.index[np.argsort(tp["Target"].values)], "Target"] = np.arange(
                len(tp)
            )

        tp *= d["w"]
        if df is None:
            df = tp
        else:
            df += tp

    df = df.reset_index()

    return df


def main():
    q = 1 / 3
    sub_paths = [
        {
            "path": CFG.DATA_ROOT
            / "subs/sub_lgbm_2024-06-23T01-27-40Z++catb_2024-06-23T01-32-12Z.csv",
            "w": q,
        },
        {
            "path": CFG.DATA_ROOT
            / "subs/sub_lgbm_2024-06-23T02-08-53Z++catb_2024-06-23T02-12-49Z.csv",
            "w": q,
        },
        {
            "path": CFG.DATA_ROOT
            / "subs/sub_lgbm_2024-06-23T02-53-32Z++catb_2024-06-23T02-58-23Z.csv",
            "w": q,
        },
    ]

    sub = get_bag_prediction(sub_paths, use_order=CFG.USE_ORDER_IN_BAGGING)

    sub.to_csv(CFG.SUB_SAVE_ROOT / (CFG.SUB_FILENAME or "submission.csv"), index=False)


if __name__ == "__main__":

    main()
