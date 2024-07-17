from time import time

from tqdm.auto import tqdm

from togofiber.bagging import get_bag_prediction
from togofiber.config import CFG
from togofiber.dataset import extend_data_for_psl, prepare_data
from togofiber.knn_feats import train_and_extract_knn_feats
from togofiber.training import catb_train_and_save, lgbm_train_and_save, predict
from togofiber.utils import print_duration


def train_and_predict(df, data, trainer, sub_save_path=None):
    model_info = trainer(data, save=True)

    sub = predict(df=df, data=data, model_info=model_info)

    sub_save_path = sub_save_path or (
        CFG.SUB_SAVE_ROOT / f'sub_{model_info["model"]["model_filename"]}'
    ).with_suffix(".csv")

    sub.to_csv(sub_save_path, index=False, float_format="%.5f")

    model_info["sub_save_path"] = sub_save_path

    return model_info, sub


def main(debug=False):
    t0 = time()

    df, data = prepare_data(debug=debug)
    df, data = train_and_extract_knn_feats(df=df, data=data)

    if CFG.DO_PSL:
        df, data = extend_data_for_psl(df=df, data=data)

    sub_paths = []

    for trainer, w in tqdm(
        [
            (lgbm_train_and_save, 0.5),
            (catb_train_and_save, 0.5),
        ]
    ):
        model_info, sub = train_and_predict(df=df, data=data, trainer=trainer)

        print(sub.head())

        sub_paths.append(
            {
                "path": model_info["sub_save_path"],
                "w": w,
            }
        )

    sub = get_bag_prediction(sub_paths, use_order=CFG.USE_ORDER_IN_BAGGING)

    sub_filename = CFG.SUB_FILENAME
    if not sub_filename:
        sub_filename = "++".join(
            [d["path"].stem.replace("sub_", "") for d in sub_paths]
        )
        sub_filename = f"sub_{sub_filename}.csv"

    sub.to_csv(CFG.SUB_SAVE_ROOT / sub_filename, index=False, float_format="%.5f")

    print_duration(time() - t0, desc="Global Duration")


if __name__ == "__main__":
    main(debug=CFG.DEBUG)
