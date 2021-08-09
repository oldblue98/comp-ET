import os
import argparse
import datetime
import logging
import json
import pandas as pd
import os
import torch
from torch import nn
#import cudf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold

from src.augmentation import *
from src.dataset import *
from src.utils import *
from src.model import *

import warnings
warnings.filterwarnings('ignore')


def main():
    # config file upload
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    # log file
    now = datetime.datetime.now()
    logging.basicConfig(
        filename='./logs/infer_' + config["model_name"] + '_'+ '{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
    )
    logging.debug('infer')
    logging.debug('date : {0:%Y,%m/%d,%H:%M:%S}'.format(now))
    log_list = ["img_size", "train_bs", "monitor"]

    for log_c in log_list:
        logging.debug(f"{log_c} : {config[log_c]}")

    # train 用 df の作成
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    df, image_paths = read_dataset()
    df_test, test_paths = read_test_dataset()

    train_df["image_path"] = image_paths
    train_df["target"] = df["target"]
    train_df["id"] = df["id"]

    test_df["image_path"] = test_paths
    test_df["target"] = df_test["target"]
    test_df["id"] = df_test["id"]

    del df

    # le = LabelEncoder()
    # train_df.label = le.fit_transform(train_df.label)

    # modelの作成
    seed_everything(config['seed'])
    device = torch.device(config['device'])

    for epoch in range(config["epochs"]-3, config["epochs"]):

        print(f'inference epoch{epoch} start')

        model = ImageModel(
                    train_df.label.nunique(),
                    config["model_name"],
                    config["model_type"],
                    config["fc_dim"],
                    config["margin"],
                    config["scale"],
                    device,
                    training=False
                )

        model.eval()

        # dataset, dataloafer作成
        folds = StratifiedKFold(
                    n_splits=config['fold_num'],
                    shuffle=True,
                    random_state=config['seed']).split(np.arange(train_df.shape[0]),
                    train_df.label.values
                )
        preds = []
        for fold, (trn_idx, val_idx) in enumerate(folds):
            if fold > 0: # 時間がかかるので最初のモデルのみ
                break
            model.load_state_dict(torch.load(f'save/{config["model_name"]}_epoch{epoch}_fold{fold}.pth'))
            model = model.to(device)

            valid_ = train_df.loc[val_idx,:].reset_index(drop=True)

            valid_ds = ImageDataset(valid_, transforms=get_valid_transforms(config["img_size"]))
            test_ds = ImageDataset(test_df, transforms=get_valid_transforms(config["img_size"]))

            valid_loader = torch.utils.data.DataLoader(
                valid_ds,
                batch_size=config["valid_bs"],
                num_workers=config["num_workers"],
                shuffle=False,
                pin_memory=True,
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=config["valid_bs"],
                num_workers=config["num_workers"],
                shuffle=False,
                pin_memory=True,
            )

            valid_predictions = get_prediction(model, valid_loader, device)
            test_prediction = get_prediction(model, test_loader, device)
            del model
            valid_["oof"] = valid_predictions
            valid_['roc'] = valid_.apply(getMetric('oof'),axis=1)

            logging.debug(f"{epoch} epoch, {fold} fold")
            logging.debug(f" CV_score : {valid_.roc.mean()}")
            # logging.debug(f" scores : {scores.mean()}")

    del model, valid_loader, valid_predictions

if __name__ == '__main__':
    main()