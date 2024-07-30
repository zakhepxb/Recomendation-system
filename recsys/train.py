import sys
from tqdm.auto import tqdm
from lightfm import LightFM
from catboost import CatBoostClassifier
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def train_lfm(hyperparams, train_matrix):
    logging.info('Train LightFM')
    lfm_model = LightFM(**hyperparams)

    for _ in tqdm(range(20)):
        lfm_model.fit_partial(train_matrix)
    logging.info('Train Finished')
    return lfm_model


def train_ctb(params, X_train, y_train, X_val, y_val):
    logging.info('Train CatBoost')
    model = CatBoostClassifier(**params)
    model.fit(X_train,
              y_train,
              eval_set=(X_val, y_val),
              early_stopping_rounds=100,
              plot=True)
    logging.info('Train Finished')
    return model
