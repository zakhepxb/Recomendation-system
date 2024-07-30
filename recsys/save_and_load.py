import sys
from catboost import CatBoostClassifier
import pickle
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def save_ctb(model):
    logging.info('Save CatBoost')
    model.save_model('recsysctb_model')
    print('CatBoost model saved')

def load_ctb(model_path):
    logging.info('Load CatBoost')
    model = CatBoostClassifier()
    model.load_model(model_path)
    logging.info('CatBoost model loaded')

    return model

def save_lfm(model):
    logging.info('Save LightFM')
    with open('lfm_model.pickle', 'wb') as fle:
        pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('LightFM model saved')


def load_lfm(model_path):
    logging.info('Load LightFM')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logging.info('LightFM model loaded')

    return model


def save_maping(dictionary):
    with open('saved_dictionary.pickle', 'wb') as f:
        pickle.dump(dictionary, f)


def load_maping(maping_path):
    with open(maping_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict