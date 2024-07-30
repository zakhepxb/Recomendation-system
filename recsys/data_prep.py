import sys
import numpy as np
import pandas as pd
from lightfm.data import Dataset
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def sort_outliers(df):

    logging.info('Sort outliers for Dataset')

    exp = df[['user_id', 'item_id']]
    exp = exp.groupby(['user_id', 'item_id'])['user_id'].count()

    result = exp.index.to_frame(index=False)
    result['count'] = exp.values

    proc_5 = np.percentile(result['count'], 95)
    result_vib = result[result['count'] > proc_5]

    df = df[~df['user_id'].isin(result_vib['user_id'].unique())].reset_index(drop=True)

    logging.info(f"df: {df.shape}")

    return df

def split_dataset(df):

    logging.info('Split Dataset on Train Test and LightFM Train')

    df['order_ts'] = pd.to_datetime(df['order_ts'])

    max_date = df['order_ts'].max()
    train = df[(df['order_ts'] < max_date - pd.Timedelta(days=14))]
    test = df[(df['order_ts'] >= max_date - pd.Timedelta(days=14))]

    logging.info(f"train: {train.shape}")
    logging.info(f"test: {test.shape}")

    lfm_date_threshold = train['order_ts'].quantile(q=0.6, interpolation='nearest')

    lfm_train = train[(train['order_ts'] < lfm_date_threshold)]

    logging.info(f"lfm_train: {lfm_train.shape}")

    return test, train, lfm_train



def prepare_dataset_and_mapper(df):

    logging.info('Prepare Dataset and Mapper for LightFM')

    dataset = Dataset()
    dataset.fit(df['user_id'].unique(), df['item_id'].unique())

    lightfm_mapping = dataset.mapping()
    lightfm_mapping = {'users_mapping': lightfm_mapping[0], 'items_mapping': lightfm_mapping[2], }

    lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
    lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}

    return dataset, lightfm_mapping



def matrix_csr(df, dataset):
    logging.info('Create Matrix')
    interactions_matrix, weights_matrix = dataset.build_interactions(zip(*df[['user_id', 'item_id']].values.T))
    return weights_matrix.tocsr()