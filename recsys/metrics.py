import sys
import numpy as np
import pandas as pd
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def ap_k(x):
    rank = x['rank']
    count = 0
    for_mean = x.values[:-1]

    for i, val in enumerate(x.values[:-1]):
        if i + 1 not in rank:
            for_mean[i] = 0.0
        else:
            count += 1

    if count == 0:
        return 0.0
    else:
        return for_mean.sum() / count


def calc_metrics_precission(actual, predicted, K, rank='rank'):
    logging.info('Count Precission and MAP')
    rating_df = actual.set_index(['user_id', 'item_id']).join(predicted.set_index(['user_id', 'item_id']))

    rating_df = rating_df.sort_values(by=['user_id', rank]).reset_index()

    rating_df = rating_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

    for k in range(1, K + 1):
        rating_df[f'precision@{k}'] = rating_df[rank] <= k
        rating_df[f'precision@{k}'] = rating_df[f'precision@{k}'] / k

    df_lists = rating_df.groupby(['user_id'])[rank].apply(set).reset_index()

    rating_df = rating_df.drop(columns=['item_id', 'order_ts', rank], axis=1)
    rating_df = rating_df.groupby(['user_id']).sum().reset_index()

    rating_df = rating_df.set_index(['user_id']).join(df_lists.set_index(['user_id']))

    rating_df['AP@K'] = rating_df.apply(ap_k, axis=1)
    rating_df = rating_df.drop(columns=[rank], axis=1)

    return rating_df


def calc_metric_mrr(actual, predicted, K, rank='rank'):
    logging.info('Count MRR')
    rating_df = actual.set_index(['user_id', 'item_id']).join(predicted.set_index(['user_id', 'item_id']))
    rating_df = rating_df.sort_values(by=['user_id', rank]).reset_index()
    rating_df = rating_df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    rating_df['reciprocal_rank'] = (1 / rating_df[rank]).fillna(0)
    result = rating_df.groupby(['user_id'])['reciprocal_rank'].max()

    return result

def count_metrics(df, prediction):
    logging.info('Count Metrics')
    final_metrics_precission = calc_metrics_precission(df, prediction, 10).mean(axis=0)
    final_metrics_mrr = calc_metric_mrr(df, prediction, 10).mean()

    logging.info(f'MRR: {final_metrics_mrr}')
    logging.info(f'MAP: {final_metrics_precission["AP@K"]}')
    logging.info(f'Precision@10: {final_metrics_precission["precision@10"]}')
