import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
tqdm.pandas()


def cand_lfm_pos(model, item_ids, N,
                 user_mapping, item_inv_mapping,
                 num_inner_cell):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(user_id, item_ids, num_threads=num_inner_cell)
        top_cols = np.argpartition(recs, -np.arange(N))[-N:][::-1]
        final_recs = [item_inv_mapping[item] for item in top_cols]
        return final_recs[:N]

    return _recs_mapper


def gen_cand_pos(df, mapper, model, mod='train'):
    logging.info('Generate candidates')
    candidates = pd.DataFrame({'user_id': df['user_id'].unique()})
    all_cols = list(mapper['items_mapping'].values())

    mapper_lfm_pos = cand_lfm_pos(
        model,
        item_ids=all_cols,
        N=100,
        user_mapping=mapper['users_mapping'],
        item_inv_mapping=mapper['items_inv_mapping'],
        num_inner_cell=32)

    candidates['item_id'] = candidates['user_id'].progress_map(mapper_lfm_pos)
    candidates = candidates.explode('item_id')
    candidates['rank'] = candidates.groupby('user_id').cumcount() + 1

    if mod == 'train':
        pos = candidates.merge(df, on=['user_id', 'item_id'], how='inner')
        pos['target'] = 1

        pos = pos.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

        return pos, candidates

    elif mod == 'inference':
        return candidates



def cand_lfm_neg(all_items):
    def _rec_neg(row):
        num = row['item_id_x']
        exeption = row['item_id_y']

        negative = all_items[~np.isin(all_items, exeption)]
        negative_res = np.random.choice(negative, num + 1)
        return negative_res

    return _rec_neg



def gen_cand_neg(df, candidates, pos):
    logging.info('Generate negative candidates')
    neg_add_1 = df.groupby('user_id')['item_id'].apply(list).reset_index()
    neg_add_2 = candidates.groupby('user_id')['item_id'].apply(list).reset_index()

    neg_add_1['item_id'] = neg_add_2['item_id'] + neg_add_1['item_id']

    neg = pos.copy()
    neg = neg.groupby('user_id')['item_id'].count()

    result = pd.merge(neg, neg_add_1, how='inner', on="user_id")
    neg = neg.reset_index()

    mapper_lfm_neg = cand_lfm_neg(all_items=df['item_id'].unique())

    neg['item_id'] = result.progress_apply(mapper_lfm_neg, axis=1)
    neg = neg[['user_id', 'item_id']].explode('item_id')
    neg['target'] = 0

    return neg
