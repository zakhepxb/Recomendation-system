import sys
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from gen_feat import feature_item, feature_user, feature_user_time, feature_item_time
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def prep_dataset_for_ctb(df, pos, neg):
    logging.info('Prepare Dataset for Catboost')
    ctb_train_users, ctb_test_users = train_test_split(df['user_id'].unique(), random_state=42, test_size=0.2)
    ctb_train_users, ctb_val_users = train_test_split(ctb_train_users, random_state=42, test_size=0.1)

    select_col = ['user_id', 'item_id', 'target']

    ctb_train = shuffle(
        pd.concat([pos[pos['user_id'].isin(ctb_train_users)], neg[neg['user_id'].isin(ctb_train_users)]])[select_col])
    ctb_val = shuffle(
        pd.concat([pos[pos['user_id'].isin(ctb_val_users)], neg[neg['user_id'].isin(ctb_val_users)]])[select_col])
    ctb_test = shuffle(
        pd.concat([pos[pos['user_id'].isin(ctb_test_users)], neg[neg['user_id'].isin(ctb_test_users)]])[select_col])

    return ctb_train, ctb_val, ctb_test


def merge_features(df, ctb_train=None, ctb_val=None, ctb_test=None, mod='inference'):
    logging.info('Create features for CatBoost')
    df['day'] = df['order_ts'].dt.day
    df['hour'] = df['order_ts'].dt.hour
    df['weekday'] = df['order_ts'].dt.weekday

    count_pop_item = feature_item(df)
    count_pop_user = feature_user(df)
    max_df_user = feature_user_time(df)
    max_df_item = feature_item_time(df)

    user_col = ['user_day', 'user_hour', 'user_weekday']
    item_col = ['item_day', 'item_hour', 'item_weekday']
    count_col_user = ['count_pop_user', 'user_rank']
    count_col_item = ['count_pop_item', 'item_rank']

    if mod=='train':
        train_feat = ctb_train.merge(max_df_user[user_col], on=['user_id'], how='left').merge(max_df_item[item_col],
                                                                                          on=['item_id'], how='left')
        val_feat = ctb_val.merge(max_df_user[user_col], on=['user_id'], how='left').merge(max_df_item[item_col],
                                                                                      on=['item_id'], how='left')
        test_feat = ctb_test.merge(max_df_user[user_col], on=['user_id'], how='left').merge(max_df_item[item_col],
                                                                                        on=['item_id'], how='left')

        train_feat = train_feat.merge(count_pop_user[count_col_user], on=['user_id'], how='left').merge(
            count_pop_item[count_col_item], on=['item_id'], how='left')
        val_feat = val_feat.merge(count_pop_user[count_col_user], on=['user_id'], how='left').merge(
            count_pop_item[count_col_item], on=['item_id'], how='left')
        test_feat = test_feat.merge(count_pop_user[count_col_user], on=['user_id'], how='left').merge(
            count_pop_item[count_col_item], on=['item_id'], how='left')

        return train_feat, val_feat, test_feat

    elif mod=='inference':
        logging.info('Merge features')
        lfm_ctb_prediction = ctb_train.copy()
        score_feat = lfm_ctb_prediction.merge(max_df_user[user_col], on=['user_id'], how='left').merge(
            max_df_item[item_col], on=['item_id'], how='left')
        score_feat = score_feat.merge(count_pop_user[count_col_user], on=['user_id'], how='left').merge(
            count_pop_item[count_col_item], on=['item_id'], how='left')
        return score_feat, lfm_ctb_prediction


def split_on_target(train_feat, val_feat, ctb_train, ctb_val):
    logging.info('Split Dataset on X and y')
    drop_col = ['user_id', 'item_id']
    target_col = ['target']

    X_train, y_train = train_feat.drop(drop_col + target_col, axis=1), ctb_train[target_col]
    X_val, y_val = val_feat.drop(drop_col + target_col, axis=1), ctb_val[target_col]

    return X_train, y_train, X_val, y_val

def predict_ctb(score_feat, lfm_ctb_prediction, ctb_model):
    logging.info('Predict Recomendations')
    drop_col = ['user_id', 'item_id']

    ctb_prediction = ctb_model.predict_proba(score_feat.drop(drop_col, axis=1, errors='ignore'))
    lfm_ctb_prediction['ctb_pred'] = ctb_prediction[:, 1]
    lfm_ctb_prediction = lfm_ctb_prediction.sort_values(by=['user_id', 'ctb_pred'], ascending=[True, False])
    lfm_ctb_prediction['rank_ctb'] = lfm_ctb_prediction.groupby('user_id').cumcount() + 1
    lfm_ctb_prediction = lfm_ctb_prediction.drop(columns=['rank', 'ctb_pred'], axis=1)
    lfm_ctb_prediction = lfm_ctb_prediction.rename(columns={"rank_ctb": "rank"})

    return lfm_ctb_prediction