import sys
import pandas as pd
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def feature_item(df):
    logging.info('Create features for items')
    count_pop_item = df[['user_id', 'item_id']].groupby(['item_id']).count().rename(
        columns={"user_id": "count_pop_item"})
    count_pop_item = count_pop_item.sort_values(['count_pop_item'], ascending=False)
    count_pop_item["item_rank"] = pd.factorize(count_pop_item["count_pop_item"].values)[0]
    count_pop_item = count_pop_item.sort_index()

    return count_pop_item

def feature_user(df):
    logging.info('Create features for users')
    count_pop_user = df[['user_id','item_id']].groupby(['user_id']).count().rename(columns={"item_id": "count_pop_user"})
    count_pop_user = count_pop_user.sort_values(['count_pop_user'], ascending=False)
    count_pop_user["user_rank"] = pd.factorize(count_pop_user["count_pop_user"].values)[0]
    count_pop_user = count_pop_user.sort_index()

    return count_pop_user


def feature_user_time(df):
    logging.info('Create time features for users')
    periods = ['day', 'hour', 'weekday']

    max_df_user = None

    for period in periods:
        sr_group = df[['user_id', 'item_id', period]].groupby(['user_id', period]).count()
        all_users = sr_group.index.get_level_values(0)
        sr_max_active_user = sr_group.groupby(all_users).idxmax().iloc[:, 0].apply(lambda x: x[1])
        max_df_user = pd.concat([max_df_user, pd.Series(sr_max_active_user).rename(f'user_{period}')], axis=1)

    return max_df_user


def feature_item_time(df):
    logging.info('Create time features for items')
    periods = ['day', 'hour', 'weekday']

    max_df_item = None

    for period in periods:
        sr_group = df[['user_id','item_id', period]].groupby(['item_id', period]).count()
        all_items = sr_group.index.get_level_values(0)
        sr_max_active_item = sr_group.groupby(all_items).idxmax().iloc[:, 0].apply(lambda x: x[1])
        max_df_item = pd.concat([max_df_item, pd.Series(sr_max_active_item).rename(f'item_{period}')], axis=1)

    return max_df_item