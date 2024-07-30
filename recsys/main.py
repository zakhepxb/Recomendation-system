import sys
import numpy as np
import pandas as pd

from data_prep import sort_outliers, split_dataset, prepare_dataset_and_mapper, matrix_csr
from train import train_lfm, train_ctb
from cand_gen import gen_cand_pos, gen_cand_neg
from ctb_prep import prep_dataset_for_ctb, merge_features, split_on_target, predict_ctb
from save_and_load import save_ctb, save_lfm, load_ctb
from metrics import count_metrics

import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

lfm_par = {'no_components': 34,
           'learning_schedule': 'adagrad',
           'loss': 'warp-kos',
           'learning_rate': 0.14815007053271476,
           'item_alpha': 5.209834088141735e-09,
           'user_alpha': 4.2367628720250326e-08,
           'max_sampled': 8}

ctb_par =  {'subsample': 0.9,
            'thread_count': 20,
            'random_state': 42,
            'verbose': 200,
            'early_stopping_rounds': 100,
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 2000}

def main():
    #Читаем данные
    df = pd.read_csv('wb_school_task_1.csv')

    #Сортировка выбросов
    df = sort_outliers(df)

    #Разбитие датасета на train test и lfm_train
    test, train, lfm_train = split_dataset(df)

    #Создаем подготавливаем данные для LightFM
    dataset, lightfm_mapping = prepare_dataset_and_mapper(df)

    #Cоздаем CSR матрицу
    matrix = matrix_csr(lfm_train, dataset)

    #Обучаем LightFM
    model_lfm = train_lfm(lfm_par, matrix)

    #Генерация кандидатов
    pos, candidates = gen_cand_pos(train, lightfm_mapping, model_lfm, mod='train')
    neg = gen_cand_neg(train, candidates, pos)

    #Подготовка данных для обучения CatBoost
    ctb_train, ctb_val, ctb_test = prep_dataset_for_ctb(train, pos, neg)

    #Очистка памяти
    del(pos,
        neg,
        model_lfm,
        matrix,
        lfm_train,
        candidates)

    #Добавление признаков в выборки
    train_feat, val_feat, test_feat = merge_features(train, ctb_train, ctb_val, ctb_test, mod='train')

    #Разбиваем на X и Y
    X_train, y_train, X_val, y_val = split_on_target(train_feat, val_feat, ctb_train, ctb_val)

    #Обучение CatBoost
    model_ctb = train_ctb(ctb_par, X_train, y_train, X_val, y_val)

    #Сохранение модели
    save_ctb(model_ctb)

    del (X_train,
         y_train,
         X_val,
         y_val,
         train_feat,
         val_feat,
         test_feat,
         model_ctb,
         ctb_train,
         ctb_val,
         ctb_test)

    #Для теста используем только пользователей из обучающей выборки
    test = test[test['user_id'].isin(train['user_id'].unique())]

    #Создание матрицы
    matrix_new = matrix_csr(train, dataset)

    #Новое обучение LightFM
    new_model_lfm = train_lfm(lfm_par, matrix_new)

    #Сохранение LightFM
    save_lfm(new_model_lfm)

    #Генерация кандидатов
    new_candidates = gen_cand_pos(test, lightfm_mapping, new_model_lfm, mod='inference')

    del (train,
         dataset,
         lightfm_mapping,
         matrix_new,
         new_model_lfm)

    #Добавление признаков
    score_feat, lfm_ctb_prediction = merge_features(df, ctb_train=new_candidates, mod='inference')

    del (df,
         new_candidates)

    #Загрузка CatBoost
    ctb_model = load_ctb('recsysctb_model')

    #Предсказания
    prediction = predict_ctb(score_feat, lfm_ctb_prediction, ctb_model)

    del (ctb_model)

    #Подсчет метрик
    count_metrics(test, prediction)

if __name__ == "__main__":
    main()