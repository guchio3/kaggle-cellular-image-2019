import gc
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .logs import dec_timer, sel_log


@dec_timer
def split_df(base_df, target_df, split_name,
             target_name, n_sections, logger=None):
    '''
    policy
    ------------
    * split df based on split_id, and set split_id as index
        because of efficiency.

    '''
    sel_log(
        f'now splitting a df to {n_sections} dfs using {split_name} ...',
        logger)
    split_ids = base_df[split_name].unique()
    splitted_ids = np.array_split(split_ids, n_sections)
    if split_name == target_name:
        target_ids = splitted_ids
    else:
        target_ids = [base_df.set_index(split_name)
                      .loc[splitted_id][target_name]
                      for splitted_id in splitted_ids]
    # Pay attention that this is col-wise splitting bacause of the
    #   data structure of this competition.
    target_df = target_df.set_index(target_name)
    dfs = [target_df.loc[target_id.astype(str)].reset_index()
           for target_id in target_ids]
    return dfs


def get_all_features(path):
    files = os.listdir(path)
    features = [_file.split('.')[0] for _file in files]
    return features


def _load_feature(feature, base_dir, logger=None):
    load_filename = base_dir + feature + '.pkl.gz'
    # sel_log(f'loading from {load_filename} ...', logger)
    feature = pd.read_pickle(load_filename, compression='gzip')
    # drop index because its very heavy to concat, and already sorted.
    feature.reset_index(drop=True, inplace=True)
    return feature


@dec_timer
def load_features(features, base_dir, nthread=os.cpu_count(), logger=None):
    loaded_features = []
    sel_log(f'now loading features ... ', None)
    with Pool(nthread) as p:
        iter_func = partial(_load_feature, base_dir=base_dir, logger=logger)
        loaded_features = p.map(iter_func, features)
        p.close()
        p.join()
        gc.collect()
    sel_log(f'now concatenating the loaded features ... ', None)
    features_df = pd.concat(loaded_features, axis=1)[features]
    return features_df


def _save_feature(feature_pair, base_dir, logger=None):
    feature, feature_df = feature_pair
    save_filename = base_dir + feature + '.pkl.gz'
    if os.path.exists(save_filename):
        sel_log(f'already exists at {save_filename} !', None)
    else:
        sel_log(f'saving to {save_filename} ...', logger)
        feature_df.reset_index(drop=True, inplace=True)
        feature_df.to_pickle(save_filename, compression='gzip')


@dec_timer
def save_features(features_df, base_dir, nthread=os.cpu_count(), logger=None):
    feature_pairs = [[feature, features_df[feature]] for feature in
                     features_df.columns]
    with Pool(nthread) as p:
        iter_func = partial(_save_feature, base_dir=base_dir, logger=logger)
        _ = p.map(iter_func, feature_pairs)
        p.close()
        p.join()
        del _


def select_features(df, importance_csv_path, metric='gain_mean', topk=10):
    ascending = True if metric in ['gain_cov', 'split_cov'] else False
    importance_df = pd.read_csv(importance_csv_path)
    importance_df.sort_values(metric, ascending=ascending, inplace=True)
    selected_df = df[importance_df.head(topk).features]
    return selected_df


@dec_timer
def init_raw_features(logger):
    sel_log('now making raw features...', logger)
    trn_goto_df = pd.read_csv('./mnt/inputs/origin/train_goto.tsv', sep='\t')\
        .set_index('pj_no').add_prefix('f000_').reset_index()
    trn_genba_df = pd.read_csv('./mnt/inputs/origin/train_genba.tsv', sep='\t')\
        .reset_index().set_index('pj_no').add_prefix('f000_genba_').reset_index()
    trn_origin_df = trn_goto_df.merge(trn_genba_df, on='pj_no', how='left')
    save_features(trn_origin_df, './mnt/inputs/features/train/')
    tst_goto_df = pd.read_csv('./mnt/inputs/origin/test_goto.tsv', sep='\t')\
        .set_index('pj_no').add_prefix('f000_').reset_index()
    tst_genba_df = pd.read_csv('./mnt/inputs/origin/test_genba.tsv', sep='\t')\
        .reset_index().set_index('pj_no').add_prefix('f000_genba_').reset_index()
    tst_origin_df = tst_goto_df.merge(tst_genba_df, on='pj_no', how='left')
    save_features(tst_origin_df, './mnt/inputs/features/test/')


def _remove_few_elements(feature, limit=9):
    cnt_dict = feature.value_counts().to_dict()
    feature.loc[feature.map(cnt_dict) < limit] = np.nan
    return feature


@dec_timer
def _mk_features(load_func, feature_func, feature_ids, logger=None):
    # Load dfs
    # Does not load if the feature_ids are not the targets.
    trn_df, tst_df = load_func(feature_ids, logger)
    if trn_df is None:
        return
    res_trn_df = feature_func(trn_df, feature_ids)
    res_tst_df = feature_func(tst_df, feature_ids)

    # dataset の違いを吸収
    trn_feature_dir = './mnt/inputs/features/train/'
    tst_feature_dir = './mnt/inputs/features/test/'
    if 'f000_id' in trn_df.columns:
        merge_col = 'f000_id'
#    elif 'f000_id' in trn_df.columns:
#        merge_col = 'pj_no'
    else:
        raise Exception('NO VALID IDs in the features')
    trn_ids = pd.DataFrame(
        pd.read_pickle(
            trn_feature_dir +
            f'{merge_col}.pkl.gz'))
    tst_ids = pd.DataFrame(
        pd.read_pickle(
            tst_feature_dir +
            f'{merge_col}.pkl.gz'))
    res_trn_df = trn_ids.merge(res_trn_df, on=merge_col, how='left')
    res_tst_df = tst_ids.merge(res_tst_df, on=merge_col, how='left')

    # Save the features
    sel_log(f'saving features ...', logger)
    save_features(res_trn_df, trn_feature_dir, os.cpu_count(), logger)
    save_features(res_tst_df, tst_feature_dir, os.cpu_count(), logger)
