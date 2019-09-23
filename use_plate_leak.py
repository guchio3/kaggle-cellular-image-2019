import numpy as np
import pandas as pd
import scipy
import scipy.special
import scipy.optimize
import json
from tqdm import tqdm

from tools.utils.args import parse_plate_args

# based on the URL below
# https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak
if __name__ == '__main__':
    args = parse_plate_args(None)
    train_df = pd.read_csv('./mnt/inputs/origin/train.csv.zip')
    test_df = pd.read_csv('./mnt/inputs/origin/test.csv')
    sub = pd.read_csv(args.original_sub)
    pred_df = pd.read_pickle(args.original_raw)

    # train part
    plate_groups = np.zeros((1108, 4), int)
    for sirna in range(1108):
        grp = train_df.loc[train_df.sirna == sirna,
                           :].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna, 0:3] = grp
        plate_groups[sirna, 3] = 10 - grp.sum()

    # test part
    all_test_exp = test_df.experiment.unique()

    group_plate_probs = np.zeros((len(all_test_exp), 4))
    for idx in range(len(all_test_exp)):
        preds = sub.loc[test_df.experiment ==
                        all_test_exp[idx], 'sirna'].values
        pp_mult = np.zeros((len(preds), 1108))
        pp_mult[range(len(preds)), preds] = 1

        sub_test = test_df.loc[test_df.experiment == all_test_exp[idx], :]
        assert len(pp_mult) == len(sub_test)

        for j in range(4):
            mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
                np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)

            group_plate_probs[idx, j] = np.array(
                pp_mult)[mask].sum() / len(pp_mult)
    exp_to_group = group_plate_probs.argmax(1)

    # apply
    predicted = np.stack(pred_df.raw_pred.values).squeeze()

    def select_plate_group(pp_mult, idx):
        sub_test = test_df.loc[test_df.experiment == all_test_exp[idx], :]
        assert len(pp_mult) == len(sub_test)
        mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
            np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
        pp_mult[mask] = 0
        return pp_mult

    def assign_plate(plate):
        probabilities = np.array(plate)
        cost = probabilities * -1
        rows, cols = scipy.optimize.linear_sum_assignment(cost)
        chosen_elements = set(zip(rows.tolist(), cols.tolist()))

        for sample in range(cost.shape[0]):
            for sirna in range(cost.shape[1]):
                if (sample, sirna) not in chosen_elements:
                    probabilities[sample, sirna] = 0

        return probabilities.argmax(axis=1).tolist()

    sub_preds_df = sub.copy()
    sub_preds_df['preds'] = None
    for idx in tqdm(range(len(all_test_exp))):
        #print('Experiment', idx)
        indices = (test_df.experiment == all_test_exp[idx])

        preds = predicted[indices, :].copy()

        preds = select_plate_group(preds, idx)
        sub.loc[indices, 'sirna'] = preds.argmax(1)
        idxes = sub_preds_df.loc[indices].index
        for _idx, pred in zip(idxes, preds):
            sub_preds_df.loc[_idx, 'preds'] = json.dumps(pred.tolist())
        #sub_preds_df.loc[indices, 'preds'] = preds

    sub_preds_df = sub_preds_df.set_index('id_code')
    current_plate = None
    plate_probabilities = []
    predicted = []
    for name in tqdm(sub['id_code']):
        experiment, plate, _ = name.split('_')
        if plate != current_plate:
            if current_plate is not None:
                predicted.extend([str(x) for x in assign_plate(plate_probabilities)])
            plate_probabilities = []
            current_plate = plate

        score_predict = np.array(json.loads(sub_preds_df.loc[name].preds))
        plate_probabilities.append(scipy.special.softmax(score_predict.squeeze()))
    predicted.extend([str(x) for x in assign_plate(plate_probabilities)])
    sub['sirna'] = predicted

    sub.to_csv(
        args.original_raw.replace(
            '_raw',
            '_plate_leak').replace(
            'pkl',
            'csv'),
        index=False)
