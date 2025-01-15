# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
import argparse
from sklearn.metrics import average_precision_score
import numpy as np
import pickle as pkl
from collections import OrderedDict
import zlib


def postprocessing1(data_name):

    def thumos_postprocessing(ground_truth, prediction, smooth=False, switch=False):
        if smooth:
            prob = np.copy(prediction)
            prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
            prob2 = np.append(prob[0, :].reshape(1, -1), prob[0: -1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob4 = np.append(prob[0: 2, :], prob[0: -2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
            probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
            prediction = np.copy(probsmooth)

        # Assign cliff diving (5) as diving (8)
        if switch:
            switch_index = np.where(prediction[:, 5] > prediction[:, 8])[0]
            prediction[switch_index, 8] = prediction[switch_index, 5]

        # Remove ambiguous (21)
        valid_index = np.where(ground_truth[:, 21] != 1)[0]

        return ground_truth[valid_index], prediction[valid_index]

    return {'THUMOS': thumos_postprocessing}.get(data_name, None)

def calibrated_average_precision_score(y_true, y_score):
    """Compute calibrated average precision (cAP), which is particularly
    proposed for the TVSeries dataset.
    """
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap

def perframe_average_precision(ground_truth,
                               prediction,
                               class_names,
                               ignore_index,
                               metrics,
                               postprocessing):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)
    ground_truth_list = ground_truth.tolist()
    prediction_list = prediction.tolist()
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                result['per_class_AP'][class_name] = compute_score(
                    ground_truth[:, idx], prediction[:, idx])
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))

    return result

def eval_perframe(pred_scores_file,**kwargs):
    with open(pred_scores_file, 'rb') as f:
        data_compressed = f.read()

    pred_scores = pkl.loads(zlib.decompress(data_compressed))

    cfg = pred_scores['cfg']
    class_names = kwargs.get('class_names', cfg.DATA.CLASS_NAMES)
    perframe_gt_targets = pred_scores['perframe_gt_targets']
    perframe_pred_scores = pred_scores['perframe_pred_scores']
    ignore_index = kwargs.get('ignore_index', cfg.DATA.IGNORE_INDEX)
    metrics = kwargs.get('metrics', cfg.DATA.METRICS)
    postprocessing = kwargs.get('postprocessing', postprocessing1(cfg.DATA.DATA_NAME))
    # Compute results
    result = perframe_average_precision(
        np.concatenate(list(perframe_gt_targets.values()), axis=0),
        np.concatenate(list(perframe_pred_scores.values()), axis=0),
        class_names,
        ignore_index,
        metrics,
        postprocessing
    )

    logging.info('Action detection perframe m{}: {:.5f}'.format(
        cfg.DATA.METRICS, result['mean_AP']
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_scores_file', type=str, required=True)
    args = parser.parse_args()

    eval_perframe(args.pred_scores_file)
