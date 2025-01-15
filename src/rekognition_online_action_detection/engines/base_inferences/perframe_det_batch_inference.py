# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import time

from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import zlib
from rekognition_online_action_detection.syops import get_model_complexity_info
from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result
# import rekognition_online_action_detection.models.energy_consumption_calculation
# from rekognition_online_action_detection.models.energy_consumption_calculation.flops_counter import get_model_complexity_info
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet
# from syops import get_model_complexity_info

try:
    from external.rulstm.RULSTM.utils import (get_marginal_indexes, marginalize, softmax,
                                                        topk_accuracy_multiple_timesteps,
                                                        topk_recall_multiple_timesteps,
                                                        tta)
except:
    raise ModuleNotFoundError


def do_perframe_det_batch_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()
    # print('aaaaaaaaaaaaaaaaaa')
    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='BatchInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS * 8,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=True,
    )

    # Collect scores and targets
    pred_scores = {}
    gt_targets = {}
    anticipation_scores = {}
    anticipation_targets = {}

    # input = torch.randn(1, 1, 28, 28, 16)
    # input_ = torch.randn(1, 1, 28, 28)
    #
    # # model = model.cpu()
    #
    # ops, params = get_model_complexity_info(model,  [(288,2048),(288,1024)],  data_loader, as_strings=True,
    #                                         print_per_layer_stat=True, verbose=True)
    # #
    # Nmac = ops[2]
    # Nac = ops[1]
    # print('{:<30}  {:<8}'.format('Computational complexity ACs:', ops[1]))
    # print('{:<30}  {:<8}'.format('Computational complexity MACs:', ops[2]))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # # Nmac = Nmac / 1e9  # G
    # Nac = Nac / 1e9  # G
    # E_mac = Nmac * 4.6  # mJ
    # E_ac = Nac * 0.9  # mJ
    # E_all = E_mac + E_ac
    # print(f"Number of operations: {Nmac} G MACs, {Nac} G ACs")
    # print(f"Energy consumption: {E_all} mJ")

    #
    # # print(args)
    # ts1 = time.time()
    #
    # # using real data
    # Nops, Nparams = get_model_complexity_info(model, [(288,2048),(288,1024)], None, as_strings=True,
    #                                           print_per_layer_stat=True, verbose=True)
    # # using random input
    # # Nops, Nparams = get_model_complexity_info(model, (3, 224, 224), dataloader=None, as_strings=True, print_per_layer_stat=True, verbose=True, syops_units='Mac', param_units=' ', output_precision=3)
    # print("Nops: ", Nops)
    # print("Nparams: ", Nparams)
    # t_cost = (time.time() - ts1) / 60
    # print(f"Time cost: {t_cost} min")
    #
    #
    # ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 384, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-384
    # ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 512, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-512
    # ssa_info = {'depth': 8, 'Nheads': 12, 'embSize': 768, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-768

    # just run one batch when debugging (Line 79 in engine.py)
    # if batch_idx >= 1: break
        # print(ops)

    with torch.no_grad():
        start = time.time()
        pbar = tqdm(data_loader, desc='BatchInference')

        # num_frames_toal = 0
        for batch_idx, data in enumerate(pbar, start=1):
            target = data[-4]
            # print(model)
            score = model(*[x.to(device) for x in data[:-4]])
            if cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES > 0:
                target, fut_target = target
                # print(len(score))
                # print(len(score[-1]))
                # score_w, fut_score_w = score[0],score[-1]
                # for i in range(len(score_w)):
                #     if i == 0:
                #         score = score_w[i].softmax(dim=-1).cpu().tolist()
                #     else:
                #         score.extend(score_w[i].softmax(dim=-1).cpu().tolist())
                # for i in range(len(fut_score_w)):
                #     if i == 0:
                #         fut_score = fut_score_w[i].softmax(dim=-1).cpu().tolist()
                #     else:
                #         fut_score.extend(fut_score_w[i].softmax(dim=-1).cpu().tolist())
                # score = np.array(score)
                # fut_score = np.array(fut_score)
                score, fut_score = score[0][-1].softmax(dim=-1).cpu().numpy(), score[-1][-1].softmax(dim=-1).cpu().numpy()
            else:
                score = score.softmax(dim=-1).cpu().numpy()
            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                    # num_frames_toal += num_frames
                    if session not in pred_scores:
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            anticipation_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                        pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                    if session not in gt_targets:
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            anticipation_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                        gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                    if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                                full_indices = torch.cat((query_indices,
                                                          torch.arange(query_indices[-1] + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE,
                                                                       query_indices[-1] + t_a + 1 + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE)),
                                                         dim=0)
                                anticipation_scores[session][full_indices, :, t_a] = score[bs][:full_indices.shape[0]]
                            anticipation_targets[session][query_indices] = target[bs][:query_indices.shape[0]]
                    else:
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                                if query_indices[-1] + t_a + 1 < num_frames:
                                    anticipation_scores[session][query_indices[-1] + t_a + 1, :, t_a] = score[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                            anticipation_targets[session][query_indices[-1]] = target[bs][-1]

                    if query_indices[0] == 0:
                        pred_scores[session][query_indices] = score[bs][:query_indices.shape[0]]
                        gt_targets[session][query_indices] = target[bs][:query_indices.shape[0]]
                    else:
                        pred_scores[session][query_indices[-1]] = score[bs][-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES - 1]
                        gt_targets[session][query_indices[-1]] = target[bs][-1]
                    # print(num_frames)
                    # for key, value in gt_targets.items():
                    #     print(value)
                        # print(len(value))
                        # print(value.shape)
                        # num_frames_toal += len(value)





    # # Save scores and targets
    pkl_data = {
        'cfg': cfg,
        'perframe_pred_scores': pred_scores,
        'perframe_gt_targets': gt_targets,
        # 'anticipation_targets':anticipation_targets,
        # 'anticipation_scores': anticipation_scores,
    }
    # cfg_data = {'cfg': cfg}
    # perframe_pred_scores_d = {'perframe_pred_scores': pred_scores}
    # perframe_gt_targets_d = {'perframe_gt_targets': gt_targets}
    # anticipation_targets_d = {'anticipation_targets': anticipation_targets}
    # anticipation_scores_d = {'anticipation_scores': anticipation_scores}
    #
    data_compressed = zlib.compress(pkl.dumps(pkl_data))
    # cfg_data_compressed = zlib.compress(pkl.dumps(cfg_data))
    # perframe_pred_scores_d_compressed = zlib.compress(pkl.dumps(perframe_pred_scores_d))
    # perframe_gt_targets_d_compressed = zlib.compress(pkl.dumps(perframe_gt_targets_d))
    # anticipation_targets_d_compressed = zlib.compress(pkl.dumps(anticipation_targets_d))
    # anticipation_scores_d_compressed = zlib.compress(pkl.dumps(anticipation_scores_d))
    #
    with open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '_result_oad.pkl', 'wb') as f:
        f.write(data_compressed)
    # with open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '_cfg_data.pkl', 'wb') as f:
    #     f.write(cfg_data_compressed)
    #
    # with open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + 'perframe_pred_scores_d_compressed.pkl', 'wb') as f:
    #     f.write(perframe_pred_scores_d_compressed)
    #
    # with open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '_perframe_gt_targets_d_compressed.pkl', 'wb') as f:
    #     f.write(perframe_gt_targets_d_compressed)
    #
    # with open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '_anticipation_targets_d_compressed.pkl', 'wb') as f:
    #     f.write(anticipation_targets_d_compressed)
    #
    # with open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '_anticipation_scores_d_compressed.pkl', 'wb') as f:
    #     f.write(anticipation_scores_d_compressed)
    # pkl.dump({
    #     'cfg': cfg,
    #     'perframe_pred_scores': pred_scores,
    #     'perframe_gt_targets': gt_targets,
    #     'anticipation_targets':anticipation_targets,
    #     'anticipation_scores': anticipation_scores,
    # }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.pkl', 'wb'))
    # print(osp.splitext(cfg.MODEL.CHECKPOINT)[0])
    # Compute results
    # print(type(anticipation_targets))
    # print(anticipation_targets)
    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
        maps_list = []
        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
            result = compute_result['perframe'](
                cfg,
                np.concatenate(list(anticipation_targets.values()), axis=0),
                np.concatenate(list(anticipation_scores.values()), axis=0)[:, :, t_a],
            )
            # logger.info('Action anticipation ({:.2f}s) perframe m{}: {:.5f}'.format(
            #     (t_a + 1) / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
            #     cfg.DATA.METRICS, result['mean_AP']
            # ))
            print('Action anticipation ({:.2f}s) perframe m{}: {:.5f}'.format(
                (t_a + 1) / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
                cfg.DATA.METRICS, result['mean_AP']
            ))
            maps_list.append(result['mean_AP'])
        # logger.info('Action anticipation (mean) perframe m{}: {:.5f}'.format(
        #     cfg.DATA.METRICS, np.mean(maps_list)
        # ))
        print('Action anticipation (mean) perframe m{}: {:.5f}'.format(
            cfg.DATA.METRICS, np.mean(maps_list)
        ))
    result = compute_result['perframe'](
        cfg,
        np.concatenate(list(gt_targets.values()), axis=0),
        np.concatenate(list(pred_scores.values()), axis=0),
    )

    end = time.time()
    num_frames_toal = 0
    for key, value in gt_targets.items():
        num_frames_toal += len(value)
    time_taken = end - start
    print(f'Processed {num_frames_toal} frames in {time_taken:.1f} seconds ({num_frames_toal / time_taken :.1f} FPS)')
    # logger.info('Action detection perframe m{}: {:.5f}'.format(
    #     cfg.DATA.METRICS, result['mean_AP']
    # ))
    print('Action detection perframe m{}: {:.5f}'.format(
        cfg.DATA.METRICS, result['mean_AP']
    ))


def do_ek100_perframe_det_batch_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='BatchInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS * 8,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect scores and targets
    pred_scores = {}
    gt_targets = {}
    gt_noun = {}
    gt_verb = {}
    pred_scores_verb = {}
    pred_scores_noun = {}
    anticipation_scores = {}
    anticipation_scores_verb = {}
    anticipation_scores_noun = {}
    cfg.DATA.NUM_VERBS = 98
    cfg.DATA.NUM_NOUNS = 301

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            if cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES > 0:
                total_target, _, = data[-4]
                target, noun_target, verb_target = total_target
                det_scores, _, noun_score, _, verb_score, _ = model(*[x.to(device) for x in data[:-4]])
                score = det_scores[-1].cpu().numpy()
                score_verb = verb_score.cpu().numpy()
                score_noun = noun_score.cpu().numpy()
            else:
                score = model(*[x.to(device) for x in data[:-4]])
                target, noun_target, verb_target = data[-4]
                score = score.cpu().numpy()
            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                    if session not in pred_scores:
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            anticipation_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                            anticipation_scores_verb[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                            anticipation_scores_noun[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                        pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                        pred_scores_verb[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS))
                        pred_scores_noun[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS))
                    if session not in gt_targets:
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                        gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                        gt_noun[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS))
                        gt_verb[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS))


                    if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                                full_indices = torch.cat((query_indices,
                                                          torch.arange(query_indices[-1] + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE,
                                                                       query_indices[-1] + t_a + 1 + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE)),
                                                         dim=0)
                                anticipation_scores[session][full_indices, :, t_a] = score[bs][:full_indices.shape[0]]
                                anticipation_scores_noun[session][full_indices, :, t_a] = score_noun[bs][:full_indices.shape[0]]
                                anticipation_scores_verb[session][full_indices, :, t_a] = score_verb[bs][:full_indices.shape[0]]
                    else:
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                                if query_indices[-1] + t_a + 1 < num_frames:
                                    anticipation_scores[session][query_indices[-1] + t_a + 1, :, t_a] = score[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                                    anticipation_scores_noun[session][query_indices[-1] + t_a + 1, :, t_a] = score_noun[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                                    anticipation_scores_verb[session][query_indices[-1] + t_a + 1, :, t_a] = score_verb[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]

                    if query_indices[0] == 0:
                        pred_scores[session][query_indices] = score[bs][:query_indices.shape[0]]
                        pred_scores_noun[session][query_indices] = score_noun[bs][:query_indices.shape[0]]
                        pred_scores_verb[session][query_indices] = score_verb[bs][:query_indices.shape[0]]
                        gt_targets[session][query_indices] = target[bs][:query_indices.shape[0]]
                        gt_noun[session][query_indices] = noun_target[bs][:query_indices.shape[0]]
                        gt_verb[session][query_indices] = verb_target[bs][:query_indices.shape[0]]
                    else:
                        pred_scores[session][query_indices[-1]] = score[bs][-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES - 1]
                        pred_scores_noun[session][query_indices[-1]] = score_noun[bs][-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES - 1]
                        pred_scores_verb[session][query_indices[-1]] = score_verb[bs][-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES - 1]
                        gt_targets[session][query_indices[-1]] = target[bs][-1]
                        gt_noun[session][query_indices[-1]] = noun_target[bs][-1]
                        gt_verb[session][query_indices[-1]] = verb_target[bs][-1]

    oad_score, oad_noun_score, oad_verb_score = np.concatenate(list(pred_scores.values()), axis=0), \
                                                np.concatenate(list(pred_scores_noun.values()), axis=0), \
                                                np.concatenate(list(pred_scores_verb.values()), axis=0)
    oad_target, oad_noun_target, oad_verb_target = np.concatenate(list(gt_targets.values()), axis=0), \
                                                   np.concatenate(list(gt_noun.values()), axis=0), \
                                                   np.concatenate(list(gt_verb.values()), axis=0)
    action_labels, noun_labels, verb_labels = np.argmax(oad_target, axis=-1), np.argmax(oad_noun_target, axis=-1), np.argmax(oad_verb_target, axis=-1)
    action_pred, oad_noun_pred, oad_verb_pred = oad_score.reshape(-1, 1, oad_score.shape[-1]), \
                                                oad_noun_score.reshape(-1, 1, oad_noun_score.shape[-1]), \
                                                oad_verb_score.reshape(-1, 1, oad_verb_score.shape[-1])
    valid_index = list(np.where(action_labels != 0))[0]
    overall_act_res = topk_recall_multiple_timesteps(action_pred[valid_index, ...], action_labels[valid_index], k=5)[0]
    overall_noun_res = topk_recall_multiple_timesteps(oad_noun_pred[valid_index, ...], noun_labels[valid_index], k=5)[0]
    overall_verb_res = topk_recall_multiple_timesteps(oad_verb_pred[valid_index, ...], verb_labels[valid_index], k=5)[0]

    # logger.info(f"Online Detection Results: Overall Action: {float(overall_act_res)*100:.6f}, Verb: {float(overall_verb_res)*100:.6f}, Noun: {float(overall_noun_res)*100:.6f}")
    print(f"Online Detection Results: Overall Action: {float(overall_act_res)*100:.6f}, Verb: {float(overall_verb_res)*100:.6f}, Noun: {float(overall_noun_res)*100:.6f}")
    path_to_data = 'external/rulstm/RULSTM/data/ek100/'
    segment_list = pd.read_csv(osp.join(path_to_data, 'validation.csv'),
                               names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                               skipinitialspace=True)
    predictions, labels, ids = [], [], []
    predictions_verb, predictions_noun = [], []
    discarded_labels, discarded_ids = [], []
    for segment in segment_list.iterrows():
        start_step = int(np.floor(segment[1]['start_f'] / 30 * cfg.DATA.FPS))
        vid = segment[1]['video'].strip()
        preds = (anticipation_scores[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
        preds_v = (anticipation_scores_verb[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
        preds_n = (anticipation_scores_noun[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
        if start_step >= cfg.MODEL.LSTR.ANTICIPATION_LENGTH:
            predictions.append(preds)
            if preds_v is not None:
                predictions_verb.append(preds_v)
            if preds_n is not None:
                predictions_noun.append(preds_n)
            labels.append(segment[1][['verb', 'noun', 'action']].values.astype(int))
            ids.append(segment[1]['id'])
        else:
            discarded_labels.append(segment[1][['verb', 'noun', 'action']].values.astype(int))
            discarded_ids.append(segment[1]['id'])

    actions = pd.read_csv(osp.join(path_to_data, 'actions.csv'), index_col='id')

    action_scores = np.concatenate(predictions)
    if len(predictions_verb) > 0:
        verb_scores = np.concatenate(predictions_verb)
    else:
        verb_scores = None
    if len(predictions_noun) > 0:
        noun_scores = np.concatenate(predictions_noun)
    else:
        noun_scores = None
    labels = np.array(labels)
    ids = np.array(ids)

    noun_scores = None
    verb_scores = None

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')


    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
    if verb_scores is None:
        verb_scores = marginalize(action_probs, vi).reshape(action_scores.shape[0],
                                                            action_scores.shape[1],
                                                            -1)
    else:
        verb_scores = softmax(verb_scores.reshape(-1, verb_scores.shape[-1]))
        verb_scores = verb_scores.reshape(action_scores.shape[0],
                                          action_scores.shape[1],
                                          -1)
    if noun_scores is None:
        noun_scores = marginalize(action_probs, ni).reshape(action_scores.shape[0],
                                                            action_scores.shape[1],
                                                            -1)
    else:
        noun_scores = softmax(noun_scores.reshape(-1, noun_scores.shape[-1]))
        noun_scores = noun_scores.reshape(action_scores.shape[0],
                                          action_scores.shape[1],
                                          -1)
    dlab = np.array(discarded_labels)
    dislab = np.array(discarded_ids)
    ids = np.concatenate([ids, dislab])
    num_disc = len(dlab)
    labels = np.concatenate([labels, dlab])
    verb_scores = np.concatenate((verb_scores, np.zeros((num_disc, *verb_scores.shape[1:]))))
    noun_scores = np.concatenate((noun_scores, np.zeros((num_disc, *noun_scores.shape[1:]))))
    action_scores = np.concatenate((action_scores, np.zeros((num_disc, *action_scores.shape[1:]))))

    verb_labels, noun_labels, action_labels = labels[:, 0], labels[:, 1], labels[:, 2]

    overall_verb_recalls = topk_recall_multiple_timesteps(verb_scores, verb_labels, k=5)
    overall_noun_recalls = topk_recall_multiple_timesteps(noun_scores, noun_labels, k=5)
    overall_action_recalls = topk_recall_multiple_timesteps(action_scores, action_labels, k=5)

    unseen = pd.read_csv(osp.join(path_to_data, 'validation_unseen_participants_ids.csv'), names=['id'], squeeze=True)
    tail_verbs = pd.read_csv(osp.join(path_to_data, 'validation_tail_verbs_ids.csv'), names=['id'], squeeze=True)
    tail_nouns = pd.read_csv(osp.join(path_to_data, 'validation_tail_nouns_ids.csv'), names=['id'], squeeze=True)
    tail_actions = pd.read_csv(osp.join(path_to_data, 'validation_tail_actions_ids.csv'), names=['id'], squeeze=True)

    unseen_bool_idx = pd.Series(ids).isin(unseen).values
    tail_verbs_bool_idx = pd.Series(ids).isin(tail_verbs).values
    tail_nouns_bool_idx = pd.Series(ids).isin(tail_nouns).values
    tail_actions_bool_idx = pd.Series(ids).isin(tail_actions).values

    tail_verb_recalls = topk_recall_multiple_timesteps(
        verb_scores[tail_verbs_bool_idx], verb_labels[tail_verbs_bool_idx], k=5)
    tail_noun_recalls = topk_recall_multiple_timesteps(
        noun_scores[tail_nouns_bool_idx], noun_labels[tail_nouns_bool_idx], k=5)
    tail_action_recalls = topk_recall_multiple_timesteps(
        action_scores[tail_actions_bool_idx], action_labels[tail_actions_bool_idx], k=5)

    unseen_verb_recalls = topk_recall_multiple_timesteps(
        verb_scores[unseen_bool_idx], verb_labels[unseen_bool_idx], k=5)
    unseen_noun_recalls = topk_recall_multiple_timesteps(
        noun_scores[unseen_bool_idx], noun_labels[unseen_bool_idx], k=5)
    unseen_action_recalls = topk_recall_multiple_timesteps(
        action_scores[unseen_bool_idx], action_labels[unseen_bool_idx], k=5)

    all_accuracies = np.concatenate(
        [overall_verb_recalls, overall_noun_recalls, overall_action_recalls,
         unseen_verb_recalls, unseen_noun_recalls, unseen_action_recalls,
         tail_verb_recalls, tail_noun_recalls, tail_action_recalls]
    )  # 9 x 8

    indices = [
        ('Overall Mean Top-5 Recall', 'Verb'), ('Overall Mean Top-5 Recall', 'Noun'),
        ('Overall Mean Top-5 Recall', 'Action'),
        ('Unseen Mean Top-5 Recall', 'Verb'), ('Unseen Mean Top-5 Recall', 'Noun'),
        ('Unseen Mean Top-5 Recall', 'Action'),
        ('Tail Mean Top-5 Recall', 'Verb'), ('Tail Mean Top-5 Recall', 'Noun'),
        ('Tail Mean Top-5 Recall', 'Action'),
    ]

    cc = np.linspace(1 / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_LENGTH,
                     1 / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
                     cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES, dtype=str)
    scores = pd.DataFrame(all_accuracies * 100, columns=cc,
                          index=pd.MultiIndex.from_tuples(indices))
    # logger.info(scores)
    print(scores)
    tta_verb = tta(verb_scores, verb_labels)
    tta_noun = tta(noun_scores, noun_labels)
    tta_action = tta(action_scores, action_labels)
    # logger.info(f'\nMean TtA(5): VERB {tta_verb:0.2f} '
    #             f'NOUN: {tta_noun:0.2f} ACTION: {tta_action:0.2f}')
    print(f'\nMean TtA(5): VERB {tta_verb:0.2f} '
                f'NOUN: {tta_noun:0.2f} ACTION: {tta_action:0.2f}')