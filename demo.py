import json
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
# import csv
if __name__ == '__main__':
    pred_scores_file_ours = os.getcwd()+'/checkpoints/configs/THUMOS/MAT/mat_long_256_work_8_kinetics_1x/best.pkl'
    pred_scores_file_mat = os.getcwd()+'/checkpoints/configs/THUMOS/MAT/mat_long_256_work_8_kinetics_1x/epoch-16.pkl'
    # pred_scores_file_lstr = os.getcwd() + '/checkpoints/configs/THUMOS/MAT/mat_long_256_work_8_kinetics_1x/epoch-15.pkl'
    pred_scores_ours = pkl.load(open(pred_scores_file_ours, 'rb'))
    pred_scores_mat = pkl.load(open(pred_scores_file_mat, 'rb'))
    # pred_scores_lstr = pkl.load(open(pred_scores_file_lstr, 'rb'))


    video_name = 'video_test_0001508'
    perframe_gt_targets_our = pred_scores_ours['perframe_gt_targets'][video_name]

    # perframe_gt_targets_mat = pred_scores_mat['perframe_gt_targets']['video_test_0000839']
    # print(pred_scores_mat['perframe_pred_scores'])
    # print(pred_scores_mat['perframe_pred_scores'])
    # print(pred_scores_ours)
    # print(type(pred_scores_ours['perframe_pred_scores']))
    perframe_pred_scores_our = pred_scores_ours['perframe_pred_scores'][video_name]
    perframe_pred_scores_mat = pred_scores_mat['perframe_pred_scores'][video_name]
    # perframe_pred_scores_lstr = pred_scores_lstr['perframe_pred_scores']['video_test_0000278']

    # perframe_pred_scores_lstr[:, 0] = 0
    perframe_pred_scores_our[:, 0] = 0
    perframe_pred_scores_mat[:, 0] = 0
    perframe_gt_targets_our[:, 0] = 0

    perframe_pred_scores_our = perframe_pred_scores_our.max(axis=1)
    # perframe_pred_scores_lstr = perframe_pred_scores_lstr.max(axis=1)
    perframe_pred_scores_mat = perframe_pred_scores_mat.max(axis=1)

    perframe_gt_targets_our = perframe_gt_targets_our.max(axis=1)

    start=0
    num=200
    perframe_pred_scores_our =perframe_pred_scores_our[start:num]
    # perframe_pred_scores_lstr = perframe_pred_scores_lstr[start:num]
    perframe_pred_scores_mat = perframe_pred_scores_mat[start:num]
    perframe_gt_targets_our = perframe_gt_targets_our[start:num]

    np.savetxt(video_name+'__data.txt', perframe_pred_scores_our)
    np.savetxt(video_name + 'no__data.txt', perframe_pred_scores_mat)
    np.savetxt(video_name+'_GT.txt', perframe_gt_targets_our)
    # perframe_gt_targets_our = perframe_gt_targets_our[start:num]
    x = np.linspace(0,len(perframe_pred_scores_our),num=len(perframe_pred_scores_our))

    # fig, ax = plt.subplots()
    fig,ax = plt.subplots(figsize=(12, 3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(x, perframe_pred_scores_our,color='red',label="Ours")
    # ax.plot(x, perframe_pred_scores_lstr,color='green', label="LSTR")
    ax.plot(x, perframe_pred_scores_mat,color='blue',label="w/o SGM")
    plt.plot(x, perframe_gt_targets_our, color='green',label="GT")
    ax.legend()
    plt.show()
    # plt.savefig('./fig.svg')
    plt.close()
