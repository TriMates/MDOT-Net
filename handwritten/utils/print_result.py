from .cluster import cluster
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def print_result(n_clusters, H, gt, count=10):
    # save H and gt
    # np.savez('data_my/data_H.npz', H=H_mean)
    # np.savez('data_my/data_gt.npz', gt=gt)
    # get best H and gt
    readd = np.load('H_gt/data_H.npz')
    readh = np.load('H_gt/data_gt.npz')
    H_fin = readd['H']
    gt_fin = readh['gt']
    acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std = cluster(n_clusters, H_fin, gt_fin, count=count)
    print('clustering h      : acc = {:.4f}, nmi = {:.4f}, ri = {:.4f}, f1 = {:.4f}'.format(acc_avg, nmi_avg, ri_avg, f1_avg))
