# -*- coding: utf-8 -*-

"""

@author: Alex Yang

@contact: alex.yang0326@gmail.com

@file: metrics.py

@time: 2020/6/13 22:19

@desc:

"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def precision_recall_fscore(pred_cate1_list, true_cate1_list,
                            pred_cate2_list, true_cate2_list,
                            pred_cate3_list, true_cate3_list):
    cate1_p, cate1_r, cate1_f1, _ = precision_recall_fscore_support(true_cate1_list,
                                                                    pred_cate1_list,
                                                                    average='weighted')
    print(f'Logging Info - Level 1 Category: ({cate1_p}, {cate1_r}, {cate1_f1})')

    cate2_p, cate2_r, cate2_f1, _ = precision_recall_fscore_support(true_cate2_list,
                                                                    pred_cate2_list,
                                                                    average='weighted')
    print(f'Logging Info - Level 2 Category: ({cate2_p}, {cate2_r}, {cate2_f1})')

    cate3_p, cate3_r, cate3_f1, _ = precision_recall_fscore_support(true_cate3_list,
                                                                    pred_cate3_list,
                                                                    average='weighted')
    print(f'Logging Info - Level 3 Category: ({cate3_p}, {cate3_r}, {cate3_f1})')

    val_p = np.mean([cate1_p, cate2_p, cate3_p])
    val_r = np.mean([cate1_r, cate2_r, cate3_r])
    val_f1 = np.mean([cate1_f1, cate2_f1, cate3_f1])
    print(f'Logging Info - ALL Level Category: ({val_p}, {val_r}, {val_f1})')

    eval_results = {
        'cate1_p': cate1_p, 'cate1_r': cate1_r, 'cate1_f1': cate1_f1,
        'cate2_p': cate2_p, 'cate2_r': cate2_r, 'cate2_f1': cate2_f1,
        'cate3_p': cate3_p, 'cate3_r': cate3_r, 'cate3_f1': cate3_f1,
        'val_p': val_p, 'val_r': val_r, 'val_f1': val_f1
    }
    return eval_results
