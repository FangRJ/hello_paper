"""
predict with the L1 trace.
"""
import sys
import shutil
import numpy as np

# sys.path.append("D:\\Data\\CODE\\AICODE\\prism-4.7\\bin\\rnn2automata-master")
sys.path.append("../")

from utils.constant import *
from target_models.model_helper import *
from experiments.application.adv_detect.detect_utils import *
from experiments.effectiveness.get_reachability_matrix import prepare_prism_data, get_state_reachability
from utils.help_func import load_pickle

cnt_sample = 0
def dp2abstrace(x, partitioner, y_pre, important_tree_idx):
    global cnt_sample
    # 将决策path转换成abstract trace
    try:
        labels = list(partitioner.predict(x))
    except:
        print("predict error in No.{}".format(cnt_sample))
        print(x)
    else:
        cnt_sample += 1
    labels = labels[0:important_tree_idx+1]
    # 逆序决策树
    labels.reverse()
    TERM_SYMBOL = get_term_symbol(y_pre)
    abs_trace = [START_SYMBOL] + labels + [TERM_SYMBOL]
    return abs_trace


def test_acc_fdlt(**kwargs):
    test_decision_path = kwargs["test_decision_path"]
    test_pre_y = kwargs["test_pre_y"]
    # print("test data size: {}".format(len(test_pre_y)))
    # print("test data size: {}".format(len(test_decision_path)))
    test_y = kwargs["test_y"]
    dfa = kwargs["dfa"]
    tmp_prims_data = kwargs["tmp_prims_data"]
    tree_num = kwargs["tree_num"]
    partitioner = kwargs["partitioner"]
    important_tree_test = kwargs["important_tree_test"]

    trans_func, trans_wfunc = dict(dfa["trans_func"]), dict(dfa["trans_wfunc"])
    acc = 0
    fdlt = 0
    unspecified = 0
    pmc_cache = {}
    pfa_pred_p = 0
    gt_p = 0
    start_p = 0
    pfa_pred_list = []
    for i in range(len(test_y)):
        #当使用决策过程时
        # x = test_decision_path[start_p: start_p + 1]
        # start_p += 1
        #当使用决策路径时
        x = test_decision_path[start_p: start_p + tree_num]
        start_p += tree_num
        # 每次取tree_num个作为x序列
        L1_trace = dp2abstrace(x, partitioner, test_pre_y[i], important_tree_test[(start_p//tree_num)-1])
        _, L2_trace = get_path_prob(L1_trace, trans_func, trans_wfunc)

        last_inner = L2_trace[-2]
        if last_inner in pmc_cache:
            probs = pmc_cache[last_inner]
        else:
            probs = get_state_reachability(tmp_prims_data, num_prop=2, start_s=last_inner)
            pmc_cache[last_inner] = probs
        pfa_pred = np.argmax(probs)
        pfa_pred_list.append(pfa_pred)
        if pfa_pred == 1:
            pfa_pred_p += 1
        # 注意test_y的维度
        if test_y[i] == 1:
            gt_p += 1
        if pfa_pred == test_y[i]:
            acc += 1
        if pfa_pred == test_pre_y[i]:
            fdlt += 1
        if L2_trace[-1] == "T":
            unspecified += 1
    return acc / len(test_y), fdlt / len(test_y), unspecified, pfa_pred_p, gt_p


def pfa_predict():
    data_source = "train"
    alpha = 64
    for tree_num in GBDT.ADULT_TREE_NUM:
        log_string = []
        pstr = "PFA predict for GBDT with {} decision tree:\n".format(tree_num)
        log_string.append(pstr)
        print(pstr)
        pstr = "k\tgbdt_acc\tpfa_acc\t\tpfa_fdlt\tpre_p\tpositive/total\n"
        log_string.append(pstr)
        print(pstr)
        ave_gbdt_acc = 0.0
        ave_pfa_acc = 0.0
        ave_fdlt = 0.0
        data = load_pickle(get_path(GBDT.ADULT_TEST_DECISION_PATH.format(tree_num)))
        # print("length of test dataset: {}".format(len(data["test_y"])))
        for k in GBDT.ADULT_KMEANS_NUM:
            dfa_file_path = GBDT.ADULT_DP_PFA.format(tree_num, k)
            trans_func_file = os.path.join(dfa_file_path, "{}_transfunc.pkl").format(data_source)
            pm_file_path = os.path.join(dfa_file_path, "{}.pm").format(data_source)
            dfa = load_pickle(get_path(trans_func_file))
            # print(dfa_file_path)
            # print(pm_file_path)
            partitioner = load_pickle(GBDT.ADULT_PARTITIONER.format(tree_num, k))
            # make reachability matrix
            total_states, tmp_prims_data = prepare_prism_data(pm_file_path, num_prop=2)

            acc, fdlt, unspecified, pfa_pred_p, gt_p = test_acc_fdlt(test_decision_path = data["test_decision_path"], test_pre_y = data["test_pre_y"],
                                                    test_y = data["test_y"], dfa = dfa, tmp_prims_data = tmp_prims_data,
                                                    partitioner = partitioner, tree_num = data["tree_num"], important_tree_test = data["important_tree_test"])
            ave_gbdt_acc += data["test_score"]
            ave_pfa_acc += acc
            ave_fdlt += fdlt
            pstr = "{}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}\t{}/{}\n".format(k, data["test_score"], acc, fdlt,
                                                                                        pfa_pred_p, gt_p,
                                                                                        len(data["test_y"]))
            log_string.append(pstr)
            print(pstr)
            shutil.rmtree(tmp_prims_data)
        numOfExp = len(GBDT.ADULT_KMEANS_NUM)
        pstr = "Avg:\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(ave_gbdt_acc/numOfExp, ave_pfa_acc/numOfExp, ave_fdlt/numOfExp)
        log_string.append(pstr)
        print(pstr)

        log_path = get_path('adult_important/log_final/balanced_data_tree_{}.txt'.format(tree_num))
        with open(log_path, 'w') as f:
            for ls in log_string:
                f.write(ls)
        print("log file saved done!")

 
if __name__ == "__main__":
    pfa_predict()


    #   ┏┛ ┻━━━━━┛ ┻┓
    #   ┃　　　　　　 ┃
    #   ┃　　　━　　　┃
    #   ┃　┳┛　  ┗┳　┃
    #   ┃　　　　　　 ┃
    #   ┃　　　┻　　　┃
    #   ┃　　　　　　 ┃
    #   ┗━┓　　　┏━━━┛
    #     ┃　　　┃   神兽保佑
    #     ┃　　　┃   代码无BUG！
    #     ┃　　　┗━━━━━━━━━┓
    #     ┃　　　　　　　    ┣┓
    #     ┃　　　　         ┏┛
    #     ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
    #       ┃ ┫ ┫   ┃ ┫ ┫
    #       ┗━┻━┛   ┗━┻━┛

