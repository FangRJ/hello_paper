#coding=utf-8
import sys
# sys.path.append("../")
sys.path.append("D:\\Data\\CODE\\gbdt2pfa-master\\")
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from utils.help_func import save_pickle, load_pickle
from utils.constant import *

def get_path_encode():
    data = load_pickle(GBDT.ADULT_ONEHOT_DATA)
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["Y"], random_state=0)
    # for i in range(len(y_train)):
    #     if y_train[i] == 0:
    #         y_train[i] = -1
    # for i in range(len(y_test)):
    #     if y_test[i] == 0:
    #         y_test[i] = -1
    # for tree_num in GBDT.ADULT_TREE_NUM:
    tree_num = 100
    clf = GradientBoostingClassifier(n_estimators=tree_num, random_state=100)
    clf.fit(X_train, y_train)
    #预测测试样本的类别
    # train_pre_y = clf.predict(X_train).tolist()
    # test_pre_y = clf.predict(X_test).tolist()
    #在测试数据集上计算模型精确度
    # test_score = clf.score(X_test, y_test)
    # print("Tree {} score: ".format(tree_num), test_score)
    s = clf.staged_decision_function(X_train)
    cnt = 0
    scon = np.array(0)
    for ss in s:
        cnt+=1
        ssT = ss.T
        if cnt == 1:
            last = ssT
            continue
        last = np.concatenate((last, ssT), axis = 0)
    print(last.shape)
    all_stage_score = last.T
    GBDT_IMPORTANT = 1.0/float(tree_num)
    cnt_zero_null = 0
    for i in range(len(all_stage_score)):
        important = False
        tmp = all_stage_score[i][::-1]
        all_tree_sum = np.sum(tmp)
        # print(tmp)
        for t in range(len(tmp)):
            ratio = float(tmp[t])/float(all_tree_sum)
            # print("tree:{}/{} = ratio:{}".format(tmp[t], all_tree_sum, ratio))
            if ratio >= GBDT_IMPORTANT:
                important = True
            else:
                if t > 10 and important:
                    cnt_zero_null+=1
                    break
                # print("no.{} tree with:{}".format(t, tmp[t]))
        all_stage_score[i] = tmp
        # _range = np.max(all_stage_score[i]) - np.min(all_stage_score[i])
        # all_stage_score[i] = (all_stage_score[i] - np.min(all_stage_score[i])) / _range
    print(cnt_zero_null)
    print("OK")




if __name__ == "__main__":
    get_path_encode()
