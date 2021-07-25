#coding=utf-8
import sys
sys.path.append("../")
# sys.path.append("D:\\Data\\CODE\\AICODE\\prism-4.7\\bin\\rnn2automata-master")
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from utils.help_func import save_pickle, load_pickle
from utils.constant import *

# 需要注意一下test是否产生了新的predicate
# 需要注意每棵树的深度是否都相同

'''
编码方式
获得训练集和测试集中的所有predicate(X3<5), 然后对path进行one hot编码
[0, 1, 1, 1, 0, 0,..., 0, 0] -- alphabet

将GBDT决策树的顺序逆序输入构造PFA

'''

min_tree_num = 10
filter_threshold = 1.2

def get_important_tree(clf, X, tree_num):
    s = clf.staged_decision_function(X)
    cnt = 0
    scon = np.array(0)
    for ss in s:
        cnt+=1
        ssT = ss.T
        if cnt == 1:
            last = ssT
            continue
        last = np.concatenate((last, ssT), axis = 0)
    # print(last.shape)
    all_stage_score = last.T
    GBDT_IMPORTANT = (1.0/float(tree_num)) * filter_threshold
    cnt_zero_null = 0
    important_tree_idx = []
    for i in range(len(all_stage_score)):
        important = False
        tmp = all_stage_score[i][::-1]
        all_tree_sum = np.sum(tmp)
        # print(tmp)
        for t in range(len(tmp)):
            ratio = float(tmp[t])/float(all_tree_sum)
            # print("tree:{}/{} = ratio:{}".format(tmp[t], all_tree_sum, ratio))
            if abs(ratio) >= GBDT_IMPORTANT:
                important = True
            else:
                if t > min_tree_num and important:
                    cnt_zero_null+=1
                    important_tree_idx.append(t)
                    break
                # print("no.{} tree with:{}".format(t, tmp[t]))
            if t == len(tmp)-1:
                important_tree_idx.append(t)
                cnt_zero_null+=1
        all_stage_score[i] = tmp
    # print(cnt_zero_null)
    # print("Get important decision tree index")
    return important_tree_idx

def get_path_encode():
    data = load_pickle(GBDT.ADULT_ONEHOT_DATA)
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["Y"], random_state=GBDT.ADULT_RANDOM_STATE)

    for tree_num in GBDT.ADULT_TREE_NUM:
        clf = GradientBoostingClassifier(n_estimators=tree_num, random_state=GBDT.ADULT_RANDOM_STATE)
        clf.fit(X_train, y_train)
        #预测测试样本的类别
        train_pre_y = clf.predict(X_train).tolist()
        test_pre_y = clf.predict(X_test).tolist()
        #在测试数据集上计算模型精确度
        test_score = clf.score(X_test, y_test)
        print("Tree {} score: ".format(tree_num), test_score)


###################### Collecting ######################
# ********************** Train **********************
        cnt_depth_error = 0
        print("\nStart loading train dataset...")
        predicate_list = []
        predicate_idx_list_train = []
        for sample_id in range(len(X_train)):
            predicate = ""
            for clf_index in range(tree_num):
                estimator = clf.estimators_[clf_index][0]
                feature = estimator.tree_.feature
                threshold = estimator.tree_.threshold
                node_indicator = estimator.decision_path([X_train[sample_id]])
                leave_id = estimator.apply([X_train[sample_id]])
                node_index = node_indicator.indices[node_indicator.indptr[0]:
                                                    node_indicator.indptr[1]]
                for node_id in node_index:
                    if leave_id[0] == node_id:
                        break
                    if (X_train[sample_id][feature[node_id]] <= threshold[node_id]):
                        predicate = "X{} <= {:.2f}; ".format(feature[node_id], threshold[node_id])
                    else:
                        predicate = "X{} > {:.2f}; ".format(feature[node_id], threshold[node_id])
                    if predicate not in predicate_list:
                        predicate_list.append(predicate)
                    predicate_idx_list_train.append(predicate_list.index(predicate))
                lenOfNode = len(node_index)
                while lenOfNode < 4:
                    cnt_depth_error += 1
                    print('tree depth error: {}'.format(len(node_index)))
                    lenOfNode += 1
                    predicate_idx_list_train.append(-1)
                    print('fill up')
        print("train dataset get {} depth error".format(cnt_depth_error))
    

# ********************** Test **********************
        cnt_depth_error = 0
        print("\nStart loading test dataset...")
        predicate_idx_list_test = []
        for sample_id in range(len(X_test)):
            predicate = ""
            for clf_index in range(tree_num):
                estimator = clf.estimators_[clf_index][0]
                feature = estimator.tree_.feature
                threshold = estimator.tree_.threshold
                node_indicator = estimator.decision_path([X_test[sample_id]])
                leave_id = estimator.apply([X_test[sample_id]])
                node_index = node_indicator.indices[node_indicator.indptr[0]:
                                                    node_indicator.indptr[1]]
                for node_id in node_index:
                    if leave_id[0] == node_id:
                        break
                    if (X_test[sample_id][feature[node_id]] <= threshold[node_id]):
                        predicate = "X{} <= {:.2f}; ".format(feature[node_id], threshold[node_id])
                    else:
                        predicate = "X{} > {:.2f}; ".format(feature[node_id], threshold[node_id])
                    if predicate not in predicate_list:
                        print("test dataset has new decision path")
                        predicate_list.append(predicate)
                    predicate_idx_list_test.append(predicate_list.index(predicate))
                # 是否需要补长
                lenOfNode = len(node_index)
                while lenOfNode < 4:
                    cnt_depth_error += 1
                    print('tree depth error: {}'.format(len(node_index)))
                    lenOfNode += 1
                    predicate_idx_list_test.append(-1)
                    print('fill up')
        print("test dataset get {} depth error".format(cnt_depth_error))
        print("length of predicate: {}".format(len(predicate_list)))



###################### Encoding ######################
# ********************** Train **********************
        one_hot_encode = []
        num_predicate = len(predicate_list)
        num_sample = len(predicate_idx_list_train) // tree_num // GBDT.ADULT_TREE_DEPTH
        print("Get all decision paths!\n{} predicates\nTotal {} items".format(num_predicate, num_sample))
        # each sample
        for dp in range(num_sample):
            # each tree
            for t in range(tree_num):
                zero = [0] * num_predicate
                # each predicate
                for d in range(GBDT.ADULT_TREE_DEPTH):
                    # += or =
                    pre_idx = predicate_idx_list_train[dp*tree_num*GBDT.ADULT_TREE_DEPTH + t*GBDT.ADULT_TREE_DEPTH + d]
                    if pre_idx == -1:
                        continue
                    zero[pre_idx] = 1
                one_hot_encode.append(zero)
        print("Load train dataset successfully!")
        print("number of train path:{}".format(len(one_hot_encode)))
        data = {}
        # 用于字母表和交叉特征的对应
        data["predicate"] = predicate_list
        data["train_decision_path"] = one_hot_encode
        data["train_pre_y"] = train_pre_y
        data["tree_num"] = tree_num
        data["important_tree_train"] = get_important_tree(clf, X_train, tree_num)
        print("Get important decision tree index.")
        save_pickle(get_path(GBDT.ADULT_TRAIN_DECISION_PATH.format(tree_num)), data)
        print("GBDT decision path in onehot for train dataset saved!")

# ********************** Test **********************
        one_hot_encode = []
        num_sample = len(predicate_idx_list_test) // tree_num // GBDT.ADULT_TREE_DEPTH
        print("Get all decision paths!\n{} predicates\nTotal {} items".format(num_predicate, num_sample))
        # each sample
        for dp in range(num_sample):
            # each tree
            for t in range(tree_num):
                zero = [0] * num_predicate
                # each predicate
                for d in range(GBDT.ADULT_TREE_DEPTH):
                    # += or =
                    pre_idx = predicate_idx_list_test[dp*tree_num*GBDT.ADULT_TREE_DEPTH + t*GBDT.ADULT_TREE_DEPTH + d]
                    if pre_idx == -1:
                        continue
                    zero[pre_idx] = 1
                one_hot_encode.append(zero)
        print("Load test dataset successfully!")
        print("number of test path:{}".format(len(one_hot_encode)))
        data = {}
        data["test_decision_path"] = one_hot_encode
        data["test_y"] = y_test
        data["test_pre_y"] = test_pre_y
        data["tree_num"] = tree_num
        data["test_score"] = test_score
        data["important_tree_test"] = get_important_tree(clf, X_test, tree_num)
        print("Get important decision tree index.")
        save_pickle(get_path(GBDT.ADULT_TEST_DECISION_PATH.format(tree_num)), data)
        print("GBDT decision path in onehot for test dataset saved!")

if __name__ == "__main__":
    get_path_encode()


