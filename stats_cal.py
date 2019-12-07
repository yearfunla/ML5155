"""
Course: CSI5155
Tiffany Nien Fang Cheng
Group 33
Student ID: 300146741
"""
from scipy.stats import friedmanchisquare, rankdata, ttest_rel, ttest_ind
import pprint
import numpy as np
import pandas as pd
# import Orange
import matplotlib.pyplot as plt

"""
The lib is to generate the caculation of statistical result
"""
algo_list = "classificationViaRegression	Hoeffiding	IBK	Randomforest	NaiveBayes	PART"

data12_3_acc = """
0.83726871	0.832988125	0.741024579	0.8023	0.836095001	0.835197459
0.848802755	0.711379835	0.938255786	0.9619	0.647873789	0.815167774
0.67674754	0.602816423	0.773540889	0.8008	0.587801154	0.673566339
0.781041793	0.764385221	0.814676491	0.8470	0.758126388	0.787401575
0.837061585	0.836578293	0.743648163	0.7925	0.837959127	0.838028169
"""
data1_23_acc = """
0.998066832	0.998204916	0.996893123	0.9982	0.993579122	0.997928749
0.995493292	0.887017887	0.997030464	0.9989	0.833531302	0.994340414
0.993947987	0.895109974	0.995435053	0.9976	0.828157421	0.992218841
0.788461538	0.807692308	0.807692308	0.9423	0.865384615	0.923076923
0.998135874	0.998204916	0.996824082	0.9982	0.995028998	0.998135874
"""
data1_2_acc = """
0.989985163	0.990356083	0.980712166	0.9900	0.982195846	0.989243323
0.988235294	0.718039216	0.990392157	0.9976	0.762745098	0.991568627
0.979775281	0.710861423	0.983146067	0.9908	0.748501873	0.985393258
0.989373814	0.990132827	0.983301708	0.9898	0.982922201	0.987096774
0.989985163	0.990356083	0.982937685	0.9896	0.986275964	0.989985163
"""
data1_23_tp = """
0.000000	0.000000	0.000000	0.000000	0.038462	0.000000
0.996598	0.819368	1.000000	0.998820	0.834849	0.994446
0.996265	0.847005	0.999308	0.997994	0.836699	0.990801
0.769231	0.884615	0.807692	1.000000	0.923077	0.923077
0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
"""
data12_3_tp = """
0.129822	0.140579	0.308605	0.224777	0.142433	0.151335
0.874457	0.559590	0.955307	0.961825	0.355680	0.890130
0.718782	0.515694	0.792416	0.796488	0.227011	0.751272
0.344214	0.167656	0.604228	0.551187	0.420252	0.373145
0.128338	0.139466	0.318249	0.238501	0.133160	0.133531
"""
data1_2_tp = """
0.000000	0.000000	0.000000	0.000000	0.115385	0.000000
0.993117	0.890631	0.997706	0.995411	0.829063	0.991205
0.979401	0.915356	0.994382	0.988390	0.825094	0.985393
0.000000	0.000000	0.000000	0.000000	0.115385	0.000000
0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
"""


def get_rank(data_list):
    """
    Get the rank along the row, each column is result of a model
    :param data_list: data of accuracy or TP from each algo
    :return: rank data and average rank per column
    """
    if isinstance(data_list, str):
        # data processing from string to list
        data = data_list.strip().splitlines()
        data_list = [x.split() for x in data]

    # get the rank of the row and append to np array
    rank_list = np.vstack([rankdata(line) for line in data_list])
    # calculate the mean of the column
    ave_rank = np.mean(rank_list, axis=0)
    return rank_list, ave_rank


def get_friedman(data_list):
    """
    Calculate the statistical result with friendman algo
    :param data_list: data of accuracy or TP from each algo
    :return: t-test value, and p value
    """
    if isinstance(data_list, str):
        # data processing
        data = data_list.strip().splitlines()
        data_list = [x.split() for x in data]

    # get t-test value and p value
    a = np.array(data_list)
    a_list = [a[:, i] for i in range(a.shape[1])]
    statistic, p_value = friedmanchisquare(*(a[:, i] for i in range(a.shape[1])))
    return statistic, p_value


def get_paired_t(data_list, names):
    """
    get the paired t result
    :param data_list:
    :return: mean, std, and t-score
    """
    print("paired t for {}".format(names))
    # print(data_list)
    # print(np.diff(data_list))
    # calculate the diff between each column
    diff_data_list = np.diff(data_list)
    # debug
    # print("rev0")
    # print(data_list[:,0])
    # print("rev1")
    # print(data_list[:,1])
    # print(diff_data_list[:,0])
    print("mean: {}".format(np.mean(diff_data_list[:, 0])))
    print("STD: {}".format(np.std(diff_data_list[:, 0])))
    print(ttest_rel(data_list[:, 0], data_list[:, 1]))
    return np.mean(diff_data_list[:, 0]), np.std(diff_data_list[:, 0]), ttest_rel(data_list[:, 0], data_list[:, 1])


def run_friedman(test_type='acc'):
    """
    To run the friedman test for different dataset
    :param test_type: accuracy or true positive
    :return:
    """
    # Do friedman test for different dataset
    # list of accuracy or list of TP
    # those are defined in the beginning of the test
    if 'acc' in test_type:
        data_list = [data12_3_acc, data1_23_acc, data1_2_acc]
    elif 'tp' in test_type:
        data_list = [data12_3_tp, data1_23_tp, data1_2_tp]
    # print(get_rank(data))
    for data in data_list:
        # data preprocessing
        if isinstance(data_list, str):
            data = data_list.lstrip().splitlines()
            data_list = [x.split() for x in data]
        # get the rank data first
        rank_list, ave_rank = get_rank(data)
        # debug
        # pprint.pprint(data)
        pprint.pprint(rank_list)
        pprint.pprint(ave_rank)

        # get the statistic result
        statistic, p_value = get_friedman(rank_list)
        print("%.4f" % statistic, "%.4f" % p_value)


def run_paired_t():
    """
    run the paired t for same data set to compare the algo
    :return:
    """
    # Do paired t test for multiclass 10folds ave accuracy result
    with open("paired_t.csv", mode="r", encoding="utf-8") as f:
        data = f.read()
    # print(data)
    # data_preprocess
    data = data.splitlines()
    names = data[0].strip().split(",")
    data_list = [np.array(line.strip().split(",")) for line in data[1:]]
    data_list = np.vstack(data_list[0:])
    data_list = data_list.astype(np.float)
    # print(data_list)
    # print(type(data_list))
    get_paired_t(data_list, names)


run_friedman('tp')
run_paired_t()
