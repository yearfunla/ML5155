
from scipy.stats import friedmanchisquare, rankdata
import pprint
import numpy as np
import Orange
import matplotlib.pyplot as plt


data12_3_acc = """
0.83726871	0.832988125	0.741024579	0.835197459	0.836095001	0.835197459
0.848802755	0.711379835	0.938255786	0.815167774	0.647873789	0.815167774
0.67674754	0.602816423	0.773540889	0.673566339	0.587801154	0.673566339
0.781041793	0.764385221	0.814676491	0.787401575	0.758126388	0.787401575
0.837061585	0.836578293	0.743648163	0.838028169	0.837959127	0.838028169

"""
data1_23_acc = """
0.998066832	0.998204916	0.996893123	0.997928749	0.993579122	0.997928749
0.995493292	0.887017887	0.997030464	0.994340414	0.833531302	0.994340414
0.993947987	0.895109974	0.995435053	0.992218841	0.828157421	0.992218841
0.788461538	0.807692308	0.807692308	0.923076923	0.865384615	0.923076923
0.998135874	0.998204916	0.996824082	0.998135874	0.995028998	0.998135874

"""
data1_2_acc = """
0.9900	0.9904	0.9807	0.9892	0.9822	0.9892
0.9882	0.7180	0.9904	0.9916	0.7627	0.9916
0.9798	0.7109	0.9831	0.9854	0.7485	0.9854
0.9894	0.9901	0.9833	0.9871	0.9829	0.9871
0.9900	0.9904	0.9829	0.9900	0.9863	0.9900
"""

def get_rank(data_list):
    if isinstance(data_list, str):
        data = data_list.lstrip().splitlines()
        data_list = [x.split() for x in data]
    rank_list = np.array([rankdata(line) for line in data_list])
    # for line in data_list:
    #     rank_list.append(rankdata(line))
    ave_rank=np.mean(rank_list, axis=0)
    return rank_list, ave_rank

def get_friedman(data_list):
    if isinstance(data_list, str):
        data = data_list.lstrip().splitlines()
        data_list = [x.split() for x in data]

    a = np.array(data_list)
    a_list = [a[:, i] for i in range(a.shape[1])]
    statistic, p_value = friedmanchisquare(*(a[:, i] for i in range(a.shape[1])))
    return statistic, p_value

data_list = [data12_3_acc,data1_23_acc, data1_2_acc]
# print(get_rank(data))
for data in data_list:
    if isinstance(data_list, str):
        data = data_list.lstrip().splitlines()
        data_list = [x.split() for x in data]
    rank_list, ave_rank = get_rank(data)
    # pprint.pprint(data)
    pprint.pprint(rank_list)
    pprint.pprint(ave_rank)
    statistic, p_value = get_friedman(rank_list)
    print("%.4f"%statistic, "%.4f"%p_value)
    names = """classificationViaRegression	Hoeffiding	IBK	Randomforest	NaiveBayes PART"""
    names = names.split()
    cd = Orange.evaluation.compute_CD(ave_rank, 6) #tested on 30 datasets
    Orange.evaluation.graph_ranks(ave_rank, names, cd=cd, width=6, textspace=1.5)
    plt.show()
