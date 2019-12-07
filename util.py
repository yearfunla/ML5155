"""
Course: CSI5155
Tiffany Nien Fang Cheng
Group 33
Student ID: 300146741
"""

"""
This is helper lib
"""
def get_features(feature_list):
    """
    The lib is to generate the arff description format
    :param feature_list: feature listed in dtype
    :return: arff format of feature_list
    """
    # append the labeled class description
    feature_list.append(("class", "REAL"))

    for i in range(len(feature_list)):
        # arff lib does not support the following transform, therefore need to manually modify them in arff file
        # try:
        #     if "DATE" in feature_list[i][0]:
        #         feature_list[i] = (feature_list[i][0], 'DATE "yyyy-MM-dd"')
        #     elif "TIME" in feature_list[i][0]:
        #         feature_list[i] = (feature_list[i][0], 'DATE "HH:mm:ss"')
        #         # @ATTRIBUTE DATE DATE "yyyy-MM-dd"
        #         # @ATTRIBUTE2 TIME DATE "HH:mm:ss"
        # if "class" in feature_list[i][0]:
        #     feature_list[i] = (feature_list[i][0], '{1, 0}')
        if isinstance(feature_list[i][1], float) or "<f" in feature_list[i][1]:
            feature_list[i] = (feature_list[i][0], "NUMERIC")
        elif isinstance(feature_list[i][1], int) or "<i" in feature_list[i][1]:
            feature_list[i]= (feature_list[i][0], "NUMERIC")
        elif isinstance(feature_list[i][1], str):
            feature_list[i] = (feature_list[i][0], "REAL")
    return feature_list

def get_arff_dump(data, features, description='sample'):
    """
    generate the arff like dict
    :param data: numpy data
    :param features: feature descroptions
    :param description: string
    :return: dict of arff description and data
    """
    arff_dump = dict(data=data,
                     attributes=features,
                     relation='myRel',
                     description=description)
    return arff_dump

def get_CD():
    """
    get the critical difference and draw the Nemenyi diagram
    however the env has some issue to run the Orange3 package
    therefore copying the working code directly from python script of Orange applicataion
    :return:
    """
    # this part because of the compile issue so ran directly in the Orange
    import Orange
    import matplotlib.pyplot as plt

    names = """classificationViaRegression	Hoeffiding	IBK	Randomforest	NaiveBayes PART"""
    names = names.split()

    rank12_3 = [4.4, 2., 4., 4.1, 2.4, 4.1]

    rank_1_23 = [4., 3.7, 3.7, 4., 1.6, 4.]
    rank12 = [4., 4., 2.4, 4.4, 1.8, 4.4]

    ave_rank = dict(rank12_3=rank12_3, rank_1_23=rank_1_23, rank12=rank12)
    for it in ave_rank:
        cd = Orange.evaluation.compute_CD(ave_rank[it], 5)  # tested on 30 datasets
        Orange.evaluation.graph_ranks(ave_rank[it], names, cd=cd, width=6, textspace=1.5)
        plt.savefig(it)
        plt.close()