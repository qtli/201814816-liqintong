import numpy as np
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN # DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from nltk.stem.snowball import SnowballStemmer
import json


def txtTojson():
    tmp = []
    with open("/home/liqintong/code/ChatBot/DM/Tweets.txt", "r") as t:
        data = t.readlines()
        for d in data:
            tmp.append(d)
    with open("/home/liqintong/code/ChatBot/DM/Tweets.json", "w") as fp:
        json.dump(tmp, fp, indent=4)  # indent参数是设置jsons缩进的


def tokenize_and_stem(text):
    # 首先对句子分词，然后区分标点
    stemmer = SnowballStemmer('english')
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:  # 过滤非字母字符
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    for token in tokens:
        if re.search('[a-zA-Z]]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def main():
    # txtTojson()

    # nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')

    with open('/Users/mac/Desktop/DM/Tweets.json', 'r') as TweetsFile:
        content = json.load(TweetsFile)   #  [{'text': "...", 'cluster': "..."}, {...}, ..]
        tweets = {}  # {0: "...", 1: "...", ...}
        clusters = {}  # {0: 37, 1: 40, ...}    # len = 89

        for i in range(len(content)):
            tweets.update({i: content[i]['text']})
            clusters.update({i: content[i]['cluster']})
            # if content[i]['cluster'] in clusters.keys():
            #     clusters[content[i]['cluster']].append(i)
            # else:
            #     clusters[content[i]['cluster']] = []


    cluster_num = max(list(clusters.values()))   # 110
    tweet_num = len(list(tweets.values()))  #  2472

    # 创建字典
    vocab_stem = []
    vocab_tokenized = []


    for i in tweets:
        tokens = tokenize_and_stem(tweets[i])
        vocab_tokenized.append(tokens)

    # TF-IDF  将文本转为TF-IDF矩阵。 首先计算文档中的词频，转换为词频矩阵TF；IDF逆文档频率，在某些文档中出现高频但是在语料库中低频的具有较高权重
    tfidf_vectorizer = TfidfVectorizer(
                                       stop_words='english',
                                       use_idf=True,
                                       tokenizer=tokenize_and_stem,
                                       # ngram_range=(1, 3)
                                    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(list(tweets.values()))
    # print(tfidf_matrix.shape)    # have ngram_range=(1, 3): (2472, 4448)  not have (2472, 29160)

    # terms是TF-IDF矩阵使用的特征列表，使用TF-IDF矩阵可以运行一系列聚类算法
    terms = tfidf_vectorizer.get_feature_names()

    # dist = 1-cosine_similarity
    dist = 1 - cosine_similarity(tfidf_matrix)

    # -----------------------------------------KMeans------------------------------------------------------------
    # 使用预定数量的clusters初始化，每个文档分配给一个簇，最小化聚类内的平方和，计算聚类的平均值并将其用作新的聚类质心，重新分配，迭代直到收敛
    # 需要多次运行以全局最优，KMeans不易达到全局最优
    num_clusters = 89
    km = KMeans(n_clusters=num_clusters).fit(tfidf_matrix)

    km_pre = km.labels_.tolist()

    # print(km.labels_[100:110])  [75 25 18 85 86 19 88 86  3 37]

    # km_result = km.fit_predict(tfidf_matrix)
    # print(km_result)

    labels_true = []
    labels_pred = []
    for i in clusters:
        labels_true.append(clusters[i])
    labels_true = sorted(labels_true)

    for i in km_pre:
        labels_pred.append(km_pre[i])
    labels_pred = sorted(labels_pred)

    km_score = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print('KMeans NMI: ', km_score)   # 0.7629953372



    X = tfidf_matrix.toarray()
    ms = MeanShift()
    ms_pre = ms.fit_predict(X)
    ms_pre = sorted(ms_pre)
    ms_score = metrics.normalized_mutual_info_score(labels_true, ms_pre)
    print('MeanShift NMI: ', ms_score)   # 0.7056324482

    # -----------------------------------------------------Affinity Propagation----------------------------------------
    ap = AffinityPropagation().fit(tfidf_matrix)
    ap_pre = ap.fit_predict(tfidf_matrix)   #  [195 272 206 ..., 213 137 109]
    ap_pre = sorted(ap_pre)
    
    ap_score = metrics.normalized_mutual_info_score(labels_true, ap_pre)
    print('AffinityPropagation NMI: ', ap_score)   #  0.775145369374


    # --------------------------------------------------Spectral Clustering---------------------------------------------
    spc = SpectralClustering().fit(tfidf_matrix)
    # spc_pre = spc.fit_predict(tfidf_matrix)
    spc_pre = spc.labels_.tolist()
    spc_pre = sorted(spc_pre)
    
    spc_score = metrics.normalized_mutual_info_score(labels_true, spc_pre)
    print('SpectralClustering NMI: ', spc_score)   # 0.47384412442

    # -------------------------------------------------Ward Hierarchical clustering-------------------------------------
    ward_hc = AgglomerativeClustering(n_clusters=89, linkage='ward')
    X = tfidf_matrix.toarray()
    ward_hc.fit(X)
    ward_hc_pre = ward_hc.labels_.tolist()
    
    ward_hc_pre = sorted(ward_hc_pre)
    ward_hc_score = metrics.normalized_mutual_info_score(labels_true, ward_hc_pre)
    print('Ward Hierarchical clustering NMI: ', ward_hc_score)   #  0.759773200943



    # ------------------------------------------------- AgglomerativeClustering-----------------------------------------
    hc = AgglomerativeClustering(n_clusters=89)
    X = tfidf_matrix.toarray()
    hc.fit(X)
    hc_pre = hc.labels_.tolist()
    
    hc_pre = sorted(hc_pre)
    hc_score = metrics.normalized_mutual_info_score(labels_true, hc_pre)
    print('AgglomerativeClustering NMI: ', hc_score)  #  0.759773200943


    # ----------------------------------------DBSCAN-------------------------------b------------------------------
    X = tfidf_matrix.toarray()
    dbscan_pre = DBSCAN().fit_predict(X)
    dbscan_pre = sorted(dbscan_pre)
    dbscan_score = metrics.normalized_mutual_info_score(labels_true, dbscan_pre)
    print('DBSCAN NMI: ', dbscan_score)   #  0.155256389516



    # -------------------------------------------Gaussian mixture models------------------------------------------
    gm = GaussianMixture(n_components=89)
    X = tfidf_matrix.toarray()
    gm.fit(X)
    gm_pre = gm.predict(X)
    gm_pre = sorted(gm_pre)
    
    gm_score = metrics.normalized_mutual_info_score(labels_true, gm_pre)
    print('Gaussian mixture models NMI: ', gm_score)  # 0.816899648742



    # --------------------------------------------Birch------------------------------------------------------------
    birch = Birch(n_clusters=89)
    X = tfidf_matrix.toarray()
    # birch.fit(X)
    # birch_pre = birch.labels_.tolist()
    birch_pre = birch.fit_predict(X)
    birch_pre = sorted(birch_pre)
    
    birch_score = metrics.normalized_mutual_info_score(labels_true, birch_pre)
    print('Birch NMI: ', birch_score)  # 0.780857693264





if __name__ == '__main__':
    main()