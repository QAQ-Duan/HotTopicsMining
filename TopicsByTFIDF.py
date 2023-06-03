from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import json
import jieba

def DataBuilding():    
    corpus = []
    with open("./user_posts.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    for user in data:
        WeiboList = data[user]
        for i in WeiboList:
            corpus.append(i["text"])
    return corpus[:2000]

def TFIDFkeywords1gram(texts:list):
    # 导入中文停用词表
    with open('./stopwords.txt','r',encoding='gbk') as f:
        stopwords = f.read().split('\n')+["#","【","】","\u3000"," ","（","）"]
    corpus = []
    topics = {}
    for doc in texts:
        tmp = [word for word in jieba.cut(doc) if word not in stopwords and len(word)>1]
        corpus.append(' '.join(tmp))
    # 计算TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # 获取词汇表
    vocab = vectorizer.get_feature_names()

    # 遍历每个文本，提取关键词
    for i in range(len(corpus)):
        # 获取当前文本的TF-IDF值
        tfidf = X[i].toarray()[0]
        # 将TF-IDF值和词汇表合并为字典
        tfidf_dict = dict(zip(vocab, tfidf))
        # 按照TF-IDF值从高到低排序
        sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
        # 获取前k个关键词
        k = 3
        keywords = [word for word, score in sorted_tfidf[:k]]
        for k in keywords:
            if k in topics:
                topics[k] += 1
            else:
                topics[k] = 1
    return topics

def TFIDFkeywords2gram(texts:list):
    # 导入中文停用词表
    with open('./stopwords.txt','r',encoding='gbk') as f:
        stopwords = f.read().split('\n')+["#","【","】","\u3000"," ","（","）"]
    corpus = []
    topics = {}
    for doc in texts:
        tmp = [word for word in jieba.cut(doc) if word not in stopwords and len(word)>1]
        tmp1 = []
        for i in range(len(tmp)-1):
            tmp1.append(tmp[i]+tmp[i+1])
        corpus.append(' '.join(tmp1))
    # 计算TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # 获取词汇表
    vocab = vectorizer.get_feature_names()

    # 遍历每个文本，提取关键词
    for i in range(len(corpus)):
        # 获取当前文本的TF-IDF值
        tfidf = X[i].toarray()[0]
        # 将TF-IDF值和词汇表合并为字典
        tfidf_dict = dict(zip(vocab, tfidf))
        # 按照TF-IDF值从高到低排序
        sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
        # 获取前k个关键词
        k = 3
        keywords = [word for word, score in sorted_tfidf[:k]]
        for k in keywords:
            if k in topics:
                topics[k] += 1
            else:
                topics[k] = 1
    return topics

def merge_dicts(dict1, dict2):
    merged_dict = dict(dict1)  # 创建一个新字典，初始化为 dict1 的内容

    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value  # 如果键已存在于合并后的字典中，将值相加
        else:
            merged_dict[key] = value  # 如果键不存在于合并后的字典中，将其添加到字典中

    return merged_dict

if __name__=='__main__':
    corpus = DataBuilding()
    Path1 = "output1gramByTFIDF.json"
    Path2 = "output2gramByTFIDF.json"
    group_size = 500
    grouped_texts = [corpus[i:i+group_size] for i in range(0, len(corpus), group_size)]

    topics = {}
    for texts in tqdm(grouped_texts):
        tmp = TFIDFkeywords1gram(texts)
        topics = merge_dicts(tmp,topics)
    sorted_topics = dict(sorted(topics.items(), key=lambda item: item[1], reverse=True))
    with open(Path1, 'w',encoding="utf-8") as json_file:
        json.dump(sorted_topics, json_file)
    
    topics = {}
    for texts in tqdm(grouped_texts):
        tmp = TFIDFkeywords2gram(texts)
        topics = merge_dicts(tmp,topics)
    sorted_topics = dict(sorted(topics.items(), key=lambda item: item[1], reverse=True))
    with open(Path2, 'w',encoding="utf-8") as json_file:
        json.dump(sorted_topics, json_file)
    