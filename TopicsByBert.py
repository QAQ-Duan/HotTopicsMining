import json
import jieba
import jieba.posseg as pseg
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
from scipy.spatial.distance import cosine

def DataBuilding():        
    corpus = []
    with open("./user_posts.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    for user in data:
        WeiboList = data[user]
        for i in WeiboList:
            corpus.append(i["text"])
    return corpus[:10]

def extract_keywords(text):
    # 导入中文停用词表
    with open('./stopwords.txt','r',encoding='gbk') as f:
        stopwords = f.read().split('\n')+["#","【","】","\u3000"," ","（","）"]
    # 使用jieba进行分词并标注词性
    words = pseg.cut(text)
    res = []
    # 定义需要保留的关键词词性
    allowed_pos = ['n', 'nr', 'ns', 'nt', 'nz']
    
    start = 0
    for word, pos in words:
        if pos in allowed_pos and word not in stopwords and len(word) > 1:
            res.append((word,(start, start + len(word))))
        start += len(word)
    
    return res

def extract_keywords_2gram(text):
    # 导入中文停用词表
    with open('./stopwords.txt','r',encoding='gbk') as f:
        stopwords = f.read().split('\n')+["#","【","】","\u3000"," ","（","）"]
    # 使用jieba进行分词并标注词性
    words = pseg.cut(text)
    res = []
    # 定义需要保留的关键词词性
    allowed_pos = ['n', 'nr', 'ns', 'nt', 'nz']
    kword = []
    
    start = 0
    for word, pos in words:
        if pos in allowed_pos and word not in stopwords and word not in kword and len(word)>1:
            res.append((word,(start, start + len(word))))
            kword.append(word)
        start += len(word)
    if not res:
        return None
    res2gram = []
    for i in range(len(res)-1):
        if res[i][1][1] == res[i+1][1][0]:
            res2gram.append((res[i][0]+res[i+1][0],(res[i][1][0],res[i+1][1][1])))
    return res2gram

def Topics1gram(TextList:list, Path:str):   
    corpus = TextList 
    corpus_tokens = []
    for doc in corpus:
        res = extract_keywords(doc)
        if res:
            corpus_tokens.append(res)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # 导入BERT模型和Tokenizer
    model = BertModel.from_pretrained('bert-base-chinese').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',max_length=512)

    topics = {}
    with torch.no_grad():
        for i in range(len(corpus_tokens)):
            if len(corpus_tokens[i]) == 0:
                continue
            encode_input = tokenizer(corpus[i],padding=True, truncation=True, return_tensors = "pt").to(device)
            output = model(**encode_input)
            sentence_embedding = output.last_hidden_state[:,0,:].cpu().detach().numpy()
            word_embeddings = output.last_hidden_state[:,1:-1,:].cpu().detach().numpy()
            scores = []
            for token_pos in corpus_tokens[i]:
                start = token_pos[1][0]
                end = token_pos[1][1]
                word_embedding = np.mean(word_embeddings[start+1:end+1])
                similarity = cosine(sentence_embedding, word_embedding)
                scores.append(similarity)

            keywords = [corpus_tokens[i][t][0] for t in np.argsort(scores)[-3:]] 
            
            for k in keywords:
                if k in topics:
                    topics[k] += 1
                else:
                    topics[k] = 1
            if i%30000 == 0:
                print(i)

    sorted_topics = dict(sorted(topics.items(), key=lambda item: item[1], reverse=True))
    with open(Path, 'w',encoding="utf-8") as json_file:
        json.dump(sorted_topics, json_file)

def Topics2gram(TextList:list, Path:str):
    corpus = TextList 
    corpus_tokens = []
    for doc in corpus:
        res = extract_keywords_2gram(doc)
        if res:
            corpus_tokens.append(res)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # 导入BERT模型和Tokenizer
    model = BertModel.from_pretrained('bert-base-chinese').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',max_length=512)
    topics = {}
    with torch.no_grad():
        for i in range(len(corpus_tokens)):
            if len(corpus_tokens[i]) == 0:
                continue
            encode_input = tokenizer(corpus[i],padding=True, truncation=True, return_tensors = "pt").to(device)
            output = model(**encode_input)
            sentence_embedding = output.last_hidden_state[:,0,:].cpu().detach().numpy()
            word_embeddings = output.last_hidden_state[:,1:-1,:].cpu().detach().numpy()
            scores = []
            for token_pos in corpus_tokens[i]:
                start = token_pos[1][0]
                end = token_pos[1][1]
                word_embedding = np.mean(word_embeddings[start+1:end+1])
                similarity = cosine(sentence_embedding, word_embedding)
                scores.append(similarity)

            keywords = [corpus_tokens[i][t][0] for t in np.argsort(scores)[-3:]] 
            
            for k in keywords:
                if k in topics:
                    topics[k] += 1
                else:
                    topics[k] = 1
            if i%30000 == 0:
                print(i)

    sorted_topics = dict(sorted(topics.items(), key=lambda item: item[1], reverse=True))
    with open(Path, 'w',encoding="utf-8") as json_file:
        json.dump(sorted_topics, json_file)

if __name__=='__main__':
    TextList = DataBuilding()
    Topics1gram(TextList,"output1gramByBert.json")
    Topics2gram(TextList,"output2gramByBert.json")