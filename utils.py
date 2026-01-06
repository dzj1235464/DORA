import networkx as nx
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from collections import defaultdict
import time
import json
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from typing import List
from sentence_transformers import SentenceTransformer
import math
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "mps"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2',device=get_device())
print("embedding model already loaded")

def calculate_text_similarity(text1:str,text2:str,embedding_model:SentenceTransformer=embedding_model)->float:
    model = embedding_model
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

def build_similarity_matrix(
    doc_list: List[str],
    embedding_model: SentenceTransformer = embedding_model,
    threshold: float = 0.5,
    embedding_batch_size: int = 256,
    sim_batch_size: int = 5000,
    show_progress: bool = True
) -> nx.Graph:
    """内存优化的相似度矩阵构建
    
    参数:
        embedding_batch_size: 嵌入生成批大小
        sim_batch_size: 相似度计算批大小
    """
    # 文档去重
    unique_docs = list(set(doc_list))
    n = len(unique_docs)
    g = nx.Graph()
    g.add_nodes_from(unique_docs)
    
    if n == 0:
        return g

    # 批量生成嵌入向量
    embeddings = []
    if show_progress:
        print(f"生成 {n} 个文档的嵌入向量...")
        pbar = tqdm(total=n, desc="文档嵌入", unit="doc")
    
    for i in range(0, n, embedding_batch_size):
        batch = unique_docs[i:i+embedding_batch_size]
        batch_emb = embedding_model.encode(
            batch,
            batch_size=min(embedding_batch_size, len(batch)),
            show_progress_bar=False,
            convert_to_tensor=False
        )
        embeddings.append(batch_emb)
        if show_progress:
            pbar.update(len(batch))
    
    if show_progress:
        pbar.close()
    
    embeddings = np.vstack(embeddings)
    
    # 分批计算相似度
    if show_progress:
        print(f"计算文档相似度 (阈值={threshold})...")
        total_pairs = n * (n - 1) // 2
        pbar = tqdm(total=total_pairs, desc="相似度计算", unit="pair")
    
    # 按批次处理相似度计算
    for i in range(0, n, sim_batch_size):
        i_end = min(i + sim_batch_size, n)
        
        # 计算当前批次与所有文档的相似度
        batch_embeddings = embeddings[i:i_end]
        sims = util.cos_sim(batch_embeddings, embeddings).numpy()
        
        # 处理当前批次的相似度
        for k in range(i_end - i):
            idx_i = i + k
            # 只处理上三角部分 (j > idx_i)
            for j in range(idx_i + 1, n):
                similarity = sims[k, j]
                if similarity >= threshold:
                    g.add_edge(
                        unique_docs[idx_i], 
                        unique_docs[j], 
                        weight=round(similarity, 4)
                    )
                
                if show_progress:
                    pbar.update(1)
    
    if show_progress:
        pbar.close()
        print(f"构建完成! 节点: {n}, 边: {g.number_of_edges()}")
    
    return g

def cluster_by_title(documents: dict, n_clusters=None, max_cluster_size=200):
    """
    对字典的键进行聚类，确保每个聚类不超过指定大小
    
    参数:
    documents -- 输入字典 {key: value}
    n_clusters -- 聚类数量（可选，自动确定如果为None）
    max_cluster_size -- 单个聚类的最大大小
    
    返回:
    clustered_dict -- 聚类结果字典 {cluster_id: {key: value}}
    """
    # 提取字典键
    keys = list(documents.keys())
    n_samples = len(keys)
    
    # 自动确定聚类数量
    if n_clusters is None:
        n_clusters = max(50, min(500, int(np.sqrt(n_samples) * np.log10(n_samples) * 1.5)))
    
    # 创建处理管道
    vectorizer = HashingVectorizer(
        analyzer='char', 
        ngram_range=(3, 5),
        n_features=2**18,
        alternate_sign=False
    )
    
    svd = TruncatedSVD(n_components=min(100, n_clusters * 2))
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    
    # 创建聚类器
    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(5000, n_samples // 10),
        max_iter=100,
        n_init=3,
        compute_labels=True
    )
    
    # 分批处理数据
    batch_size = 5000
    all_labels = np.empty(n_samples, dtype=np.int32)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_keys = keys[i:batch_end]
        
        # 向量化和降维
        X_batch = vectorizer.transform(batch_keys)
        X_batch = lsa.fit_transform(X_batch)
        
        # 部分拟合聚类器
        if i == 0:
            clusterer.partial_fit(X_batch)
        else:
            clusterer.partial_fit(X_batch)
        
        # 预测标签
        batch_labels = clusterer.predict(X_batch)
        all_labels[i:batch_end] = batch_labels
    
    # 构建初始聚类结果
    initial_clusters = defaultdict(dict)
    for key, label in zip(keys, all_labels):
        initial_clusters[int(label)][key] = documents[key]
    
    # 应用大小限制 - 分割过大的聚类
    final_clusters = {}
    next_cluster_id = 0
    
    for cluster_id, items in initial_clusters.items():
        cluster_size = len(items)
        
        if cluster_size <= max_cluster_size:
            # 符合大小要求
            final_clusters[next_cluster_id] = items
            next_cluster_id += 1
        else:
            # 计算子聚类数量
            sub_clusters_count = max(2, math.ceil(cluster_size / max_cluster_size))
            
            # 提取当前大聚类的键
            sub_keys = list(items.keys())
            
            # 子聚类向量化
            sub_vectorizer = HashingVectorizer(
                analyzer='char', 
                ngram_range=(3, 5),
                n_features=2**16
            )
            X_sub = sub_vectorizer.fit_transform(sub_keys)
            
            # 降维
            sub_svd = TruncatedSVD(n_components=min(50, sub_clusters_count * 2))
            sub_normalizer = Normalizer(copy=False)
            sub_lsa = make_pipeline(sub_svd, sub_normalizer)
            X_sub_reduced = sub_lsa.fit_transform(X_sub)
            
            # 子聚类
            sub_clusterer = MiniBatchKMeans(
                n_clusters=sub_clusters_count,
                batch_size=min(1000, cluster_size),
                max_iter=50,
                n_init=2
            )
            sub_labels = sub_clusterer.fit_predict(X_sub_reduced)
            
            # 添加子聚类到最终结果
            for sub_label in np.unique(sub_labels):
                sub_cluster_keys = [k for k, sl in zip(sub_keys, sub_labels) if sl == sub_label]
                final_clusters[next_cluster_id] = {k: documents[k] for k in sub_cluster_keys}
                next_cluster_id += 1
    
    # 最终检查 - 确保没有超过限制的聚类
    for cluster_id, items in list(final_clusters.items()):
        if len(items) > max_cluster_size:
            # 移除并分割超大聚类
            items_list = list(items.items())
            num_chunks = math.ceil(len(items) / max_cluster_size)
            del final_clusters[cluster_id]
            
            for i in range(num_chunks):
                chunk = dict(items_list[i*max_cluster_size : (i+1)*max_cluster_size])
                final_clusters[next_cluster_id] = chunk
                next_cluster_id += 1
    
    return final_clusters

def extract_keynodes_with_pagerank(G:nx.graph, top_k_rate=0.3, 
                                  damping=0.85, 
                                  max_iter=100, 
                                  tol=1.0e-6):
    """
    基于加权PageRank算法从文档图中提取关键节点
    
    参数:
    G: NetworkX图 (节点为文档ID, 边权重为embedding相似度)
    top_k: 返回的关键节点数量
    damping: PageRank阻尼系数 (通常0.85)
    max_iter: 最大迭代次数
    tol: 收敛阈值
    
    返回:
    top_nodes: 按重要性排序的前top_k个节点
    pr_scores: 所有节点的PageRank分数字典
    """
    
    graph_size=len(G.nodes)
    top_k=int(top_k_rate*graph_size)
    # 1. 预处理：确保权重存在并归一化
    if not nx.get_edge_attributes(G, 'weight'):
        # 如果没有权重属性，添加默认权重
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
    else:
        # 归一化权重到[0.1, 1]范围 (避免权重为0)
        weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
        if len(weights) > 0:
            norm_weights = minmax_scale(weights, feature_range=(0.1, 1))
            for i, (u, v) in enumerate(G.edges()):
                G[u][v]['weight'] = norm_weights[i]
    
    # 2. 执行标准加权PageRank
    pr_scores = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=tol, weight='weight')
    
    # 3. 对孤立节点应用惩罚因子
    for node in G.nodes():
        if G.degree(node) == 0:
            pr_scores[node] *= 0.3  # 孤立节点重要性降低
    
    # 4. 提取Top-k关键节点
    sorted_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, score in sorted_nodes[:top_k]]
    
    return top_nodes

def build_large_similarity_matrix(
    doc_list: List[str],
    embedding_model: SentenceTransformer,
    threshold: float = 0.6,
    embedding_batch_size: int = 256,
    sim_batch_size: int = 10000,
    show_progress: bool = True
) -> nx.Graph:
    """构建大型文档相似度图（内存优化版）
    
    处理超过10,000文档的优化版本，使用分批相似度计算
    
    参数:
        doc_list: 文档列表
        embedding_model: 文本嵌入模型
        threshold: 相似度阈值
        embedding_batch_size: 嵌入批处理大小
        sim_batch_size: 相似度计算批处理大小
        show_progress: 是否显示进度条
        
    返回:
        networkx图对象
    """
    # 文档去重和索引映射
    unique_docs = list(set(doc_list))
    n = len(unique_docs)
    
    g = nx.Graph()
    g.add_nodes_from(unique_docs)
    
    if n == 0:
        return g
    
    if show_progress:
        print(f"处理 {n} 个文档 (阈值={threshold})...")
    
    # 分批生成文档嵌入
    embeddings = []
    if show_progress:
        pbar = tqdm(total=n, desc="生成文档嵌入", unit="doc")
    
    for i in range(0, n, embedding_batch_size):
        batch = unique_docs[i:i+embedding_batch_size]
        batch_emb = embedding_model.encode(
            batch,
            convert_to_tensor=False,
            show_progress_bar=False,
            batch_size=min(embedding_batch_size, len(batch))
        )
        embeddings.append(batch_emb)
        if show_progress:
            pbar.update(len(batch))
    
    if show_progress:
        pbar.close()
    
    embeddings = np.vstack(embeddings)
    
    if show_progress:
        print(f"分批次计算相似度 (批大小={sim_batch_size})...")
        pbar = tqdm(total=(n*(n-1)//2), desc="计算相似度", unit="pair")
    
    # 分批计算相似度并添加边
    for i in range(0, n, sim_batch_size):
        i_end = min(i + sim_batch_size, n)
        
        # 计算当前批次与所有文档的相似度
        batch_embeddings = embeddings[i:i_end]
        sims = util.cos_sim(batch_embeddings, embeddings).numpy()
        
        # 处理当前批次的相似度
        for k in range(i_end - i):
            idx_i = i + k
            # 只处理上三角部分 (j > idx_i)
            for j in range(idx_i + 1, n):
                similarity = sims[k, j]
                if similarity >= threshold:
                    g.add_edge(unique_docs[idx_i], unique_docs[j], weight=round(similarity, 4))
                
                if show_progress:
                    pbar.update(1)
    
    if show_progress:
        pbar.close()
        print(f"构建完成！节点: {n}, 边: {g.number_of_edges()}")
    
    return g

if __name__ == "__main__":
    #process data

    # documents={}
    # with open("hotpot-document.jsonl",'r') as f:
    #     while(True):
    #         line=f.readline()
    #         if line=="":
    #             break
    #         json_line=json.loads(line)
    #         documents.update({json_line['title']:json_line['context']})
    #     f.close()
    
    # dic=cluster_by_title(documents,max_cluster_size=64)
    # with open("cluster-result.json",'w') as f:
    #     f.write(json.dumps(dic,indent=4))
    
    # build similarity matrix
    dic=json.load(open("cluster-result.json",'r'))
    for i in dic.keys():
        print("processing the document {}".format(i))
        lst=list(dic[i].values())
        matrix=build_large_similarity_matrix(lst,embedding_model)
        key_nodes=extract_keynodes_with_pagerank(matrix)
        for j in key_nodes:
            with open("key_docs.txt",'a') as file:
                file.write(j)
                file.write("\n")
            file.close()
        print("document {} already porcessed".format(i))