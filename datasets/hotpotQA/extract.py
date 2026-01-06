import json
from tqdm import tqdm

def extract_documents(data_path):
    """针对实际格式的文档提取函数"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_docs = {}
    
    for sample in tqdm(data, desc=f"Processing {data_path}"):
        # 获取context字段（列表的列表）
        context = sample.get("context", [])
        
        for doc in context:
            # 每个文档是包含两个元素的列表: [标题, 句子列表]
            if len(doc) >= 2:  # 确保有标题和句子列表
                title = doc[0]  # 第一个元素是标题
                sentences = doc[1]  # 第二个元素是句子列表
                
                # 合并文档中的所有句子
                content = " ".join(sentences)
                
                # 只保留首次出现的文档版本
                if title not in all_docs:
                    all_docs[title] = content
    
    return all_docs

def extract_qa_pairs(data_path):
    """针对实际格式的QA对提取函数"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = []
    
    for sample in tqdm(data, desc=f"Extracting QA from {data_path}"):
        # 获取问题、答案和支持事实
        question = sample["question"]
        answer = sample["answer"]
        
        # 提取支持事实的文档标题
        support_facts = sample.get("supporting_facts", [])
        support_titles = set()
        
        for fact in support_facts:
            if len(fact) >= 1:  # 至少包含标题
                support_titles.add(fact[0])  # 支持事实的第一个元素是标题
        
        # 获取支持文档内容
        support_contents = []
        context = sample.get("context", [])
        
        for doc in context:
            if len(doc) >= 1 and doc[0] in support_titles:  # 匹配标题
                if len(doc) >= 2:  # 确保有句子列表
                    content = " ".join(doc[1])
                    support_contents.append(content)
        
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "support_docs": list(support_titles),
            "support_content": " ||| ".join(support_contents)
        })
    
    return qa_pairs

path="datasets/hotpotQA/hotpot_dev_distractor_v1.json"

docs=extract_documents(path)

qa_pairs=extract_qa_pairs(path)


with open("hotpot-document.jsonl",'w') as f:
    for i in docs.keys():
        dic={}
        dic.update({"title":i})
        dic.update({"context":docs[i]})
        f.write(json.dumps(dic)+'\n')

with open("hotpot-qa.jsonl",'w') as f:
    for i in qa_pairs:
        f.write(json.dumps(i)+'\n')
