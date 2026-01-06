from rag import RAG
from utils import *
import json

def build_qasper_database(rate):
    dataset="qasper-"+str(rate)
    rag=RAG(dataset)
    with open("qasper/documents.json",'r') as fp:
        all_docs=json.load(fp=fp)
    for i in list(all_docs.keys()):
        docs=all_docs[i]
        rag.add_documents(docs,tag=True,title=i)
        
        g=build_similarity_matrix(docs)
        key_docs=extract_keynodes_with_pagerank(g,top_k_rate=rate)
        fake=rag.generate_large_counterfactual_documents(key_docs,batch_size=5,max_workers=8)
        rag.add_documents(fake,False,title=i)

def build_hotpot_databate(rate):
    dataset="hotpot-qa-"+str(rate)
    rag=RAG(dataset=dataset)
    with open("hotpot-qa/cluster-result.json",'r') as fp:
        all_docs=json.load(fp)
    for i in list(all_docs.keys()):
        docs=list(all_docs[i].values())
        rag.add_documents(docs,True)
        g=build_similarity_matrix(docs)
        key_docs=extract_keynodes_with_pagerank(g,rate)
        fake=rag.generate_large_counterfactual_documents(key_docs,5,8)
        rag.add_documents(fake,False)
    
if __name__=="__main__":
    # build_qasper_database(rate=0.1)
    all_docs=[]
    with open("hotpot-qa/hotpot-document.jsonl",'r') as fp:
        docs=fp.readlines()
        for doc in docs:
            all_docs.append(json.loads(doc)["context"])
    rag=RAG(dataset="hotpot-qa-1.0")
    fake=rag.generate_large_counterfactual_documents(all_docs,10,64)
    rag.add_documents(fake,False)