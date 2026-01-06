import json
import tqdm

path="datasets/qasper/qasper_papers_dev.jsonl"

docs=open(path,'r').readlines()

def integrate(l):
    s=""
    for i in l:
        s+=i
        s+="\n"
    return s

def extract_docs(doc:str):
    doc=json.loads(doc)
    title=doc["title"]
    paragraphs=doc["full_text"]["paragraphs"]
    abstract=doc["abstract"]
    content=[]
    for i in paragraphs:
        content.extend(i)
    return {
        title:content
    }

def extract_qas(doc):
    qa=[]
    doc=json.loads(doc)
    questions=doc["qas"]["question"]
    answers=doc["qas"]["answers"]
    for i in range(len(questions)):
        # print(answers[i]["answer"]["extractive_spans"])
        qa.append({
            "title":doc["title"],
            "question":questions[i],
            "answer":integrate(answers[i]["answer"][0]["extractive_spans"])})
    return qa

if __name__=="__main__":
    qas=[]
    documents={}
    for i in docs:
        qas.extend(extract_qas(i))
        documents.update(extract_docs(i))
    with open("qasper/qas.jsonl",'w') as fp:
        for i in qas:
            fp.write(json.dumps(i))
            fp.write("\n")
    with open("qasper/documents.json",'w') as fp:
        fp.write(json.dumps(documents,indent=4))