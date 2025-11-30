import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from tqdm import tqdm, trange
import os

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.max_seq_length = 2048
    def __call__ (self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, max_length=2048)
        return embeddings
        
class ChromaDB():
    def __init__(self, name, distance_function, model_name):
        #distance_function can be "cosine", "ip", "l2"
        self.name = name
        self.distance_function = distance_function
        self.model_name = model_name
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(
            name=name,
            #embedding_function=MyEmbeddingFunction(model_name=self.model_name),
            metadata={
                "hnsw:space": distance_function
            }
        )
        
    def parseRegs(self, folder, parser):
        
        docs = []
        acts = []
        ids = []
        titles = []
        terms = []

        xml_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))
        
        for file in tqdm(xml_files, desc="Processing Regulations"):
            with open(file, "r") as f:
                file_text = f.read()
             
            soup = BeautifulSoup(file_text, parser)
            
            body = soup.find_all("Text")
            title = soup.find("ShortTitle")
            num = soup.find("InstrumentNumber")
            if num:
                ids.append(num.get_text())
            else:
                ids.append("NO NUMBER")
            if not title:
                title = soup.find("LongTitle")
            defined_terms = soup.find_all("DefinedTermEn")
            terms_int = []
            for term in defined_terms:
                if term not in terms_int:
                    terms_int.append(term.get_text())
            terms.append(terms_int)
            
            placeholder = []
            for segment in body:
                placeholder.append(segment.get_text())
                
            full_text = "\n".join(placeholder)
            if title:
                full_text = title.get_text().upper() + "\n" + full_text
                titles.append(title.get_text())
                
            else:
                titles.append(" ")
            docs.append(full_text)
            
            act_tag = soup.find("XRefExternal")
            if act_tag:
                act = act_tag.get_text()
                acts.append({"act": act})
            else:
                acts.append({"act": "NO ACT"})
                
        metadatas = []
        for act, title, term in zip(acts, titles, terms):
            metadatas.append({
                "act": act["act"],
                "title": title,
                "category": "Regulation",
                "terms": term
            })
                
        for m in metadatas:
            m['terms'] = ", ".join(m["terms"])
        
        self.collection.add(
                ids=ids,
                documents=docs,
                metadatas=metadatas
        )        
        # batch-add the elements to the collection to avoid embedding OOM
        #for idx in trange(0,len(ids)):
        #    print(len(docs[idx]))
        #    self.collection.add(
        #            ids=[ids[idx]],
        #            documents=[docs[idx]],
        #            metadatas=[metadatas[idx]]
        #)
            
    def addRegulationLinks(self, folder, parser):
        for root, dirs, files in tqdm(os.walk(folder)):
            for file in files:
                if file.endswith('.xml'):
                    path = os.path.join(root, file)
                    with open(path, "r") as f:
                        file_text = f.read()
                    soup = BeautifulSoup(file_text, parser)
                    name = soup.find("InstrumentNumber").get_text()
                    try:
                        ref = soup.find("XRefExternal", {"reference-type": "regulation"})
                        url = ref.get("link")
                    except AttributeError:
                        url = name
                        url = url.replace("/", "-")
                    if url is None:
                        url = name
                        url = url.replace("/", "-")
                    self.collection.update(ids=[name],
                                            metadatas=[{
                                                "url": f"https://laws-lois.justice.gc.ca/eng/regulations/{url}"
                                            }]
                                            )
    def parseActs(self, folder, parser):
        docs = []
        ids = []
        titles = []
        terms = []

        xml_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))

        for path in tqdm(xml_files, desc="Processing XML files"):
            with open(path, "r") as f:
                file_text = f.read()

            soup = BeautifulSoup(file_text, parser)

            body = soup.find_all("Text")
            title = soup.find("ShortTitle")
            num = soup.find("ConsolidatedNumber")
            if num:
                ids.append(num.get_text())
            else:
                ids.append("NO NUMBER")
            if not title:
                title = soup.find("LongTitle")
            defined_terms = soup.find_all("DefinedTermEn")
            term_list = []
            for term in defined_terms:
                if term not in term_list:
                    term_list.append(term.get_text())
            terms.append(term_list)
                    
            placeholder = []
            for segment in body:
                placeholder.append(segment.get_text())

            full_text = "\n".join(placeholder)
            if title:
                full_text = title.get_text().upper() + "\n" + full_text
                titles.append(title.get_text())
            else:
                titles.append(" ")
            docs.append(full_text)

        act_metadatas = []
        for title, term in zip(titles, terms):
            act_metadatas.append({
                "title": title,
                "terms": term,
                "category": "Act",
                "act": "N/A"
            }) 
        for m in act_metadatas:
            m['terms'] = ", ".join(m['terms'])

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=act_metadatas
        )
        #for idx in trange(0,len(ids)):
        #    print(len(docs[idx]))
        #    self.collection.add(
        #            ids=[ids[idx]],
        #            documents=[docs[idx]],
        #            metadatas=[act_metadatas[idx]]
        #)

    def addActLinks(self, folder, parser):
        for root, dirs, files in tqdm(os.walk(folder)):
            for file in files:
                if file.endswith('.xml'):
                    path = os.path.join(root, file)
                    with open(path, "r") as f:
                        file_text = f.read()
                    soup = BeautifulSoup(file_text, parser)
                    name = soup.find("ConsolidatedNumber").get_text()
                    self.collection.update(ids=[name],
                                           metadatas=[{
                                               "url": f"https://laws-lois.justice.gc.ca/eng/acts/{name}"
                                           }])
