import torch
torch.cuda.set_per_process_memory_fraction(0.9, 0)
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import gc
import time
import os
from queue import Queue
import json
import numpy as np
import chromadb
from chromadb.errors import InternalError
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
from sentence_transformers import SentenceTransformer
import prompts
import chroma
import tqdm
from QueryState import QueryState

#minimum service interface
class Service:   
    # useful for logging
    def getName(self):
        raise NotImplementedError
    
    # invoked by coordinator to add data to processing queue
    def submit(self, data):
        raise NotImplementedError
    
    # invoked within processing loop - pulls next element off queue, invokes required
    # processing, sends back to coordinator
    def processQueue(self, retMon):
        raise NotImplementedError
    
    # used to shut down process threads
    def isStillActive():
        raise NotImplementedError

# Andrew TODO:
# create functions to handle proposal text input, which will be the "data" in submit
# service will hold database, models, etc
# and will pass data elements through the pipeline
# as it does this, it will "phone home" to the QueryCoordinator to update the QID's status
# finally, it will update the QID's output data with QueryCoordinator.updateQIDData(qid, data)
# which appends to the qid's output list
class SearchEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        return embeddings
        
class SearchService(Service):
    def __init__(self, embed_model_name =  "Qwen/Qwen3-Embedding-0.6B",
                       reranker_model_name = "Qwen/Qwen3-Reranker-0.6B",
                       query_model_name = "Qwen/Qwen3-4B-Instruct-2507-FP8",
                       device_map = "cuda:0",
                       console_verbose=False):
        
        self.processing_queue = Queue()
        self.console_verbose = console_verbose
        # embedding initialization
        self.embed_model_name = embed_model_name
        self.db = chroma.ChromaDB("documents", "cosine", self.embed_model_name )
        num_docs = self.db.collection.count()
        if num_docs == 0:
            regs_folder = "laws-lois-xml/eng/regulations"
            acts_folder = "laws-lois-xml/eng/acts"
            self.db.parseRegs(regs_folder, "xml")
            self.db.addRegulationLinks(regs_folder, 'xml')
            self.db.parseActs(acts_folder, 'xml')
            self.db.addActLinks(acts_folder, 'xml')

        # general-purpose query model initialization
        self.llm_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            query_model_name,
            torch_dtype="auto",
            device_map=device_map
        )
        
        #re-ranker initialization
        self.rerank_model_name = reranker_model_name
        self.rr_tokenizer = AutoTokenizer.from_pretrained(self.rerank_model_name, padding_side='left')
        self.rr_model = AutoModelForCausalLM.from_pretrained(self.rerank_model_name, device_map=device_map).eval()
        self.rr_token_false_id = self.rr_tokenizer.convert_tokens_to_ids("no")
        self.rr_token_true_id = self.rr_tokenizer.convert_tokens_to_ids("yes")
        self.rr_max_length = 8192
    
    def submit(self, data):
        self.processing_queue.put(data)
        
    def getName(self):
        return "Search Service"
    
    # generate search query from document, search collection and return results
    def collectionQuery(self, document_text, n_results=50):
        messages = [
            {"role": "system", "content": prompts.QUERY_INSTRUCTIONS},
            {"role": "user", "content": prompts.QUERY_PROMPT.format(proposal_txt=document_text)}
        ]
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=16384
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = self.llm_tokenizer.decode(output_ids, skip_special_tokens=True)
        
        results = self.db.collection.query(
            query_texts = content,
            n_results = n_results
        )
        return results, content
    
    # generate search query from document, search collection and return results
    def collectionQueryMultiprompt(self, document_text, n_results=10):
        results = {}
        for query_prompt in prompts.QUERY_PROMPTS:
            messages = [
                {"role": "system", "content": prompts.QUERY_INSTRUCTIONS},
                {"role": "user", "content": query_prompt.format(proposal_txt=document_text)}
            ]
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)

            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=256
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            content = self.llm_tokenizer.decode(output_ids, skip_special_tokens=True)
            query_results = self.db.collection.query(
                query_texts = content,
                n_results = n_results
            )
            for k in query_results.keys():
                if type(query_results[k]) is type([]) and type(query_results[k][0]) is type([]):
                    results[k] = results.get(k, []) + query_results[k][0]
        messages = [
            {"role": "system", "content": prompts.QUERY_INSTRUCTIONS},
            {"role": "user", "content": prompts.QUERY_RERANKPROMPT.format(proposal_txt=document_text)}
        ]
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=256
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = self.llm_tokenizer.decode(output_ids, skip_special_tokens=True)
        # deduplicate results in case of multiple hits
        dedup_results = {}
        for r_idx in range(len(results['ids'])):
            if results['ids'][r_idx] not in dedup_results.get('ids',[]):
                for k in results.keys():
                    dedup_results[k] = dedup_results.get(k,[])+ [results[k][r_idx]]
        if self.console_verbose:
            print("de-duplicated search results size: ",len(dedup_results['ids']))
        return dedup_results, content
    
    def rerankFormatInstruction(self, query, regulation):
        instruction = prompts.RERANK_INSTRUCTION
        output = prompts.RERANK_PROMPT.format(instruction=instruction,query=query, doc=regulation)
        rr_inputs = self.rr_tokenizer(output, padding=False, max_length=self.rr_max_length,truncation='longest_first',return_attention_mask=False,return_tensors='pt')
        #rr_inputs = self.rr_tokenizer.pad(rr_inputs, padding=True, return_tensors="pt", max_length=8192)
        for key in rr_inputs:
            rr_inputs[key] = rr_inputs[key].to(self.rr_model.device)
        return rr_inputs
    
    def rerankComputeLogits(self, query, pbar = None):
        with torch.no_grad():
            batch_logits = self.rr_model(**query).logits[:, -1, :].to("cpu")
        pos_vector = batch_logits[:, self.rr_token_true_id]
        neg_vector = batch_logits[:, self.rr_token_false_id]
        batch_scores = F.log_softmax(
            torch.stack([pos_vector, neg_vector],dim=1), dim=1
        )
        if pbar is not None:
            pbar.update(1)
        return batch_scores[:,0].exp().tolist()
    
    # take list of regulations from search step 0,
    # and re-rank them with the search query
    # return sorted list of scores and regulations
    def rerankResults(self, search_query, reg_results):
        pbar = tqdm.tqdm(total = len(reg_results['documents']), desc="Re-Ranking Regulations")
        rr_scores = [
            self.rerankComputeLogits(
                self.rerankFormatInstruction(search_query, reg),pbar
            ) for reg in reg_results['documents']
        ]
        pbar.close()
        rr_scores = np.array([score[0] for score in rr_scores]) # flatten
        sorted_scores, sorted_regs = [],[]
        for idx in np.argsort(rr_scores)[::-1]:
            sorted_scores.append(rr_scores[idx])
            sorted_reg = {
                'ids':reg_results['ids'][idx],
                'document':reg_results['ids'][idx],
                'metadata':reg_results['metadatas'][idx]
            }
            sorted_regs.append(sorted_reg)
        if self.console_verbose:
            print("re-ranked dedup findings")
            for sc,scr in zip(sorted_scores, sorted_regs):
                print(scr['metadata']['act'],">",scr['metadata']['title'])
        return sorted_scores, sorted_regs
    
    def replace_strs(self,txt):
        txt = txt.replace("\"violation\": Uncertain","\"violation\": \"uncertain\"")
        txt = txt.replace("\"violation\": uncertain","\"violation\": \"uncertain\"")
        txt = txt.replace("\"violation\": true","\"violation\": \"true\"")
        txt = txt.replace("\"violation\": True","\"violation\": \"true\"")
        txt = txt.replace("\"violation\": false","\"violation\": \"false\"")
        txt = txt.replace("\"violation\": False","\"violation\": \"false\"")
        
        txt = txt.replace("\"applicable\": Uncertain","\"applicable\": \"uncertain\"")
        txt = txt.replace("\"applicable\": uncertain","\"applicable\": \"uncertain\"")
        txt = txt.replace("\"applicable\": true","\"applicable\": \"true\"")
        txt = txt.replace("\"applicable\": True","\"applicable\": \"true\"")
        txt = txt.replace("\"applicable\": false","\"applicable\": \"false\"")
        txt = txt.replace("\"applicable\": False","\"applicable\": \"false\"")
        return txt
    
    def finalPassResults(self, document_text, reg_results):
        outputs_final = []
        pbar = tqdm.tqdm(total=len(reg_results), desc="Generating Output")
        #for regulation_entry, metadata, ids in zip(reg_results['documents'], reg_results['metadatas'], reg_results['ids']):
        for regulation_entry in reg_results:
            metadata = regulation_entry['metadata']
            summary_prompt = [
                {"role": "system", "content": "You are a helpful government assistant."},
                {"role": "user", "content": prompts.SUMMARY_PROMPT.format(prop_txt=document_text, reg_title = metadata['title'],reg_txt = regulation_entry['document'])}
            ]
            text = self.llm_tokenizer.apply_chat_template(summary_prompt, tokenize=False, add_generation_prompt=True)
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)
            outputs = self.llm_model.generate(**model_inputs, max_new_tokens=2048)
            delta = outputs.size(1) - model_inputs['input_ids'].size(1)
            summ_txt = self.llm_tokenizer.batch_decode(outputs[:,-delta:], skip_special_tokens=True)[0]
            
            output_prompt = [
                {"role": "system", "content": "You are a helpful government assistant."},
                {"role": "user", "content": prompts.OUTPUT_PROMPT.format(reg_title = metadata['title'], prop_txt=document_text, reg_summ_txt = summ_txt, json_form=prompts.OUTPUT_JSON_FORMAT)}
            ]
            
            text = self.llm_tokenizer.apply_chat_template(output_prompt, tokenize=False, add_generation_prompt=True)
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)
            outputs = self.llm_model.generate(**model_inputs, max_new_tokens=2048)
            delta = outputs.size(1) - model_inputs['input_ids'].size(1)
            output_txt = self.llm_tokenizer.batch_decode(outputs[:,-delta:], skip_special_tokens=True)[0]
    
            
            ld_res = json.loads(self.replace_strs(output_txt))
            outputs_final.append({
                'title' : metadata['title'],
                'link' : metadata['url'],
                'id' : regulation_entry['ids'],
                'act' : metadata['act'],
                'applicable' : ld_res['applicable'].lower(),
                'violation' : ld_res['violation'].lower(),
                'notes': ld_res['notes'],
                'intermediate_summary':summ_txt
            })
            
            pbar.update(1)
        
        pbar.close()
        if self.console_verbose:
            print("Applicable:")
            for o in outputs_final:
                if o['applicable'] == "true":
                    print(o['title'])
                    print(o['link'])
                    print(o['applicable'],o['violation'])
                    #print(o['intermediate_summary'])
                    print(o['notes'])
                    print("~"*15)
            print("~"*15 + "\n" + "~"*15)
            print("Not Applicable:")
            for o in outputs_final:
                if o['applicable'] == "false":
                    print(o['title'])
                    print(o['link'])
                    print(o['applicable'],o['violation'])
                    #print(o['intermediate_summary'])
                    print(o['notes'])
                    print("~"*15)
        return outputs_final
        
    
    def processQueue(self, retMon):
        while not self.processing_queue.empty():
            data = self.processing_queue.get() # pop next off the queue
            torch.manual_seed(1337)
            retMon.updateQIDState(data['qid'], QueryState.PROCESSING_EMBEDDING) # alert QueryCoordinator
            regulations,prop_summ = self.collectionQueryMultiprompt(data['data'])
            
            retMon.updateQIDState(data['qid'], QueryState.PROCESSING_RERANKING) 
            scores_rr, regulations_rr = self.rerankResults(prop_summ,regulations)
            
            retMon.updateQIDState(data['qid'], QueryState.PROCESSING_OUTPUT)
            outputs = self.finalPassResults(data['data'], regulations_rr)
            
            retMon.updateQIDData(data['qid'], outputs)
            retMon.updateQIDState(data['qid'], QueryState.COMPLETE)
        
            
        
    def isStillActive(self):
        return True
    
