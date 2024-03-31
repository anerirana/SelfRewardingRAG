import numpy as np
from evaluate import load
from datasets import load_metric

from llm import LLM
from sentence_transformers import SentenceTransformer, util
from document_retriver import DocumentRetrievalModel 
from constants import *
from preference_pair_generator import PreferencePairGenerator
from document_retriver import DocumentRetrievalModel 
from constants import *
import re
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu
# Utilities
def exists(val):
    return val is not None

def default(val, default_value):
    return val if exists(val) else default_value

class TrainingMode:
  def __init__(self):
    self.SimiliarityScoreCitation = "similiarity_score_citation"  
    self.ResponseWithCitation = "response_with_citation"
    self.IsolatedCitation = "isolated_citation"

class RAGPipeline:
    def __init__(self, config: dict):
        '''Executes each block of RAGPipeline to train query augmentation and RAG models

        Parameters:
        -----------
        num_documents
            Numnber of docuemts to retrieve per query by the Document Retrieval Model 
        '''
        self.p = default(config.get('NumberOfRetrievedDocuments'), 5)
        self.m = default(config.get('NumberOfQuerySets'), 5)
        self.n = default(config.get('NumberOfAugementedQueries'), 5)
        self.l = default(config.get('NumberOfResponses'), 5)
        self.k = default(config.get('NumberOfTopkDocuments'), 5)
        self.language_model = LLM(default(config.get('LanguageModelName'), 'mistralai/Mistral-7B-Instruct-v0.1'))
        self.citation_model = SentenceTransformer(default(config.get('CitationModelName'), 'sentence-transformers/all-mpnet-base-v2'))
        self.training_mode = default(config.get('TrainingMode'), TrainingMode().SimiliarityScoreCitation)

        self.document_retrieval_model = DocumentRetrievalModel()   
        self.pp_generator = PreferencePairGenerator(self.language_model)


    #TODO: Implement prediction to get RAG responses and their rewards after training

    def train(self, original_queries, epoch, real_ans,doc_ids=None):
        # for i in range(0,len(original_queries[0])):
        #     print("-----")
        #     print(original_queries[0][i])
        #     print(real_ans[i])
        #     self.prediction(original_queries[0][i],doc_ids,real_ans[i])
        '''Executes a training loop of the RAGPipeline

        Parameters:
        -----------
        original_query
            The original query to generate responses for
        '''
        dpo_dataset_dict = {}
        count = tqdm(total=self.m*len(original_queries), desc='RAG Iterations', position=0)
        for i, doc_id in enumerate(doc_ids):
            for original_query in original_queries[i]:
                print(original_query)
                qa_prompt = QUERY_AUGMENTATION_PROMPT.format(n=self.n-1, original_query=original_query)
                aug_queries = []
                all_documents = []
                top_documents = []
                all_responses = []
                all_rewards = np.zeros((self.m, self.l), dtype=float)
                contributing_documents = []
                first_pps = []

                for i in range(self.m):
                    queries = self.get_augmented_queries(qa_prompt, original_query)
                    print("Got queries")
                    top_k_docs, all_docs = self.document_retrieval_model.forward(queries, [doc_id], self.p, self.k)
                    print("top k")
                    
                    knowledge_base = []
                    if self.training_mode == TrainingMode().ResponseWithCitation:
                        ctr = 0
                        for doc in top_k_docs:
                            knowledge_base.append(f"Source {ctr+1}: {doc}")
                            ctr+=1

                    if self.training_mode == TrainingMode().ResponseWithCitation:
                        rag_prompt = RAG_CITATION_PROMPT.format(original_query = original_query, knowledge_base = "\n\n".join(knowledge_base))
                    else:
                        rag_prompt = RAG_PROMPT.format(original_query = original_query, knowledge_base = "\n\n".join(knowledge_base))
                    print("prompt ready")
                    responses, contri_docs = self.get_query_responses(rag_prompt, original_query, top_k_docs, i)
                    print("got responses")
                    rewards = [self.get_rewards(original_query, response) for response in responses]
                    with open("output/response_rewards.txt","a+") as f:
                        f.write("responses: ")
                        f.write(str(responses))
                        f.write("rewards: ")
                        f.write(str(rewards))
                    
                    print("above pp1")

                    pp1 = self.pp_generator.generateFirstPP(rag_prompt, responses, rewards)
                    


                    first_pps.append(pp1)
                    aug_queries.append(queries)
                    all_documents.append(all_docs)           
                    top_documents.append(top_k_docs[0])
                    all_responses.append(responses)
                    all_rewards[i] = rewards
                    contributing_documents.append(contri_docs)
                    count.update(1)
                print("outside")
                all_responses=np.array(all_responses)
                pp2 = self.pp_generator.generateSecondPP(qa_prompt, aug_queries, all_documents, top_documents, all_rewards, contributing_documents)

                # print(">>"*100)
                # print("aug_queries: ")            
                # print(str(self.find_list_dimensions(aug_queries)))        
                # print("all_documents: ")
                # print(str(self.find_list_dimensions(all_documents)))
                # print("top_documents: ")
                # print(str(self.find_list_dimensions(top_documents)))
                # print("all_responses: ")
                # print(str(all_responses.shape))
                # print("all_rewards: ")
                # print(str(all_rewards.shape))
                # print("contributing_documents: ")
                # print(str(self.find_list_dimensions(contributing_documents)))
                # print("first_pps: ")
                # print(str(self.find_list_dimensions(first_pps)))
                # print("second_pps: ")
                # print(len(pp2))
                # print(">>"*100)

                with open("output/all_variables_epoch_" + str(epoch) + ".txt","a+") as f:
                    f.write("original_query: ")
                    f.write(str(original_query))
                    f.write(">>"*100)
                    f.write("aug_queries: ")
                    f.write(str(aug_queries))
                    f.write(">>"*100)
                    f.write("all_documents: ")
                    f.write(str(all_documents))
                    f.write(">>"*100)
                    f.write("top_documents: ")
                    f.write(str(top_documents))
                    f.write(">>"*100)
                    f.write("all_responses: ")
                    f.write(str(all_responses))
                    f.write(">>"*100)
                    f.write("all_rewards: ")
                    f.write(str(all_rewards))
                    f.write(">>"*100)
                    f.write("contributing_documents: ")
                    f.write(str(contributing_documents))
                    f.write(">>"*100)
                    f.write("first_pps: ")
                    f.write(str(first_pps))
                    f.write(">>"*100)
                    f.write("second pp")
                    f.write(str(pp2))
                    f.write(">>"*100)
                
                dpo_dataset_dict.update(self.dpo_parsing(first_pps,pp2))       
        print("Number of training pairs = ", len(dpo_dataset_dict['prompt']))

        # torch.cuda.set_device(0)  # Assuming you want to use the first GPU

        # Train the model on that GPU
        self.language_model.train(epoch, dpo_dataset_dict)
    def compute_scores(self,references, candidates):
        """
        Compute multiple scores (BLEU, ROUGE, METEOR, etc.) given references and candidate translations.

        Args:
            references (list of list of str): A list of lists, each inner list contains reference translations for one sentence.
            candidates (list of str): A list of candidate translations.

        Returns:
            dict: Dictionary of scores including BLEU, ROUGE, METEOR, etc.
        """
        scores = {}

        # Load and compute BLEU score
        bleu_metric = load("bleu")
        scores['BLEU'] = bleu_metric.compute(predictions=candidates, references=references)

        # Load and compute ROUGE score
        rouge_metric = load_metric("rouge")
        scores['ROUGE'] = rouge_metric.compute(predictions=candidates, references=references)

        # Load and compute METEOR score
        meteor_metric = load_metric("meteor")
        scores['METEOR'] = meteor_metric.compute(predictions=candidates, references=references)

        # You can add more metrics here in a similar fashion

        return scores


    def prediction(self,query,doc_ids,real_ans):
        # print(doc_ids)
        queries=[]
        queries.append(query)        
        top_k_docs, all_docs = self.document_retrieval_model.forward(queries, doc_ids, self.p, self.k) 
        top_k_docs=top_k_docs[0]  
        rag_prompt = RAG_PROMPT.format(original_query = query, knowledge_base = "\n\n".join(top_k_docs))
        responses=self.language_model(rag_prompt, SAMPLING_PARAMS_DICT).split("[/INST]")[-1]
        # print("RESPONSES")
        # print(type(responses))
        print('==========='*50)
        print(query)
        print(real_ans)
        print(responses)
        q=[responses]
        z=[]
        l=[]
        for i in top_k_docs:
            l.append([i])
        z.append(l)
        print(self.compute_scores([[real_ans]],q))
        rewards = [self.get_rewards(original_query, response) for response in responses]
        print(rewards)
        print('==========='*50)



    def dpo_parsing(self,first_pps,pp2):
        dataset_dict={"prompt": [],"chosen": [], "rejected": []}
        for i in range(0,len(first_pps)):
            dataset_dict["prompt"].append(first_pps[i][0])
            dataset_dict["chosen"].append(first_pps[i][1])
            dataset_dict["rejected"].append(first_pps[i][2])
        
        for i in range(0,len(pp2)):
            dataset_dict["prompt"].append(pp2[i][0])
            dataset_dict["chosen"].append(pp2[i][1])
            dataset_dict["rejected"].append(pp2[i][2])
        
        return dataset_dict
    
    def find_list_dimensions(self,lst):
        if not isinstance(lst, list) or not lst:  # Base case: not a list or empty list
            return []
        return [len(lst)] + self.find_list_dimensions(lst[0])
          
    def get_query_responses(self, rag_prompt, original_query, top_k_docs, i):
        
        responses = []
        contri_docs = []
 
        
        for i in range(self.l):         
            if self.training_mode == TrainingMode().ResponseWithCitation:
                answer=self.language_model(rag_prompt, SAMPLING_PARAMS_DICT).split("[/INST]")[-1]
                try:
                    answer, sources = re.split("sources?\s?used", answer, flags=re.IGNORECASE)
                except ValueError as e:
                    print("Response does not have correct sources format")
                    #TODO: Fall back to sentence similarity using llm's answer
                source_list = re.findall("source.*\d+", sources, flags=re.I)
                contri_docs.append(source_list)
            elif self.training_mode == TrainingMode().SimiliarityScoreCitation:
                answer=self.language_model(rag_prompt, SAMPLING_PARAMS_DICT).split("[/INST]")[-1]

            responses.append(answer)
            
            # print(sources)
            # print(source_list)
        
        # contri_docs = top_k_docs*self.l
        if self.training_mode == TrainingMode().SimiliarityScoreCitation:
            contri_docs = self.get_cited_documents(responses, original_query, top_k_docs)

        return responses, contri_docs   

    def get_cited_documents(self, responses, original_query, top_k_docs):
        cited_documents = []
        docs_embedding = [self.citation_model.encode(doc, convert_to_tensor=True) for doc in top_k_docs] 
        top_k_docs = np.asarray(top_k_docs) 
        for response in responses:
            response_embedding = self.citation_model.encode(response, convert_to_tensor=True)
            scores = [util.pytorch_cos_sim(response_embedding, doc_embedding) for doc_embedding in docs_embedding]
            scores = np.asarray(scores[0].cpu())
            doc_indx = np.where(scores > 0.5)
            cited_documents.append(top_k_docs[doc_indx])
        
        # for i in range(self.l):
        #     prompt = EXTRACT_CITATION_PROMPT.format(original_query=original_query, extracts=top_k_docs, answer=responses[i])
        #     self.language_model(prompt)
        # return
        return cited_documents

    def get_rewards(self, original_query, response):
        max_tries = 5
        j = 0
        do_retry = True
        reward=-1
        reward_prompt = REWARD_PROMPT.format(original_query = original_query, answer = response)
        while j < max_tries and do_retry:
            
            j=j+1
            reward_response = self.language_model(reward_prompt)
            
            match = re.search("Final Score: ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("Total score = ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("Score: ([0-9]+) out of 5", reward_response)

            

            if match:
                reward = match.group(1)  
                do_retry = False
        
        if do_retry:
            with open("output/reward_error.txt","a") as f:
                f.write("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(str(max_tries), reward_prompt.encode('utf-8'), reward_response.encode('utf-8')))


                # raise Exception("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))
        
        return reward
    
    def get_augmented_queries(self, qa_prompt, original_query):
        '''
        Extracts query samples from the language model
        '''

        max_tries = 5
        response = ""
        j = 0
        sanity_check = False

        while j < max_tries and sanity_check == False:
            j += 1
            response = self.language_model(qa_prompt, SAMPLING_PARAMS_DICT)
            sanity_check = True
            for i in range(1, self.n+1):
                if f"{i}." not in response:
                    sanity_check = False
                    break
        
        response = response.split(qa_prompt)[-1] #Remove the prompt from the response       
        response = re.sub(r'<s>|</s>|<pad>|[\[|\"|\'|\/]+INST\]', '', response) #Remove special tokens
        queries = []
        for i in range(1, self.n):
            pattern = "Version " + str(i)
            match = re.search(pattern + r"\.(.*\?)",response)
            if not match:
                match = re.search(str(i)+r"\.(.*\?)",response)
            if not match:
                with open("output/query_aug_error.txt","a+") as f:
                    f.write(f"Query version {i} not found in response:")
                    f.write(str(response))
                break
            else:
                queries.append(match.group(1))
        
        # for i in range(1, self.n+1):
        #     queries = re.sub(f"Version {i}.", "", queries)
        #     queries = re.sub(f"{i}.", "", queries)
        
        # queries = re.sub(r'Here are three possible variations on your initial query:', '', queries)
        # queries = re.sub(r'Here are three possible variations on your request for information:', '', queries)
        
        # queries = queries.strip()
        # queries = re.split(r"\n{1,2}", queries) #Split the response into a list of queries
        
        queries = [q.strip() for q in queries]
        # queries = [q for q in queries if q != '']
        queries.append(original_query)
        if len(queries) < self.n:
            queries.extend(random.choices(queries, k=self.n-len(queries)))
        else:
            queries = queries[:self.n]
        return queries









