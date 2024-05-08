import numpy as np
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
import json
import time

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
        self.language_model = LLM(default(config.get('LanguageModelName'), 'mistralai/Mistral-7B-Instruct-v0.2'))
        self.citation_model = SentenceTransformer(default(config.get('CitationModelName'), 'sentence-transformers/all-mpnet-base-v2'))
        self.training_mode = default(config.get('TrainingMode'), TrainingMode().SimiliarityScoreCitation)

        self.document_retrieval_model = DocumentRetrievalModel(self.k, self.p)   
        self.pp_generator = PreferencePairGenerator(self.language_model)


    #TODO: Implement prediction to get RAG responses and their rewards after training

    def train(self, original_queries, epoch, doc_ids=None):
        '''Executes a training loop of the RAGPipeline

        Parameters:
        -----------
        original_query
            The original query to generate responses for
        '''
        dpo_dataset_dict = {}
        count = tqdm(total=self.m*len(original_queries), desc='RAG Iterations', position=0)
        f = open(OUTPUT_DIRECTORY + "all_variables_epoch_" + str(epoch) + ".txt","x")
        # for i, doc_id in enumerate(doc_ids):
        #     for original_query in original_queries[i]:
        for doc_id, original_query in zip(doc_ids, original_queries):
            
            qa_prompt = QUERY_AUGMENTATION_PROMPT.format(n=self.n-1, original_query=original_query)
            aug_queries = []
            all_documents = []
            top_documents = []
            all_responses = []
            all_rewards = np.zeros((self.m, self.l), dtype=float)
            contributing_documents = []
            first_pps = []

            for i in range(self.m):
                start_time = time.time()
                queries = self.get_augmented_queries(qa_prompt, original_query)
                print("Query aug time: ", str(time.time()-start_time))
                
                start_time = time.time()
                top_k_docs, all_docs = self.document_retrieval_model.train(queries, doc_id)
                print("Doc retrieval time: ", str(time.time()-start_time))
                
                start_time = time.time()
                knowledge_base = []
                ctr = 0
                for doc in top_k_docs:
                    knowledge_base.append(f"Source {ctr+1}: {doc}")
                    ctr+=1
                
                rag_prompt = RAG_CITATION_PROMPT.format(original_query = original_query, knowledge_base = "\n\n".join(knowledge_base))
                
                responses, contri_docs = self.get_query_responses(rag_prompt, original_query, top_k_docs, i)
                print("Response generation time: ", str(time.time()-start_time))

                start_time = time.time()      
                rewards = self.get_rewards(original_query, responses, rag_prompt)
                print("Reward generation time: ", str(time.time()-start_time))
                # try:
                #     f = open(OUTPUT_DIRECTORY + "response_rewards.txt","a")                    
                # except:
                #     f = open(OUTPUT_DIRECTORY +  "response_rewards.txt","w")
                # f.write("responses: ")
                # f.write(str(responses))
                # f.write("rewards: ")
                # f.write(str(rewards))
                # f.close()
                pp1 = self.pp_generator.generateFirstPP(rag_prompt, responses, rewards)

                first_pps.append(pp1)
                aug_queries.append(queries)
                all_documents.append(all_docs)           
                top_documents.append(top_k_docs)
                all_responses.append(responses)
                all_rewards[i] = rewards
                contributing_documents.append(contri_docs)
                count.update(1)
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

            with open(OUTPUT_DIRECTORY + "all_variables_epoch_" + str(epoch) + ".txt","a") as f:
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
        
        with open(OUTPUT_DIRECTORY + "dpo_preference_pairs_" + str(epoch) + ".json", "w") as f: 
            json.dump(dpo_dataset_dict, f)
        torch.cuda.set_device(0)  # Assuming you want to use the first GPU

        # # Train the model on that GPU
        self.language_model.train(epoch)



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
        bleu_metric = load_metric("bleu")
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
        q=[responses]
        z=[]
        l=[]
        for i in top_k_docs:
            l.append([i])
        z.append(l)
        rewards = [self.get_rewards(original_query, response) for response in responses]

        [doc_ids_flat.extend([doc_id]*len(original_queries[i])) for i,doc_id in enumerate(doc_ids)]
        
        count = tqdm(total=len(original_queries), desc='RAG Iterations', position=0)
        for original_query, doc_id, gold_answer in zip(original_queries, doc_ids_flat, gold_answers): 
            qa_prompt = QUERY_AUGMENTATION_PROMPT.format(n=self.n-1, original_query=original_query)
            aug_queries = self.get_augmented_queries(qa_prompt, original_query)
            top_k_docs, _ = self.document_retrieval_model.train(aug_queries, doc_id)
            knowledge_base = []
            ctr = 0
            for doc in top_k_docs:
                knowledge_base.append(f"Source {ctr+1}: {doc}")
                ctr+=1
            
            rag_prompt = RAG_CITATION_PROMPT.format(original_query = original_query, knowledge_base = "\n\n".join(knowledge_base))
            answer=self.language_model(rag_prompt).split("[/INST]")[-1]
            try:
                answer, sources = re.split("sources?\s?used", answer, flags=re.IGNORECASE)
                source_list = re.findall("source.*\d+", sources, flags=re.I)
                contri_docs.append(source_list)
            except ValueError as e:
                contri_docs.append([])
                print("Response does not have correct sources format") 
            
            reward = self.get_rewards(original_query, answer, rag_prompt)
            gold_reward = self.get_rewards(original_query, gold_answer)
            prompts.append(rag_prompt)
            answers.append(answer)
            rewards.append(reward)
            gold_rewards.append(gold_reward)
            count.update(1)
        
        
        return prompts, answers, rewards, gold_rewards, contri_docs

    def generate_answer(self, original_queries, doc_ids, gold_answers):      
        print("="*30 + " Generating Answers " + "="*30)
        answers = []
        rewards = []
        gold_rewards = []
        prompts = []
        contri_docs = []

        all_docs = self.document_retrieval_model.prediction(original_queries, doc_ids)
        original_queries = [query for queries in original_queries for query in queries]
        count = tqdm(total=len(original_queries), desc='RAG Iterations', position=0)
        
        for original_query, docs, gold_answer in zip(original_queries, all_docs, gold_answers):      
            knowledge_base = []
            ctr = 0
            for doc in docs:
                knowledge_base.append(f"Source {ctr+1}: {doc}")
                ctr+=1
            
            rag_prompt = RAG_CITATION_PROMPT.format(original_query = original_query, knowledge_base = "\n\n".join(knowledge_base))
            answer=self.language_model(rag_prompt).split("[/INST]")[-1]
            try:
                answer, sources = re.split("sources?\s?used", answer, flags=re.IGNORECASE)
                source_list = re.findall("source.*\d+", sources, flags=re.I)
                contri_docs.append(source_list)
            except ValueError as e:
                contri_docs.append([])
                print("Response does not have correct sources format") 
            
            reward = self.get_rewards(original_query, answer, rag_prompt)
            gold_reward = self.get_rewards(original_query, gold_answer)
            prompts.append(rag_prompt)
            answers.append(answer)
            rewards.append(reward)
            gold_rewards.append(gold_reward)
            count.update(1)
        
        
        return prompts, answers, rewards, gold_rewards, contri_docs
        
        


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
        responses=self.language_model([rag_prompt]*self.l, batch_process=True)
        responses = [response.split("[/INST]")[-1] for reponse in responses]
        answers = []
        contri_docs = []
        # for i in range(self.l):         
        if self.training_mode == TrainingMode().ResponseWithCitation:   
            for reponse in responses             
                try:
                    answer, sources = re.split("sources?\s?used", reponse, flags=re.IGNORECASE)
                    #TODO: Fall back to sentence similarity using llm's answer
                    source_list = re.findall("source.*\d+", sources, flags=re.I)
                except ValueError as e:
                    try:
                        with open("output/doc_citation_error.txt","a") as f:
                            f.write("Response does not have correct sources format:\nResponse:\n{1}".format(reponse.encode('utf-8')))
                    except:
                        with open("output/doc_citation_error.txt","w") as f:
                            f.write("Response does not have correct sources format:\nResponse:\n{1}".format(reponse.encode('utf-8')))
                answers.append(answer)
                contri_docs.append(source_list)
        else:
            answers = responses
            contri_docs = self.get_cited_documents(responses, original_query, top_k_docs)

        
        # contri_docs = top_k_docs*self.l            

        return answers, contri_docs   

    def get_cited_documents(self, responses, original_query, top_k_docs):
        cited_documents = []
        docs_embedding = [self.citation_model.encode(doc, convert_to_tensor=True) for doc in top_k_docs] 
        for response in responses:
            response_embedding = self.citation_model.encode(response, convert_to_tensor=True)
            scores = np.array([util.pytorch_cos_sim(response_embedding, doc_embedding).tolist()[0][0] for doc_embedding in docs_embedding])
            
            doc_indx = np.where(scores > 0.5)[0]
            cited_documents.append(np.array(top_k_docs)[doc_indx])
                
            
        
        # for i in range(self.l):
        #     prompt = EXTRACT_CITATION_PROMPT.format(original_query=original_query, extracts=top_k_docs, answer=responses[i])
        #     self.language_model(prompt)
        # return
        return cited_documents

    def get_rewards(self, original_query, responses, rag_prompt=None):
        # max_tries = 5
        # j = 0
        # do_retry = True
        reward_prompts = [REWARD_PROMPT.format(original_query = rag_prompt, answer = response) for response in responses]
        
        # reward=-1
        # if rag_prompt:
        #     reward_prompt = REWARD_PROMPT.format(original_query = rag_prompt, answer = response)
        # else:
        #     reward_prompt = REWARD_PROMPT.format(original_query = original_query, answer = response)

        # while j < max_tries and do_retry:
        #     if j >1:
        #         print("Reward generation trail #", str(j))
        #         try:
        #             with open("output/reward_error.txt","a") as f:
        #                 f.write("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(str(max_tries), reward_prompt.encode('utf-8'), reward_response.encode('utf-8')))
        #                 # raise Exception("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))
        #         except e:
        #             print("Exception in reward generation ", e)
        #             pass

        #     j=j+1
        reward_responses = self.language_model(reward_prompts, batch_process=True)
        rewards = []
        for reward_response in reward_responses:
            match = re.search("Final Score: ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("Total score = ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("Score: ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("([0-9]+) out of 5", reward_response)       

            if match:
                rewards.append(match.group(1)) 
            else:
                rewards.append(None)
                try:
                    with open("output/reward_error.txt","a") as f:
                        f.write("Unable to generate reward with the prompt:\n{1} \nResponse:\n{2}".format(reward_prompt.encode('utf-8'), reward_response.encode('utf-8')))
                        # raise Exception("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))
                except:
                    with open("output/reward_error.txt","w") as f:
                        f.write("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(str(max_tries), reward_prompt.encode('utf-8'), reward_response.encode('utf-8')))
                    
                
        
        # if do_retry:
        #     try:
        #         with open("output/reward_error.txt","a") as f:
        #             f.write("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(str(max_tries), reward_prompt.encode('utf-8'), reward_response.encode('utf-8')))
        #             # raise Exception("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))
        #     except:
        #         with open("output/reward_error.txt","w") as f:
        #             f.write("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(str(max_tries), reward_prompt.encode('utf-8'), reward_response.encode('utf-8')))
                    
        
        return reward
    
    def get_augmented_queries(self, qa_prompt, original_query):
        '''
        Extracts query samples from the language model
        '''

        # max_tries = 5
        response = ""
        # j = 0
        # sanity_check = False

        # while j < max_tries and sanity_check == False:
        #     j += 1
        response = self.language_model(qa_prompt, SAMPLING_PARAMS_DICT)
            # sanity_check = True
            # for i in range(1, self.n+1):
            #     if f"{i}." not in response:
            #         sanity_check = False
            #         break
        
        response = response.split(qa_prompt)[-1] #Remove the prompt from the response       
        response = re.sub(r'<s>|</s>|<pad>|[\[|\"|\'|\/]+INST\]', '', response) #Remove special tokens
        queries = []
        for i in range(1, self.n):
            pattern = "Version " + str(i)
            match = re.search(pattern + r"\.(.*\?)",response)
            if not match:
                match = re.search(str(i)+r"\.(.*\?)",response)
            if not match and i == self.n - 1:
                match = re.search(str(i)+r"\.(.*)",response)
            if not match:
                try:
                    f = open("output/query_aug_error.txt","a")
                except:
                    f = open("output/query_aug_error.txt","w")
                f.write(f"Query version {i} not found in response:")
                f.write(str(response))
                f.close()
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









