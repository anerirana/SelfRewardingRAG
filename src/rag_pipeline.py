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
        
        self.query_aug_batch_size = default(config.get('QueryAugmentationBatchSize'), 16)
        self.answer_gen_batch_size = default(config.get('AnswerGenerationBtachSize'), 8)
        self.reward_gen_batch_size = default(config.get('RewardGenerationBtachSize'), 8)
        self.language_model = LLM(default(config.get('LanguageModelName'), 'mistralai/Mistral-7B-Instruct-v0.2'))
        self.citation_model = SentenceTransformer(default(config.get('CitationModelName'), 'sentence-transformers/all-mpnet-base-v2'))
        self.training_mode = default(config.get('TrainingMode'), TrainingMode().SimiliarityScoreCitation)

        self.document_retrieval_model = DocumentRetrievalModel(self.k, self.p, PATH_TO_INDEX)   
        self.pp_generator = PreferencePairGenerator(self.language_model, self.m, self.n, self.l)


    #TODO: Implement prediction to get RAG responses and their rewards after training

    def train(self, original_queries, epoch, doc_ids=None):
        '''Executes a training loop of the RAGPipeline

        Parameters:
        -----------
        original_query
            The original query to generate responses for
        '''
        dpo_dataset_dict = {}
        # f = open(OUTPUT_DIRECTORY + "all_variables_epoch_" + str(epoch) + ".txt","x")

        start_time = time.time()
        qa_prompts = [QUERY_AUGMENTATION_PROMPT.format(n=self.n-1, original_query=original_query) for original_query in original_queries]
        print("len(qa_prompts): ", len(qa_prompts), flush=True) # len(original_queries)
        aug_queries = self.get_augmented_queries(qa_prompts, original_queries, doc_ids)
        print("len(aug_queries): ", len(aug_queries), flush=True)
        query_aug_time = time.time()-start_time
        print("Query aug time: ", str(query_aug_time), flush=True)

        start_time = time.time()
        top_k_docs, all_docs = self.document_retrieval_model.train_batch(aug_queries)
        print("len(all_docs): ", len(all_docs), flush=True)
        print("len(all_docs)[0]: ", len(all_docs[0]), flush=True)
        print("len(top_k_docs): ", len(top_k_docs), flush=True)
        print("len(top_k_docs)[0]: ", len(top_k_docs[0]), flush=True)
        doc_ret_time = time.time()-start_time
        print("Doc retrieval time: ", str(doc_ret_time), flush=True)

        start_time = time.time()
        rag_prompts = self.get_rag_prompts(top_k_docs, original_queries)
        print("len(rag_prompts): ", len(rag_prompts), flush=True)
        answers, contri_docs = self.get_query_responses(rag_prompts, top_k_docs)
        print("len(answers): ", len(answers), flush=True)
        print("len(contri_docs): ", len(contri_docs), flush=True)
        print("len(contri_docs[0]): ", len(contri_docs[0]), flush=True)
        resp_gen_time = time.time()-start_time
        print("Response generation time: ", str(resp_gen_time), flush=True)

        start_time = time.time()      
        rewards = self.get_rewards(answers, rag_prompts)  
        print("len(rewards): ", len(rewards), flush=True)
        reward_gen_time = time.time()-start_time
        print("Reward generation time: ", str(reward_gen_time), flush=True)

        pp1 = self.pp_generator.generateFirstPP(rag_prompts, answers, rewards)
        print("numer of preference pairs type 1: ", len(pp1), flush=True)
        pp2 = self.pp_generator.generateSecondPP(qa_prompts, aug_queries, all_docs, top_k_docs, rewards, contri_docs)
        print("numer of preference pairs type 2: ", len(pp2), flush=True)      
            
        dpo_dataset_dict = self.dpo_parsing(pp1,pp2)
                
        with open(OUTPUT_DIRECTORY + "dpo_preference_pairs_" + str(epoch) + ".json", "w") as f: 
            json.dump(dpo_dataset_dict, f)
        
        # DPO Trainining
        self.language_model.train(epoch)
  
    def get_rag_prompts(self, top_k_docs, original_queries):
        rag_prompts = []
        for i,docs in enumerate(top_k_docs):
            original_query = original_queries[i//self.m]
            knowledge_base = []
            ctr = 0
            for doc in docs:
                knowledge_base.append(f"Source {ctr+1}: {doc}")
                ctr+=1
            
            rag_prompts.append(RAG_CITATION_PROMPT.format(original_query = original_query, knowledge_base = "\n\n".join(knowledge_base)))
        return rag_prompts
    
    def compute_scores(self,references, predictions):
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
        scores['BLEU'] = bleu_metric.compute(predictions=predictions, references=references)

        # Load and compute ROUGE score
        rouge_metric = load_metric("rouge")
        scores['ROUGE'] = rouge_metric.compute(predictions=predictions, references=references)

        # Load and compute METEOR score
        meteor_metric = load_metric("meteor")
        scores['METEOR'] = meteor_metric.compute(predictions=predictions, references=references)

        # You can add more metrics here in a similar fashion

        return scores

    def prediction(self,original_queries,doc_ids,gold_answers):
        start_time = time.time()
        qa_prompts = [QUERY_AUGMENTATION_PROMPT.format(n=self.n-1, original_query=original_query) for original_query in original_queries]
        print("len(qa_prompts): ", len(qa_prompts), flush=True) #x
        aug_queries = self.get_augmented_queries(qa_prompts, original_queries, doc_ids)
        print("len(aug_queries): ", len(aug_queries), flush=True) # x * m
        print("len(aug_queries[0][1]):", len(aug_queries[0][1]), flush=True)
        query_aug_time = time.time()-start_time
        print("Query aug time: ", str(query_aug_time), flush=True)

        start_time = time.time()
        top_k_docs, all_docs = self.document_retrieval_model.train_batch(aug_queries)
        print("len(all_docs): ", len(all_docs), flush=True)
        print("len(all_docs)[0]: ", len(all_docs[0]), flush=True)
        print("len(top_k_docs): ", len(top_k_docs), flush=True)
        print("len(top_k_docs)[0]: ", len(top_k_docs[0]), flush=True)
        doc_ret_time = time.time()-start_time
        print("Doc retrieval time: ", str(doc_ret_time), flush=True)

        start_time = time.time()
        rag_prompts = self.get_rag_prompts(top_k_docs, original_queries)
        print("len(rag_prompts): ", len(rag_prompts), flush=True)
        answers, contri_docs = self.get_query_responses(rag_prompts, top_k_docs)
        print("len(answers): ", len(answers), flush=True)
        print("len(contri_docs): ", len(contri_docs), flush=True)
        print("len(contri_docs[0]): ", len(contri_docs[0]), flush=True)
        resp_gen_time = time.time()-start_time
        print("Response generation time: ", str(resp_gen_time), flush=True)

        # start_time = time.time()      
        # rewards = self.get_rewards(answers, rag_prompts)  
        # print("len(rewards): ", len(rewards))
        # reward_gen_time = time.time()-start_time
        # print("Reward generation time: ", str(reward_gen_time))

        # gold_rewards = self.get_rewards(gold_answers, rag_prompts)
        # print("len(gold_rewards): ", len(gold_rewards))  
        
        return rag_prompts, answers, [], [], contri_docs

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
        
    def dpo_parsing(self,pp1,pp2):
        dataset = []
        for i in range(0,len(pp1)):
            dataset.append({"prompt": pp1[i][0],
                            "chosen": pp1[i][1],
                            "rejected": pp1[i][2]})
                
        for i in range(0,len(pp2)):
            dataset.append({"prompt": pp2[i][0],
                            "chosen": pp2[i][1],
                            "rejected": pp2[i][2]})
                 
        return dataset
    
    def find_list_dimensions(self,lst):
        if not isinstance(lst, list) or not lst:  # Base case: not a list or empty list
            return []
        return [len(lst)] + self.find_list_dimensions(lst[0])
          
    def get_query_responses(self, rag_prompts, top_k_docs):
        responses=self.language_model(rag_prompts, num_return_sequences=self.l, batch_process=True, batch_size=self.answer_gen_batch_size)
        # print("len(responses): ", len(responses))

        answers = []
        contri_docs = []
        for response in responses:
            response = response.split("[/INST]")[-1]
              
            try:
                answer, sources = re.split("sources?\s?used", response, flags=re.IGNORECASE)
                #TODO: Fall back to sentence similarity using llm's answer
                source_list = re.findall("source.*\d+", sources, flags=re.I)
                answers.append(answer)
                contri_docs.append(source_list)
            except ValueError as e:
                answers.append(response)
                contri_docs.append([])
                try:
                    with open("output/doc_citation_error.txt","a") as f:
                        f.write("Response does not have correct sources format:\nResponse:\n{0}".format(response.encode('utf-8')))
                except:
                    with open("output/doc_citation_error.txt","w") as f:
                        f.write("Response does not have correct sources format:\nResponse:\n{0}".format(response.encode('utf-8')))
            

        if self.training_mode == TrainingMode().SimiliarityScoreCitation:
            contri_docs = []
            for i, docs in  enumerate(top_k_docs):   
                contri_docs.append(self.get_cited_documents(answers[i*self.l:(i+1)*self.l], docs))
        return answers, contri_docs   

    def get_cited_documents(self, responses, top_k_docs):
        cited_documents = []
        docs_embedding = [self.citation_model.encode(doc, convert_to_tensor=True) for doc in top_k_docs] 
        for response in responses:
            response_embedding = self.citation_model.encode(response, convert_to_tensor=True)
            scores = np.array([util.pytorch_cos_sim(response_embedding, doc_embedding).tolist()[0][0] for doc_embedding in docs_embedding])
            
            doc_indx = np.where(scores > 0.5)[0]
            cited_documents.append(np.array(top_k_docs)[doc_indx])
        return cited_documents

    def get_rewards(self, answers, rag_prompts):
        reward_prompts = [REWARD_PROMPT.format(original_query = rag_prompts[i//self.l], answer = answer) for i,answer in enumerate(answers)]
        reward_responses = self.language_model(reward_prompts, batch_process=True, batch_size=self.reward_gen_batch_size)
        
        rewards = []
        for i,reward_response in enumerate(reward_responses):
            
            match = re.search("Final Score: ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("Total score = ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("Score: ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("([0-9]+) out of 5", reward_response) 

            if match:
                rewards.append(int(match.group(1))) 
            else: # look for numeric rewards
                patterns = [
                    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten) out of (?:5|five)",
                    r"Score: (\d+|one|two|three|four|five|six|seven|eight|nine|ten) ?(?:/|out of) ?(5|five)",
                    r"Total Score: (\d+|one|two|three|four|five|six|seven|eight|nine|ten) out of (?:10|ten)",
                    r"a score of (\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
                ]
                found = False
                for pattern in patterns:
                    match = re.search(pattern, reward_response, flags=re.IGNORECASE)
                    if match:     
                        found = True
                        break
           
                if found:
                    # Assuming the score is always the first group
                    score_text = match.group(1)
                    score = self.text_to_number(score_text)
                    rewards.append(int(score)) # Convert score to integer
                else:
                    rewards.append(None)
                    try:
                        with open(OUTPUT_DIRECTORY + "reward_error.txt","a") as f:
                            f.write("Unable to generate reward with the prompt:\n{0} \nResponse:\n{1}".format(reward_prompts[i].encode('utf-8'), reward_response.encode('utf-8')))
                            # raise Exception("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))
                    except:
                        with open(OUTPUT_DIRECTORY + "reward_error.txt","w") as f:
                            f.write("Unable to generate reward with the prompt:\n{0} \nResponse:\n{1}".format(reward_prompts[i].encode('utf-8'), reward_response.encode('utf-8')))
            
        return rewards
    
    def get_augmented_queries(self, qa_prompts, original_queries, doc_ids):
        '''
        Extracts augmented query samples from the language model
        '''   
        responses = self.language_model(qa_prompts, num_return_sequences=self.m, batch_process=True, batch_size=self.query_aug_batch_size)
        
        all_augmented_queries = []
        for i,response in enumerate(responses):
            qa_prompt = qa_prompts[i//self.m]
            original_query = original_queries[i//self.m]
            doc_id = doc_ids[i//self.m]
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
                        f = open(OUTPUT_DIRECTORY + "query_aug_error.txt","a")
                    except:
                        f = open(OUTPUT_DIRECTORY + "query_aug_error.txt","w")
                    f.write(f"Query version {i} not found in response:")
                    f.write(str(response))
                    f.close()
                    break
                else:
                    queries.append(match.group(1))
                
            queries = [q.strip() for q in queries]
            queries.append(original_query)
            if len(queries) < self.n:
                queries.extend(random.choices(queries, k=self.n-len(queries)))
            else:
                queries = queries[:self.n]
            all_augmented_queries.append((doc_id, queries))
        return all_augmented_queries
    
    def text_to_number(self,text):
      numbers = {
          'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
          'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
      }
      return numbers.get(text.lower(), text)










