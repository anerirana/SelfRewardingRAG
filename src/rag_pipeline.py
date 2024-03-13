import numpy as np
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer,AutoModelForCausalLM,TrainingArguments
from document_retriver import DocumentRetrievalModel 
from constants import *
import re
from trl import DPOTrainer
from unsloth import FastLanguageModel


# Utilities
def exists(val):
    return val is not None

def default(val, default_value):
    return val if exists(val) else default_value

class LLM(nn.Module):
    def __init__(self, model_name):
        super(LLM, self).__init__()
        '''Retrival Augmented Generation model to generate responses for a given query 
        and a set of retieved documents
        '''
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                    trust_remote_code=True,
                    device_map="auto")
                    
        
        self.model = torch.compile(self.model, mode = "max-autotune", backend="inductor")
    def forward(self, prompt, param_dict=None):
        '''Generate response for the given prompt

        Parameters:
        -----------
        prompt
            The prompt used as input to the generative model.
            Could be a RAG or QA or reward prompt.'
        '''
        messages = [{"role":"user", "content":prompt}]
        encoded = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        inputs = encoded.to(self.model.device)
        # input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)    
        if param_dict is None:
            outputs = self.model.generate(inputs, temperature=0.7, top_p=0.99, repetition_penalty=1.2, min_new_tokens=16, max_new_tokens=2048, do_sample=True)
        else:
            #TODO: try fixing temperature=0.7
            outputs = self.model.generate(inputs, temperature=param_dict["temperature"], top_p=param_dict["top_p"], repetition_penalty=param_dict["repetition_penalty"], min_new_tokens=param_dict["min_new_tokens"], max_new_tokens=param_dict["max_new_tokens"], do_sample=True)

        return self.tokenizer.decode(outputs[0])

    def train(self, training_dataset, batch_size=32, num_epochs=3):
        dpo_trainer = DPOTrainer(
            model,
            model_ref=None,
            args=training_args,
            beta=0.1,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
        dpo_trainer.train()
        '''Train the RAGModel on the given training_dataset

        Parameters:
        -----------
        training_dataset
            Contains the training data for query augemntation or reward generation or answer generator
        '''
        
        pass

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
        self.base_model = default(config.get('BaseModel'), 'google/flan-t5-xxl') 


    def train(self, original_query, doc_ids):
        '''Executes a training loop of the RAGPipeline

        Parameters:
        -----------
        original_query
            The original query to generate responses for
        '''
        # Create instances of required models
        language_model = LLM(self.base_model)
        document_retrieval_model = DocumentRetrievalModel()   
        pp_generator = PreferencePairGenerator(language_model)

        qa_prompt = QUERY_AUGMENTATION_PROMPT.format(n=self.n, original_query=original_query)

        aug_queries = []
        all_documents = []
        top_documents = []
        all_responses = []
        all_rewards = np.zeros((self.m, self.l), dtype=float)
        contributing_documents = []
        first_pps = []
        count = tqdm(total=self.m, desc='Iterations', position=0)

        for i in range(self.m):
            queries = self.extract_query_samples(language_model, qa_prompt)

            top_k_docs, all_docs = document_retrieval_model.forward(queries, doc_ids, self.p, self.k)

            rag_prompt = RAG_PROMPT.format(original_query = original_query, documents = top_k_docs)

            #TODO: Need to sample from the language model to get l answers
            responses = self.get_query_responses(language_model, rag_prompt)
            

            # TODO: check this dimension
            contri_docs = top_k_docs*self.l
            # responses, contri_docs = self.parser_responses(responses,i)                
                 
            rewards = [self.get_rewards(language_model, original_query, response) for response in responses]
            with open("./output2.txt","w") as f:
                f.write("responses: ")
                f.write(str(responses))
                f.write("rewards: ")
                f.write(str(rewards))
            # break

            pp1 = pp_generator.generateFirstPP(rag_prompt, responses, rewards)
            


            first_pps.append(pp1)
            aug_queries.append(queries)
            all_documents.append(all_docs)           
            top_documents.append(top_k_docs[0])
            all_responses.append(responses)
            all_rewards[i] = rewards
            contributing_documents.append(contri_docs)
            count.update(1)
        all_responses=np.array(all_responses)
        pp2 = pp_generator.generateSecondPP(qa_prompt, aug_queries, all_documents, top_documents, all_rewards, contributing_documents)

        print("aug_queries: ")            
        print(str(self.find_list_dimensions(aug_queries)))
        print(">>"*100)
        print("all_documents: ")
        print(str(self.find_list_dimensions(all_documents)))
        print(">>"*100)
        print("top_documents: ")
        print(str(self.find_list_dimensions(top_documents)))
        print(">>"*100)
        print("all_responses: ")
        print(str(all_responses.shape))
        print(">>"*100)
        print("all_rewards: ")
        print(str(all_rewards.shape))
        print(">>"*100)
        print("contributing_documents: ")
        print(str(self.find_list_dimensions(contributing_documents)))
        print(">>"*100)
        print("first_pps: ")
        print(str(self.find_list_dimensions(first_pps)))
        print(">>"*100)
        print("second_pps: ")
        print(len(pp2))

        with open("./output.txt","w") as f:
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

        #TODO: load pp1 and pp2 in a dataset loader for training        
        language_model.train()
    
    def find_list_dimensions(self,lst):
        if not isinstance(lst, list) or not lst:  # Base case: not a list or empty list
            return []
        return [len(lst)] + self.find_list_dimensions(lst[0])
          
    def get_query_responses(self, language_model, rag_prompt):
        responses = []
        for i in range(self.l):
            text=language_model(rag_prompt, SAMPLING_PARAMS_DICT)
            answer_index = text.find("documents you used to answer the question.")
            # If "\nAnswer:" is found, extract the text after it
            if answer_index != -1:
                text = text[answer_index + len('documents you used to answer the question.')+1:]  # +8 to skip past the "\nAnswer:" part
            else:
                print("The string '\nAnswer:' was not found in the text.")

            # r = r.split(rag_prompt)[-1] #Remove the prompt from the response
            # r = r.replace("<s>", "").replace("</s>", "").replace("<pad>", "") #Remove special tokens

            responses.append(text)
       
    
        return responses    

    # TODO
    def parser_responses(self, responses, index):
        return {}

    def get_rewards(self, language_model, original_query, response):
        max_tries = 5
        j = 0
        do_retry = True
        reward=-1
        while j < max_tries and do_retry:
            
            j=j+1
            reward_response = language_model(REWARD_PROMPT.format(original_query = original_query, answer = response))
            
            match = re.search("Final Score: ([0-9]+) out of 5", reward_response)
            if not match:
                match =  re.search("Score: ([0-9]+) out of 5", reward_response)

            if match:
                reward = match.group(1)  
                do_retry = False
        
            if do_retry:
                with open("./error.txt","a") as f:
                    f.write("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))


                # raise Exception("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))
        
        return reward
    
    def extract_query_samples(self, language_model, qa_prompt):
        '''
        Extracts query samples from the language model
        '''

        max_tries = 5
        response = ""
        j = 0
        sanity_check = False

        while j < max_tries and sanity_check == False:
            j += 1
            response = language_model(qa_prompt, SAMPLING_PARAMS_DICT)
            sanity_check = True
            for i in range(1, self.n+1):
                if f"{i}." not in response:
                    sanity_check = False
                    break
        
        queries = response.split(qa_prompt)[-1] #Remove the prompt from the response
        queries = queries.replace("<s>", "").replace("</s>", "").replace("<pad>", "") #Remove special tokens
        
        for i in range(1, self.n+1):
            queries = queries.replace(f"{i}.", "")
        
        queries = queries.strip().split("\n") #Split the response into a list of queries
        queries=queries[2:]

        return queries


class PreferencePairGenerator:
    def __init__(self, rag_model: LLM):
        '''Generate preference pairs for a training loop of RAG pipeline

        Parameters:
        -----------
        rag_model
            RAG model to generate responses and corresponding rewards
        '''
        self.rag_model = rag_model
    
    def get_document_rewards(self, rho, winning_answers, top_documents, contributing_documents):
        reward_docs = None
        for i, top_docs in enumerate(top_documents):
            if rho[i] < 0:
                new_rewards = np.full(len(top_docs), np.nan)
            else:
                new_rewards = np.array([rho[i] if doc in contributing_documents[i][winning_answers[i]] else 0 for doc in top_docs])
            
            if reward_docs is None:
                reward_docs = new_rewards
            else:
                reward_docs = np.vstack([reward_docs, new_rewards])
        return reward_docs     
    
    def get_augment_query_rewards(self,all_documents, document_rewards, top_documents, rho, aug_queries):
        rewards_aug_query = []
        for i in range(0,len(aug_queries)):
            if rho[i] < 0:
                # Set the entire row to NaN if rho[i] is negative
                rewards_aug_query.append([np.nan] * len(aug_queries[1]))
            else:
                z = []
                for j in range(len(aug_queries[i])):
                    p = 0
                    for h, top_doc in enumerate(top_documents[i]):
                        top_doc=repr(top_doc)
                        if top_doc in all_documents[i][j] and len(top_doc)>0:
                            index_in_all_docs = np.where(all_documents[i][j] == top_doc)
                            p += document_rewards[i][h] / (index_in_all_docs[0][0] + 1)
                    if p > 0:
                        z.append(p)
                    else:
                        z.append(np.nan)
                rewards_aug_query.append(z)
        
        # Convert list of lists to a NumPy array
        rewards_aug_query = np.array(rewards_aug_query)

        return rewards_aug_query
    
    def agg_query_rewards(self, aug_query_rewards, aug_queries):
        unique_queries = np.unique(aug_queries)  
        agg_query_rewards = {}
        aug_queries=np.array(aug_queries)

        for query in unique_queries:
            indices = np.where(aug_queries == str(query))
            avg_reward = np.mean(aug_query_rewards[indices], axis=0)
            agg_query_rewards[query] = avg_reward
        return agg_query_rewards

    def generateFirstPP(self, prompt, responses, rewards):
        '''Generates the first preference pair matrix
        '''
        responses = np.asarray(responses)
        rewards = np.asarray(rewards)

        max_idx = np.argmax(rewards)
        min_idx = np.argmin(rewards)
        
        return (prompt, responses[max_idx], responses[min_idx])  # Placeholder for a matrix

    def generateSecondPP(self, qa_prompt, aug_queries, all_documents, top_documents, all_rewards, contributing_documents):
        '''generate the second preference pair matrix
        
        '''
        rho = np.max(all_rewards, axis=1)
        # rho normalize
        rho = rho - np.mean(rho)

        winning_answers = np.argmax(all_rewards, axis=1)
        document_rewards = self.get_document_rewards(rho, winning_answers,top_documents,contributing_documents)
        
        aug_query_rewards = self.get_augment_query_rewards(all_documents, document_rewards, top_documents,rho,aug_queries)

        agg_query_rewards = self.agg_query_rewards(aug_query_rewards, aug_queries)
        

        unique_queries = list(agg_query_rewards.keys())
        pairs = []
        for i in range(len(unique_queries)):
            for j in range(i + 1, len(unique_queries)):
                query1, query2 = unique_queries[i], unique_queries[j]
                reward1, reward2 = agg_query_rewards[query1], agg_query_rewards[query2]
                if reward1 > reward2:
                    pairs.append((qa_prompt, query1, query2))
                else:
                    pairs.append((qa_prompt, query2, query1))
        return pairs