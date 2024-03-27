
import numpy as np

from llm import LLM
from sentence_transformers import SentenceTransformer, util
from document_retriver import DocumentRetrievalModel 
from constants import *
from preference_pair_generator import PreferencePairGenerator
from document_retriver import DocumentRetrievalModel 
from constants import *
import re
from tqdm import tqdm


# Utilities
def exists(val):
    return val is not None

def default(val, default_value):
    return val if exists(val) else default_value


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
        self.citation_model = SentenceTransformer(default(config.get('CitationModel'), 'sentence-transformers/all-mpnet-base-v2'))
        

    #TODO: Implement prediction to get RAG responses and their rewards after training
    def prediction(self):
        pass

    def train(self, original_query, doc_ids=None):
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
    
        qa_prompt = QUERY_AUGMENTATION_PROMPT.format(n=self.n-1, original_query=original_query)

        aug_queries = []
        all_documents = []
        top_documents = []
        all_responses = []
        all_rewards = np.zeros((self.m, self.l), dtype=float)
        contributing_documents = []
        first_pps = []
        count = tqdm(total=self.m, desc='Iterations', position=0)

        for i in range(self.m):
            queries = self.extract_query_samples(language_model, qa_prompt, original_query)
            top_k_docs, all_docs = document_retrieval_model.forward(queries, doc_ids, self.p, self.k)

            knowledge_base = []
            ctr = 0
            for doc in top_k_docs:
                knowledge_base.append(f"Source {ctr+1}: {doc}")
                ctr+=1
            rag_prompt = RAG_PROMPT.format(original_query = original_query, knowledge_base = "\n\n".join(knowledge_base))
            responses = self.get_query_responses(language_model, rag_prompt)
            

            # TODO: check this dimension
            # contri_docs = top_k_docs*self.l
            # responses, contri_docs = self.parser_responses(responses,i)     
            contri_docs = self.get_cited_documents(language_model, responses, original_query, top_k_docs)
               
                 
            rewards = [self.get_rewards(language_model, original_query, response) for response in responses]
            with open("output/response_rewards.txt","w") as f:
                f.write("responses: ")
                f.write(str(responses))
                f.write("rewards: ")
                f.write(str(rewards))

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
        pp2 = pp_generator.generateSecondPP(qa_prompt, aug_queries, all_documents, top_documents, all_rewards, contributing_documents)

        print("second_pps: ")
        print(len(pp2))
        dpo_dataset_dict=self.dpo_parsing(first_pps,pp2)

        with open("output/all_variables.txt","w") as f:
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
            f.write("DPO dataset")
            f.write(str(dpo_dataset_dict))
            f.write(">>"*100)

        #TODO: load pp1 and pp2 in a dataset loader for training     
        
        
        language_model.train(dpo_dataset_dict)

    def dpo_parsing(self,first_pps,pp2):
        dpo_dataset_dict={"prompt": [],"chosen": [], "rejected": []}
        for i in range(0,len(first_pps)):
            dpo_dataset_dict["prompt"].append(first_pps[i][0])
            dpo_dataset_dict["chosen"].append(first_pps[i][1])
            dpo_dataset_dict["rejected"].append(first_pps[i][2])
        
        for i in range(0,len(pp2)):
            dpo_dataset_dict["prompt"].append(pp2[i][0])
            dpo_dataset_dict["chosen"].append(pp2[i][1])
            dpo_dataset_dict["rejected"].append(pp2[i][2])
        
        return(dpo_dataset_dict)
    
    def find_list_dimensions(self,lst):
        if not isinstance(lst, list) or not lst:  # Base case: not a list or empty list
            return []
        return [len(lst)] + self.find_list_dimensions(lst[0])
          
    def get_query_responses(self, language_model, rag_prompt):
        responses = []
        for i in range(self.l):
            llm_answer=language_model(rag_prompt, SAMPLING_PARAMS_DICT).split("[/INST]")[-1]
            try:
                answer, sources = re.split("sources?\s?used", llm_answer, flags=re.IGNORECASE)
            except ValueError as e:
                print("Response not in right format")
                #TODO: Fall back to sentence similarity using llm_answer

            # If "\nAnswer:" is found, extract the text after it
            # if answer_index != -1:
            #     text = text[answer_index+3:]  # +8 to skip past the "\nAnswer:" part
            # else:
            #     print("The string '\nAnswer:' was not found in the text.\n", text)

            # r = r.split(rag_prompt)[-1] #Remove the prompt from the response
            # r = r.replace("<s>", "").replace("</s>", "").replace("<pad>", "") #Remove special tokens
                
            source_list = re.findall("source.*\d+", sources, flags=re.I)

            responses.append(answer)
            print(sources)
            print(source_list)

        return responses    

    def get_cited_documents(self, language_model, responses, original_query, top_k_docs):
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
        #     language_model(prompt)
        # return
        return cited_documents

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
            with open("output/reward_error.txt","a") as f:
                f.write("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))


                # raise Exception("Unable to generate reward after {0} retries with the prompt:\n{1} \nResponse:\n{2}".format(max_tries, REWARD_PROMPT.format(original_query = original_query, answer = response), reward_response))
        
        return reward
    
    def extract_query_samples(self, language_model, qa_prompt, original_query):
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
        queries = re.sub(r'<s>|</s>|<pad>|[\[|\"|\'|\/]+INST\]', '', queries) #Remove special tokens
        
        for i in range(1, self.n+1):
            queries = re.sub(f"Version {i}.", "", queries)
            queries = re.sub(f"{i}.", "", queries)
        
        queries = queries.strip()
        queries = re.split(r"\n{1,2}", queries) #Split the response into a list of queries
        
        queries = [q.strip() for q in queries]
        queries = [q for q in queries if q != '']
        queries.append(original_query)
        
        if len(queries) < self.n:
            queries.extend(random.choice(queries, k=self.n-queries))
        else:
            queries = queries[:self.n]
        return queries
