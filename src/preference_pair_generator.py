import numpy as np

from llm import LLM

class PreferencePairGenerator:
    def __init__(self, rag_model: LLM, m:int, n: int, l: int,):
        '''Generate preference pairs for a training loop of RAG pipeline

        Parameters:
        -----------
        rag_model
            RAG model to generate responses and corresponding rewards
        '''
        self.rag_model = rag_model     
        self.m = m
        self.n = n
        self.l = l
    
    def get_document_rewards(self, rho, winning_answers_idx, top_documents, contributing_documents):
        reward_docs = None
        for i, top_docs in enumerate(top_documents):
            if rho[i] < 0:
                new_rewards = np.full(len(top_docs), np.nan)
            else:
                temp_contri_docs = contributing_documents[(i*self.l)+winning_answers_idx[i]]
                new_rewards = np.array([rho[i] if doc in temp_contri_docs else 0 for doc in top_docs])

            if reward_docs is None:
                reward_docs = new_rewards
            else:
                reward_docs = np.vstack([reward_docs, new_rewards])
        return reward_docs     

    def get_augment_query_rewards(self,all_documents, document_rewards, top_documents, rho):
        rewards_aug_query = []
        for i in range(self.m):
            if rho[i] < 0:
                # Set the entire row to NaN if rho[i] is negative
                rewards_aug_query.extend([np.nan] * self.n)
            else:
                temp_rewards = []
                for j in range(self.n):
                    sum = 0
                    for k, top_doc in enumerate(top_documents[i]):
                        # top_doc=repr(top_doc)
                        if top_doc in all_documents[i][j]: # and len(top_doc)>0
                            indices_in_all_docs = np.where(np.asarray(all_documents[i][j]) == top_doc)
                            sum += document_rewards[i][k] / (indices_in_all_docs[0][0] + 1)
                    if sum > 0:
                        temp_rewards.append(sum)
                    else:
                        temp_rewards.append(np.nan)
                rewards_aug_query.extend(temp_rewards)

        # Convert list of lists to a NumPy array
        rewards_aug_query = np.array(rewards_aug_query)

        return rewards_aug_query

    def agg_query_rewards(self, aug_query_rewards, aug_queries):
        unique_queries = np.unique(aug_queries)  
        query_rewards = {}
        aug_queries=np.array(aug_queries)

        for query in unique_queries:
            indices = np.where(aug_queries == query)
            avg_reward = np.mean(aug_query_rewards[indices], axis=0)
            query_rewards[query] = avg_reward
        return query_rewards
    
    def get_max(self, arr):
        max_idx = 0
        max_value = 0
        for i, value in enumerate(arr):
            if value and value > max_value:
                max_idx = i
                max_value = value
        return max_value, max_idx
    
    def get_min(self, arr):
        min_idx = 0
        min_value = float('inf')
        for i, value in enumerate(arr):
            if value and value < min_value:
                min_idx = i
                min_value = value
        return min_value, min_idx

    def generateFirstPP(self, rag_prompts, answers, rewards):
        '''Generates the first preference pair matrix
        '''
        pref_pairs = []
        for i,rag_prompt in enumerate(rag_prompts):
            temp_ans = np.asarray(answers[i*self.l:(i+1)*self.l])
            temp_rewards = np.asarray(rewards[i*self.l:(i+1)*self.l])
            
            _, max_idx = self.get_max(temp_rewards)
            _, min_idx = self.get_min(temp_rewards)
            if max_idx != min_idx:
                pref_pairs.append((rag_prompt, temp_ans[max_idx], temp_ans[min_idx]))
            else:
                print("All rewards are same the in set: ", str(i))
            
        return pref_pairs

    def generateSecondPP(self, qa_prompts, aug_queries, all_docs, top_k_docs, all_rewards, contri_docs):
        '''generate the second preference pair matrix
        '''
        pairs = []
        for i, qa_prompt in enumerate(qa_prompts): 
            temp_rewards = []
            temp_contri_docs = []
            temp_aug_queries = []
            docs = contri_docs[i*(self.l*self.m) : (i+1)*(self.l*self.m)]
            rewards = all_rewards[i*(self.l*self.m) : (i+1)*(self.l*self.m)]
            for j in range(self.m):         
                temp_rewards.append(rewards[j*self.l : (j+1)*self.l])
                temp_contri_docs.extend(docs[j*self.l : (j+1)*self.l]) 
                temp_aug_queries.extend(aug_queries[(i*self.m)+j][1])                    
            
            
            rho = []
            winning_answers_idx = []
            for rewards in temp_rewards: 
                value, idx = self.get_max(rewards)       
                rho.append(value)
                winning_answers_idx.append(idx)
                
            # rho normalize
            rho = rho - np.mean(rho)

            temp_top_docs = top_k_docs[i*self.m : (i+1)*self.m]
            temp_all_docs = all_docs[i*self.l : (i+1)*self.l]       
            document_rewards = self.get_document_rewards(rho, winning_answers_idx, temp_top_docs, temp_contri_docs)         
            aug_query_rewards = self.get_augment_query_rewards(temp_all_docs, document_rewards, temp_top_docs, rho)
            
            agg_aug_query_rewards = self.agg_query_rewards(aug_query_rewards, temp_aug_queries)

            unique_queries = list(agg_aug_query_rewards.keys())           
            for i in range(len(unique_queries)):
                for j in range(i + 1, len(unique_queries)):
                    query1, query2 = unique_queries[i], unique_queries[j]
                    reward1, reward2 = agg_aug_query_rewards[query1], agg_aug_query_rewards[query2]
                    if reward1 > reward2:
                        pairs.append((qa_prompt, query1, query2))
                    else:
                        pairs.append((qa_prompt, query2, query1))
        return pairs