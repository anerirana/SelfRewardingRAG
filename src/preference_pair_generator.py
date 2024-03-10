import numpy as np

from llm import LLM

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
        print("RHO value is")
        print(rho)  
        # print()
        for i, top_docs in enumerate(top_documents):
            if rho[i] < 0:
                new_rewards = np.full(len(top_docs), np.nan)
            else:
                new_rewards = np.array([rho[i] if doc in contributing_documents[i][winning_answers[i]] else 0 for doc in top_docs])
            
            if reward_docs is None:
                reward_docs = new_rewards
            else:
                reward_docs = np.vstack([reward_docs, new_rewards])
        # print("REWARDS")
        # print(reward_docs)
        # print(reward_docs.shape)
        return reward_docs     
    
    def get_augment_query_rewards(self,all_documents, document_rewards, top_documents, rho, aug_queries):
        # print(top_documents)
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
                        # print(top_doc)
                        # print(all_documents[i][j])
                        if top_doc in all_documents[i][j] and len(top_doc)>0:
                            print("HII")
                            print()
                            index_in_all_docs = np.where(all_documents[i][j] == top_doc)
                            print(index_in_all_docs)
                            p += document_rewards[i][h] / (index_in_all_docs[0][0] + 1)
                    if p > 0:
                        z.append(p)
                    else:
                        z.append(0)
                rewards_aug_query.append(z)
        
        # Convert list of lists to a NumPy array
        rewards_aug_query = np.array(rewards_aug_query)

        return rewards_aug_query
    
    def agg_query_rewards(self, aug_query_rewards, aug_queries):
        unique_queries = np.unique(aug_queries)  
        agg_query_rewards = {}
        print(aug_queries)
        aug_queries=np.array(aug_queries)

        for query in unique_queries:
            print(query)
            indices = np.where(aug_queries == str(query))
            print(indices)
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
        # print(document_rewards)
        aug_query_rewards = self.get_augment_query_rewards(all_documents, document_rewards, top_documents,rho,aug_queries)

        agg_query_rewards = self.agg_query_rewards(aug_query_rewards, aug_queries)
        print(agg_query_rewards)

        unique_queries = list(agg_query_rewards.keys())
        pairs = []
        for i in range(len(unique_queries)):
            for j in range(i + 1, len(unique_queries)):
                query1, query2 = unique_queries[i], unique_queries[j]
                reward1, reward2 = agg_query_rewards[query1], agg_query_rewards[query2]
                # print(reward1,reward2)
                if reward1 > reward2:
                    pairs.append((qa_prompt, query1, query2))
                else:
                    pairs.append((qa_prompt, query2, query1))
        return pairs