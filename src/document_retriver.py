from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page
from functools import partial
from collections import defaultdict
   
class DocumentRetrievalModel:
    def __init__(self, k, p, path_to_index = "/scratch/workspace/arana_umass_edu-goldamn_project/credit_agreement_pilot_8"):
        # Initialize components specific to Document Retrieval
        self.path_to_index = path_to_index
        self.k = k
        self.p = p
        self.RAG = RAGPretrainedModel.from_index(path_to_index)

    def prediction(self, queries, doc_ids):
        all_documents = []
        for i, doc_id in enumerate(doc_ids):
            for query in queries[i]:
                result = self.RAG.search(query=query, k=self.k, doc_ids=[doc_id]) 
                row = []
                for x in result:
                    row.append(x['content'])            
                all_documents.append(row)
        return all_documents
        
        
    def train(self, aug_queries, doc_id):
        # Logic to retrieve and return a list of documents based on the query
        # Placeholder implementation
        all_documents = []
        all_ranks = []
        for query in aug_queries:   
            result = self.RAG.search(query=query, k=self.p, doc_ids=[doc_id])
            row = []      
            for x in result:
                row.append(x['content'])
                all_ranks.append(x['rank'])               
            all_documents.append(row)
            
        top_k_documents = self.reciprocal_rank_fusion(all_documents, all_ranks)
        return top_k_documents, all_documents
    
    def reciprocal_rank_fusion(self, all_documents, all_ranks):
        '''
        Apply RRF to multiple lists of items with their ranks, and return top-k
        documents based on thecombined scores.
        
        Parameters:
        -----------
            param rank_lists: A list of lists, where each inner list represents a rank list.
            param k: A constant added to the denominator in the RRF formula. Default is 60.
            return: A dictionary with item IDs as keys and their combined RRF scores as values.
        '''
        all_documents = [item for sublist in all_documents for item in sublist]
        rrf_scores = {}
        for item, rank in zip(all_documents, all_ranks):
            if item not in rrf_scores:
                rrf_scores[item] = 0
            rrf_scores[item] = rrf_scores[item]+ (1 / (self.k + rank))
       
        sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse=True)     
        return sorted_items[:self.k]
