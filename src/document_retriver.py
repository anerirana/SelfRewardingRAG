from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page
from functools import partial
from collections import defaultdict
   
class DocumentRetrievalModel:
    def __init__(self, path_to_index = "../.ragatouille/colbert/indexes/credit_agreement_database"):
        # Initialize components specific to Document Retrieval
        self.path_to_index = path_to_index
        self.RAG = RAGPretrainedModel.from_index(path_to_index)
        
    def forward(self, queries, doc_ids, p, k):
        # Logic to retrieve and return a list of documents based on the query
        # Placeholder implementation
        all_results = self.RAG.search(query=queries, k=p, doc_ids=doc_ids)
        top_k_documents = []
        all_documents = []
        all_ranks = []
        for result in all_results:     
            row = []      
            for x in result:
                row.append(x['content'])
                all_ranks.append(x['rank'])               
            all_documents.append(row)
            
        top_k_documents.append(self.reciprocal_rank_fusion(all_documents, all_ranks, k))
        return top_k_documents, all_documents
    
    def reciprocal_rank_fusion(self, all_documents, all_ranks, k):
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
            rrf_scores[item] = rrf_scores[item]+ (1 / (k + rank))
       
        sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse=True)          
        return sorted_items[:k]
