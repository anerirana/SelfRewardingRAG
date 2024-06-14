from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page
from functools import partial
from collections import defaultdict
   
class DocumentRetrievalModel:
    def __init__(self, k, p, path_to_index):
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
        
    def train(self, aug_queries_with_doc_id):
        '''
        Retrieves and returns all the retrieved and top-k document chunks for the given augmented queries

        Parameters:
        -----------
        aug_queries_with_doc_id (tuple)
            A tuple of document ids and corresponding list of augmented queries.
        '''
        all_documents = []
        top_k_documents = []
        for doc_id, queries in aug_queries_with_doc_id:
            retrieved_docs = []
            all_ranks = []
            
            for query in queries:
                result = self.RAG.search(query=query, k=self.p, doc_ids=[doc_id])
                   
                for x in result:
                    retrieved_docs.append(x['content'])
                    all_ranks.append(x['rank'])  
            all_documents.append(retrieved_docs)  
            top_k_documents.append(self.reciprocal_rank_fusion(retrieved_docs, all_ranks))
        return top_k_documents, all_documents
    
    def reciprocal_rank_fusion(self, all_documents, all_ranks):
        '''
        Apply RRF to multiple lists of items with their ranks, and return top-k
        documents based on thecombined scores.
        
        Parameters:
        -----------
            all_documents: A list of lists, where each inner list represents a ranked list of documents ids.
            all_ranks: A list of lists, where each inner list represents the ranks of list of documents ids.
            return: A dictionary of document ids sorted by their combined RRF scores.
        '''
        # all_documents = [item for sublist in all_documents for item in sublist]
        rrf_scores = {}
        for item, rank in zip(all_documents, all_ranks):
            if item not in rrf_scores:
                rrf_scores[item] = 0
            rrf_scores[item] = rrf_scores[item]+ (1 / (self.k + rank))
       
        sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse=True)     
        return sorted_items[:self.k]
