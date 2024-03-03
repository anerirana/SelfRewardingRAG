QUERY_AUGMENTATION_PROMPT = "Give {n} different versions of the query: {original_query}"
RAG_PROMPT = "{original_query} only using information from the relevant extracts: {documents}. Clearly highlight which extracts were used in the answer"
REWARD_PROMPT = "For this question: {original_query}, give a score out of 5 to this answer: {answer}."