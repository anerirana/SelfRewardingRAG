import random
TRANSFORMERS_TOKEN = ""
OUTPUT_DIRECTORY = "/home/arana_umass_edu/SelfRewardingRAG/output/"
TRAIN_OUTPUT_DIRECTORY = "/scratch/workspace/arana_umass_edu-goldamn_project/training_output"
PATH_TO_INDEX = "/scratch/workspace/arana_umass_edu-goldamn_project/400_docs_index"
QUERY_AUGMENTATION_PROMPT = """Generate {n} different versions of the query: {original_query}. 

- Each version of the query should be distinct and relevant to the original query.
- Itemize each query with a number and a period (e.g. "1. ").
"""

RAG_PROMPT = """
Answer the folowing question using the knowledge base provided.  

#Rules to answer:
- Provide a concise, well-formatted response to the user's question.
- The response should be in bullet points if needed.

Knowledge Base: 
\"{knowledge_base}\"

Question : \"{original_query}\"
"""

RAG_CITATION_PROMPT = """Answer the question using the following rules and knowledge base provided. 

Rules:
- The answer should comprise of two sections - Answer and Sources used.
- In the answer section, provide a concise, well-formatted response to the user's question.
- In the sources used section, justify which sources from the knowledge base were used in generating the answers.

Knowledge Base: 
\"{knowledge_base}\"

Question : \"{original_query}\"
"""

REWARD_PROMPT = """Review the user's question with the provided knowledge base and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Give 0 points if the response is completely irrelevant.
- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question by combining several pieces of information from the knowledge base, although the answer might not be complete.
- Grant a fourth point if the response is clearly written and addresses the user's question directly. However, there is room for improvement by using additional contextual information from the knowledge base.
- Bestow a fifth point for a response that contains the key information requested in the user's question. The response is clear, comprehensive, and summarizes all relevant pieces of the information provided in the knowledge base.

User: {original_query}
Response: {answer}

After examining the user's instruction and the response:

-Briefly criticize your total score.
-Note that the answer is allowed to combine several pieces of information even though the answer is not directly quoted from the knowledge base.
-Conclude with the score using the format: “Score: [0-5] out of 5”

Remember to assess from a financial AI Assistant perspective. """

EXTRACT_CITATION_PROMPT = """

User Question: {original_query}

Knowledge Base: {extracts}

Answer Generated: {answer}

List all the extracts from the knowledge base that were used in the answer.
"""
