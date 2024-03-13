import random

QUERY_AUGMENTATION_PROMPT = """Generate {n} different versions of the query: {original_query}. 

- Each version of the query should be distinct and relevant to the original query.
- Itemize each query with a number and a period (e.g. "1. ").
"""

RAG_PROMPT = """You are a financial document expert. 

Question : \"{original_query}\"

Knowledge Base: {documents}. 

#Rules to answer:
- Provide a concise response to the user's question.
- The response should be in bullet points. 
- Last bullet point has to be "Extracts used: [extract numbers]"

Remember, both the response and the extracts used are important.
"""

REWARD_PROMPT = """Review the user's question and the corresponding response using the 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Award 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Award 2 points if the response addresses a substantial portion of the user's question but does not completely resolve the query or provide a direct answer.
- Award 3 points if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Award 4 points if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organizedand helpful,even if there is slight room for improvement in clarity, conciseness or focus.
- Award 5 points for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: {original_query} 
Response: {answer}

After examining the user's instruction and the response:

-Briefly justify your total score.
-Conclude with the score using the format: 
“Score:<Points awarded> out of 5”

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. """

SAMPLING_PARAMS_DICT = {
            "temperature":random.choice([0.1,0.2,0.3,0.4]), 
            "top_p":random.choice([0.99, 0.8, 0.7, 0.6, 0.5]), 
            "repetition_penalty":random.choice([1.2, 1.3, 1.4, 1.5]), 
            "min_new_tokens":random.choice([16, 32, 64, 128]), 
            "max_new_tokens":random.choice([2016, 2032, 2064, 2128])
          }

EXTRACT_CITATION_PROMPT = """

User Question: {original_query}

Knowledge Base: {extracts}

Answer Generated: {answer}

List all the extracts from the knowledge base that were used in the answer.
"""
