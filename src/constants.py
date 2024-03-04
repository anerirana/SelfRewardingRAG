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
- Clearly highlight which extracts from knowledge base were used to answer the question.

"""

REWARD_PROMPT = """Review the user's question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add second point if the response addresses a substantial portion of the user's question but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organizedand helpful,even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: {original_query} 
Response: {answer}

After examining the user's instruction and the response:

-Briefly justify your total score, upto 100 words.
-Conclude with the score using the format: 
“Score:<totalpoints>” 
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary.To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria. """