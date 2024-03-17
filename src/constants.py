import random

QUERY_AUGMENTATION_PROMPT = """Generate {n} different versions of the query: {original_query}. 

- Each version of the query should be distinct and relevant to the original query.
- Itemize each query with a number and a period (e.g. "1. ").
"""

RAG_PROMPT = """Answer the folowing question using the knowledge base provided. 

Question : \"{original_query}\"

Knowledge Base: {documents}. 

#Rules to answer:
- Provide a concise response to the user's question.
- Justify which extract was used in generating the answers.
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

- Briefly justify your total score.
- Conclude with the score using the format: 

“Score:<Points awarded> out of 5”

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. """

REWARD_PROMPT_ALTERNATE = """Review the user's question and the corresponding response using the 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

User: {original_query} 
Response: {answer}
Scoring System: The score should range from 0 to 5 based on how well the response answers the user's query. 

- Briefly justify your total score.
- Conclude with the score using the format: 

“Score:<Points awarded> out of 5”     """

SAMPLING_PARAMS_DICT = {
            "temperature":random.choice([0.1,0.2,0.3,0.4]), 
            "top_p":random.choice([0.99, 0.8, 0.7, 0.6, 0.5]), 
            "repetition_penalty":random.choice([1.2, 1.3, 1.4, 1.5]), 
            "min_new_tokens":random.choice([16, 32, 64, 128]), 
            "max_new_tokens":random.choice([2016, 2032, 2064, 2128])
          }

EXTRACT_CITATION_PROMPT = """

List all the extracts from the knowledge base that were used to answer the user's question.

User Question: {original_query}

Knowledge Base: {documents}

Answer Generated: {answer}

Examples: 
1. User question: On what date will the commitments under this credit agreement terminate?

Knowledge Base: "extract_1":"(d) Unless the Applicable Agent shall have received notice from the\nCompany prior to the date on which any payment is due to the Administrative\nAgent for the account of the Lenders or an Issuing Bank hereunder that the\nCompany and the Borrowers will not make such payment, the Applicable Agent may\nassume that the Company or a Borrower has made such payment on such date in\naccordance herewith and may, in reliance upon such assumption, distribute to the\nLenders or such Issuing Bank, as the case may be, the amount due.', 'SECTION 2.18. PAYMENTS GENERALLY; PRO RATA TREATMENT; SHARING OF\nSET-OFFS. (a) The Company and the Borrowers shall make each payment required to\nbe made by them hereunder or under any other Loan Document (whether of\nprincipal, interest, fees or reimbursement of LC Disbursements, or of amounts\npayable under Section 2.15, 2.16 or 2.17, or otherwise) prior to the time\nexpressly required hereunder or under such other Loan Document for such\npayment (or, if no such time is expressly required, prior to 12:00 noon,\nLocal Time), on the date when due, in immediately available funds, without\nset-off or counterclaim.  Any amounts received after such time on any date\nmay, in the\n\n\n                                       49\n<PAGE>\n\ndiscretion of the Administrative Agent, be deemed to have been received on\nthe next succeeding Business Day for purposes of calculating interest thereon.","extract_2":""MATURITY DATE" means the fifth anniversary of the Effective Date.\n\n           "MOODY\'S" means Moody\'s Investors Service, Inc.\n\n           "MULTIEMPLOYER PLAN" means a multiemployer plan as defined in\nSection 4001(a)(3) of ERISA.\n\n           "OBLIGATIONS" has the meaning assigned to such term in the\nGuarantee Agreement.\n\n           "OECD" means the Organization for Economic Cooperation and\nDevelopment.\n\n           "OTHER TAXES" means any and all present or future  stamp or\ndocumentary taxes or any other excise or property  taxes, charges or similar\nlevies arising from any payment made under any Loan Document or from the\nexecution, delivery or enforcement of, or otherwise with respect to, any Loan\nDocument.\n\n           "PARTICIPANT" has the meaning set forth in Section 9.04(c)(i).\n\n           "PBGC" means the Pension Benefit Guaranty Corporation referred to\nand defined in ERISA and any successor entity performing similar functions.\n\n           "PERMITTED ENCUMBRANCES" means:\n\n          (a) Liens imposed by law for taxes that are not yet due or are being\n     contested in compliance with Section 5.04;","extract_3":"(c) EXPIRATION DATE.  Each Letter of Credit shall expire at or prior\nto the close of business on the earlier of (i) the date one year after the date\nof the issuance of such Letter of Credit (or, in the case of any renewal or\nextension thereof, one year after such renewal or extension) and (ii) the\ndate that is five Business Days prior to the Maturity Date.\n\n          (d)   PARTICIPATIONS.  By the issuance of a Letter of Credit (or an\namendment to a Letter of Credit increasing the amount thereof) and without any\nfurther action on the part of the applicable Issuing Bank or the Lenders, such\nIssuing Bank hereby grants to each US Tranche Lender, and each such Lender\nhereby acquires from such Issuing Bank, a participation in such Letter of\nCredit equal to such Lender's US Tranche Percentage of the aggregate amount\navailable to be drawn under such Letter of Credit."

Answer Generated: The commitments will terminate on the Maturity Date, which is defined as the fifth anniversary of the Effective Date.


Extracts used to generate answer: extract_2, extract_3
"""
