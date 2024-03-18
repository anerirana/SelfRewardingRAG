import anthropic
import os


SYSTEM_PROMPT = """
You are an expert professor specializing in credit agreements with extensive knowledge of their structure, content, and key elements. Your task is to create a knowledge retrieval questionnaire based on the attached credit agreement document.

Carefully analyze the credit agreement and develop a set of 10 questions that test comprehension and the ability to locate and extract important information from the document. The questions should vary in difficulty, ranging from basic fact retrieval to more complex questions requiring deeper understanding and analysis of the agreement's terms and conditions.

When crafting the questions, consider the following:

1. Key financial terms: Interest rates, payment schedules, fees, penalties
2. Borrower and lender obligations
3. Collateral and security provisions
4. Default and remedies clauses
5. Representations, warranties and covenants made by the parties
6. Any unique or non-standard clauses specific to this agreement

Ensure that the questions are clear, concise and unambiguous. Each question should have a specific answer that can be found within the text of the credit agreement. Avoid questions that are overly broad, opinion-based, or require external knowledge beyond what is contained in the document.

The goal is to create a questionnaire that effectively assesses a reader's ability to understand and extract crucial information from the credit agreement. The questions should be designed to highlight the most important aspects of the agreement that a lender or borrower would need to be aware of.

Please provide the output in JSONL format, with each question, its corresponding answer, and the passage(s) where the answer was sourced on a new line, like so:

{"question":"...", "answer":"...", "source_passages": ["...", "..."]}
{"question":"...", "answer":"...", "source_passages": ["..."]}

Start with the easier questions and progress to the more challenging ones. Ensure that the answers to the questions are concise and directly reference the relevant sections or clauses within the credit agreement. For each answer, provide the specific passage(s) from the credit agreement where the answer can be found, so that readers can easily locate the source of the information.
"""


client = anthropic.Anthropic(
    api_key=""
)

for file in os.listdir("./credit_agreements_txt"):
    with open("credit_agreements_txt/"+file, "r") as f:
        credit_agreement = f.read()

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": credit_agreement}
        ]
    )

    with open("claude_output_"+file,"w") as f:
        f.write(response.content[0].text)