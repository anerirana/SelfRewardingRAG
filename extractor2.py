from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page
import requests
import time
from functools import partial


from bs4 import BeautifulSoup

def get_wikipedia_page(line):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    url = line

    headers = {
      
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15'
    }
    params = {
        "action": "query",
        "format": "json",
        "titles": "samveg",
        "prop": "extracts",
        "explaintext": True,
    }


    response = requests.get(url,params=params,headers=headers)
    print(response, "REASPONSE")
    if response.status_code == 200:
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extracting data (modify this based on the data you need)
        # For example, extracting all text:
        text = soup.get_text(separator='\n')
        # with open('/Users/samvegshah/Desktop/output2.txt', 'w') as output_file:
        #     output_file.write(text)

        # # Return text or convert to JSON as per requirement
        return text
    print(response.text[:500])
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

# def get_wikipedia_page(title: str):
#     with open('/Users/samvegshah/Downloads/Exhibit.html', 'r', encoding='utf-8') as file:
#         html_content = file.read()

#     # Parse the HTML content using BeautifulSoup
#     soup = BeautifulSoup(html_content, 'html.parser')

#     # Extract text from the parsed HTML (you can modify this to extract specific parts)
#     text_content = soup.get_text()

#     # Save the extracted text to a file
#     with open('/Users/samvegshah/Desktop/output.txt', 'w') as output_file:
#         output_file.write(text_content)

#     return text_content

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
start_time = time.time()
z=[]
my_documents=[]
with open('/home/samvegvipuls_umass_edu/CA_links.txt', 'r') as file:
    for line in file:
        print(line)
        line = line.strip()  # Remove newline character
        document_content = get_wikipedia_page(line)
        if document_content:  # Add content to the list if it's not None
            my_documents.append(document_content)

file_path = '/home/samvegvipuls_umass_edu/outputs_final.txt'

# Writing the array to a file
with open(file_path, 'w') as file:
    for item in my_documents:
        file.write("%s\n" % item)
index_path = RAG.index(index_name="/home/samvegvipuls_umass_edu/test_indexing3", collection=my_documents)

end_time = time.time()
excution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")