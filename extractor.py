from ragatouille import RAGPretrainedModel
import requests
import time
from functools import partial
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


# from bs4 import BeautifulSoup

def get_html_page(line):
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
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

def fetch_docs_from_html(links_file_path='./CA_links.txt', index_path="./credit_agreement_database"):
    z=[]
    my_documents=[]
    with open(links_file_path, 'r') as file:
        for line in file:
            print(line)
            line = line.strip()  # Remove newline character
            document_content = get_html_page(line)
            if document_content:  # Add content to the list if it's not None
                my_documents.append(document_content)

    # file_path = './outputs_final.txt'
    # # Writing the array to a file
    # with open(file_path, 'w') as file:
    #     for item in my_documents:
    #         file.write("%s\n" % item)
    print("Number of indexed documents: ", len(my_documents))
    index_path = RAG.index(index_name=index_path, collection=my_documents)

def read_file_as_string(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content


def fetch_docs_from_text(dir_path, index_path="./credit_agreement_database"): 
  file_names = [f.split('.')[0] for f in listdir(dir_path)][:1000]
#   file_names = file_names[:10]
  doc_collection = []
#   print(file_names[:5])
  for i in tqdm(range(len(file_names))):
      file_content = read_file_as_string(dir_path + file_names[i] + '.txt')
      doc_collection.append(file_content)

  print("Number of read documents: ", len(file_names))
  RAG.index(index_name=index_path, collection=doc_collection, document_ids=file_names)
  print("Number of indexed documents: ", len(doc_collection))


if __name__ == "__main__":
    dir_path = '/work/pi_dhruveshpate_umass_edu/aneelakantes_umass_edu/SelfRewardingRAG/credit_agreements_txt/'
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    start_time = time.time()
    fetch_docs_from_text(dir_path, index_path="/work/pi_dhruveshpate_umass_edu/aneelakantes_umass_edu/new_index4")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
