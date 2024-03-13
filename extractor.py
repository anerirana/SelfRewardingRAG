from ragatouille import RAGPretrainedModel
import time
from os import listdir
from tqdm import tqdm


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
