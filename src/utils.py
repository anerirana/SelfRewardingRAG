import os
import re

from sklearn.model_selection import train_test_split

def read_file_as_string(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

def doc_train_test_split(training_data_dir):
    training_data_dir = training_data_dir
    file_name_template = "{doc_id}_response.txt"
    file_names = os.listdir(training_data_dir)
    docs_with_sources = []
    docs_without_sources = []
    docs_with_error = []
    for file_name in file_names:
        file_content = read_file_as_string(training_data_dir + file_name)
        if re.search(r'error in calling llm', file_content, flags=re.I):
            docs_with_error.append(file_name)
        elif re.search(r'source_passages', file_content, flags=re.I) or re.search(r'\*\*Source Passage(s)?:\*\*', file_content, flags=re.I):
            docs_with_sources.append(file_name)
        else:
            docs_without_sources.append(file_name)

    print("Number of documents with source passages = ", len(docs_with_sources))
    print("Number of documents without source passages = ", len(docs_without_sources))
    print("Number of documents with error = ", len(docs_with_error))
    print(docs_without_sources)


    train_docs, test_docs = train_test_split(docs_with_sources, train_size=299, random_state=41, shuffle=True)
    print("len of train_docs: ", len(train_docs))
    test_docs += docs_without_sources
    print("len of test_docs: ", len(test_docs))

    for file_name in train_docs:    
        os.system("cp " + training_data_dir + " " + training_data_dir + "train_docs/")

    for file_name in test_docs:    
        os.system("cp " + training_data_dir + " " + training_data_dir + "test_docs/")
