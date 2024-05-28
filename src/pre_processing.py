from constants import *
import jsonlines
import json
import os

def create_text_row(instruction, output):
    text_row = f"""<s>[INST]{RAG_PROMPT.format(knowledge_base = "", original_query = instruction)}[/INST] {output} </s>"""
    return text_row

def process_jsonl_file(output_file_path):
    with open(output_file_path, "w") as output_jsonl_file:
        for file in os.listdir("./training_dataset"):
            with jsonlines.open("./training_dataset/"+file) as f:
                for item in f.iter():
                    json_object = {
                        "text": create_text_row(item["question"], item["answer"]),
                        "question": item["question"],
                        "answer": item["answer"]
                    }
                    output_jsonl_file.write(json.dumps(json_object) + "\n")

process_jsonl_file("./training_dataset.jsonl")