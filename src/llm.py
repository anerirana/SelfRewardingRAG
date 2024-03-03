import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModelForCausalLM

class LLM(nn.Module):
    def __init__(self, model_name):
        super(LLM, self).__init__()
        '''Retrival Augmented Generation model to generate responses for a given query 
        and a set of retieved documents
        '''
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                    trust_remote_code=True,
                    device_map="auto")                        
        self.model = torch.compile(self.model, mode = "max-autotune", backend="inductor")
    
    def forward(self, prompt, temp=0.5):
        '''Generate response for the given prompt

        Parameters:
        -----------
        prompt
            The prompt used as input to the generative model.
            Could be a RAG or QA or reward prompt.
        '''
        messages = [{"role":"user", "content":prompt}]
        encoded = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        inputs = encoded.to(self.model.device)     
        
        outputs = self.model.generate(inputs, temperature=temp, top_k=0, top_p=1.0, repetition_penalty=1.4, min_new_tokens=400, max_new_tokens=4096, do_sample=True)
        
        return self.tokenizer.decode(outputs[0])

    def train(self, training_dataset, batch_size=32, num_epochs=3):
        '''Train the RAGModel on the given training_dataset

        Parameters:
        -----------
        training_dataset
            Contains the training data for query augemntation or reward generation or answer generator
        '''
        
        pass