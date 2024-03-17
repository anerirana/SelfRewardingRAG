import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModelForCausalLM
from trl import DPOTrainer
# from unsloth import FastLanguageModel

class LLM(nn.Module):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        super(LLM, self).__init__()
        '''Retrival Augmented Generation model to generate responses for a given query 
        and a set of retieved documents
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                    trust_remote_code=True,
                    device_map="auto")                        
        self.model = torch.compile(self.model, mode = "max-autotune", backend="inductor")
    
    def forward(self, prompt, param_dict=None):
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
        
        if param_dict is None:
            outputs = self.model.generate(inputs, temperature=0.7, top_k=0, top_p=1, repetition_penalty=1.4, min_new_tokens=16, max_new_tokens=2048, do_sample=True)
        else:
            #TODO: try fixing temperature=0.7
            outputs = self.model.generate(inputs, temperature=param_dict["temperature"], top_p=param_dict["top_p"], repetition_penalty=param_dict["repetition_penalty"], min_new_tokens=param_dict["min_new_tokens"], max_new_tokens=param_dict["max_new_tokens"], do_sample=True)
        
        return self.tokenizer.decode(outputs[0])

    def train(self, training_dataset, batch_size=32, num_epochs=3):
        dpo_trainer = DPOTrainer(
            model,
            model_ref=None,
            args=training_args,
            beta=0.1,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
        dpo_trainer.train()
        '''Train the RAGModel on the given training_dataset
        Parameters:
        -----------
        training_dataset
            Contains the training data for query augemntation or reward generation or answer generator
        '''

        pass