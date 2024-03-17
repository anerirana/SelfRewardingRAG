
import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModelForCausalLM
from trl import DPOTrainer
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import PatchDPOTrainer

from datasets import Dataset
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
            outputs = self.model.generate(inputs, temperature=0.7, top_p=0.99, repetition_penalty=1.2, min_new_tokens=16, max_new_tokens=2048, do_sample=True)
        else:
            #TODO: try fixing temperature=0.7
            outputs = self.model.generate(inputs, temperature=param_dict["temperature"], top_p=param_dict["top_p"], repetition_penalty=param_dict["repetition_penalty"], min_new_tokens=param_dict["min_new_tokens"], max_new_tokens=param_dict["max_new_tokens"], do_sample=True)
        
        return self.tokenizer.decode(outputs[0])

    def train(self, training_dataset, batch_size=32, num_epochs=3):
        # print(training_dataset)
        # url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
        # dataset = load_dataset("json", data_files="my_file.json")
        # dataset = load_dataset("json", data_files = {"train" : url}, split = "train")
        print(training_dataset)
        dataset = Dataset.from_dict(training_dataset)
        # dataset=load_dataset(training_dataset)


        
        max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
# Get LAION dataset
        # 4bit pre quantized models we support - 4x faster downloading!

        # Load Llama model
        # model, tokenizer = FastLanguageModel.from_pretrained(
        #     model_name = "unsloth/mistral-7b-bnb-4bit", # Supports Llama, Mistral - replace this!
        #     max_seq_length = max_seq_length,
        #     dtype = None,
        #     load_in_4bit = True,
        # )

        # Do model patching and add fast LoRA weights
        # model = FastLanguageModel.get_peft_model(
        #     self.model,
        #     r = 16,
        #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
        #                     "gate_proj", "up_proj", "down_proj",],
        #     lora_alpha = 16,
        #     lora_dropout = 0, # Supports any, but = 0 is optimized
        #     bias = "none",    # Supports any, but = "none" is optimized
        #     use_gradient_checkpointing = True,
        #     random_state = 3407,
        #     max_seq_length = max_seq_length,
        #     use_rslora = False,  # We support rank stabilized LoRA
        #     loftq_config = None, # And LoftQ
        # )

        training_args = TrainingArguments(output_dir="/home/samvegvipuls_umass_edu/src/dataset/dummy_index5")

        max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/mistral-7b-bnb-4bit",
                max_seq_length = max_seq_length,
                dtype = None,
                load_in_4bit = True,
            )

        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 64,
            lora_dropout = 0, # Currently only supports dropout = 0
            bias = "none",    # Currently only supports bias = "none"
            use_gradient_checkpointing = True,
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        PatchDPOTrainer()

        dpo_trainer = DPOTrainer(
            model = model,
            ref_model = None,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_ratio = 0.1,
                num_train_epochs = 3,
                learning_rate = 5e-6,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.0,
                lr_scheduler_type = "linear",
                seed = 42,
                output_dir = "outputs",
            ),
            beta = 0.1,
            train_dataset = dataset,
            # eval_dataset = raw_datasets["test"],
            tokenizer = tokenizer,
            max_length = 1024,
            max_prompt_length = 512,
        )
        print("HELOOOOOOOO")
        dpo_trainer.train()
