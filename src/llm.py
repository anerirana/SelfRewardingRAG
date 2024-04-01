
import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer,unsloth_save_model
from datasets import Dataset

class LLM(nn.Module):
    def __init__(self, model_name):
        super(LLM, self).__init__()
        '''Retrival Augmented Generation model to generate responses for a given query 
        and a set of retieved documents
        '''
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name,
        #             trust_remote_code=True,
        #             device_map="auto")                        
        # self.model = torch.compile(self.model, mode = "max-autotune", backend="inductor")

        max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = None,
                load_in_4bit = True,
            )

        # Do model patching and add fast LoRA weights
        self.model = FastLanguageModel.get_peft_model(
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
            outputs = self.model.generate(inputs, pad_token_id=self.tokenizer.eos_token_id, temperature=0.7, top_p=0.99, repetition_penalty=1.2, min_new_tokens=16, max_new_tokens=2048, do_sample=True)
        else:
            outputs = self.model.generate(inputs, pad_token_id=self.tokenizer.eos_token_id, temperature=param_dict["temperature"], top_p=param_dict["top_p"], repetition_penalty=param_dict["repetition_penalty"], min_new_tokens=param_dict["min_new_tokens"], max_new_tokens=param_dict["max_new_tokens"], do_sample=True)
        
        return self.tokenizer.decode(outputs[0])

    def train(self, epoch, training_dataset, batch_size=32, num_epochs=3):
        dataset = Dataset.from_dict(training_dataset)

        PatchDPOTrainer()
        dpo_trainer = DPOTrainer(
            model = self.model,
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
                output_dir = "output/training_arguments",
            ),
            beta = 0.1,
            train_dataset = dataset,
            # eval_dataset = raw_datasets["test"],
            tokenizer = self.tokenizer,
            max_length = 1024,
            max_prompt_length = 512,
        )
        print(">>"*40 + " BEGINING TRAINING " + ">>"*40)
        dpo_trainer.train()
        unsloth_save_model(self.model, self.tokenizer, "output/model_epoch_" + str(epoch), push_to_hub=False, token=None)
        print(">>"*40 + " END TRAINING " + ">>"*40)

