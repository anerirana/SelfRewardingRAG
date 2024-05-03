
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer, DPOTrainer
from peft import LoraConfig, PeftModel
# from unsloth import FastLanguageModel, PatchDPOTrainer,unsloth_save_model
from datasets import Dataset, load_dataset
from constants import *
from training_args import *

class LLM(nn.Module):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        super(LLM, self).__init__()
        '''Retrival Augmented Generation model to generate responses for a given query 
        and a set of retieved documents
        '''
        #LoRA parameters
        # LoRA attention dimension
        self.lora_r = 64

        # Alpha parameter for LoRA scaling
        self.lora_alpha = 16

        # Dropout probability for LoRA layers
        self.lora_dropout = 0

        #bitsandbytes parameters
        # Activate 4-bit precision base model loading
        use_4bit = True

        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"

        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"

        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False

        # Load the base model with QLoRA configuration
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto", #get_kbit_device_map(),
            token=TRANSFORMERS_TOKEN
        )                        

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load MitsralAi tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=TRANSFORMERS_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
        # dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        # load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        # model, self.tokenizer = FastLanguageModel.from_pretrained(
        #         model_name = model_name,
        #         max_seq_length = max_seq_length,
        #         dtype = None,
        #         load_in_4bit = True,
        #     )

        # # Do model patching and add fast LoRA weights
        # self.model = FastLanguageModel.get_peft_model(
        #     model,
        #     r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
        #                     "gate_proj", "up_proj", "down_proj",],
        #     lora_alpha = 64,
        #     lora_dropout = 0, # Currently only supports dropout = 0
        #     bias = "none",    # Currently only supports bias = "none"
        #     use_gradient_checkpointing = True,
        #     random_state = 3407,
        #     use_rslora = False,  # We support rank stabilized LoRA
        #     loftq_config = None, # And LoftQ
        # )
    
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

    def train(self, epoch, training_dataset=None, batch_size=32, num_epochs=3):
        dataset = load_dataset('json', data_files='output/dpo_preference_pairs_0.json', split="train")

        # Set LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Set training parameters
        training_arguments = TrainingArguments(
            output_dir=OUTPUT_DIRECTORY,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            # bf16=bf16,  
            max_grad_norm=max_grad_norm,
            max_steps=100, # the total number of training steps to perform
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type
        )

        # Initialize the SFTTrainer for fine-tuning
        # trainer = SFTTrainer(
        #     model=self.model,
        #     train_dataset=dataset,
        #     peft_config=peft_config,
        #     dataset_text_field="text",
        #     max_seq_length=max_seq_length,  # You can specify the maximum sequence length here
        #     tokenizer=self.tokenizer,
        #     args=training_arguments,
        #     packing=packing
        # )

        # trainer.train()
        # dataset = Dataset.from_dict(training_dataset)

        # PatchDPOTrainer()
        dpo_trainer = DPOTrainer(
            model = self.model,
            ref_model = None,
            peft_config=peft_config,
            args = training_arguments,
            beta = 0.1,
            train_dataset = dataset,
            # eval_dataset = raw_datasets["test"],
            tokenizer = self.tokenizer,
            # max_length = None,
            # max_prompt_length = None,
        )
        print(">>"*40 + " BEGINING TRAINING " + ">>"*40)
        dpo_trainer.train()
        # unsloth_save_model(self.model, self.tokenizer, OUTPUT_DIRECTORY + "model_epoch_" + str(epoch), push_to_hub=False, token=None)
        dpo_trainer.save_model()
        print(">>"*40 + " END TRAINING " + ">>"*40)

