import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer,unsloth_save_model
from datasets import Dataset
import transformers

class LLM(nn.Module):
    def __init__(self, model_name):
        super(LLM, self).__init__()
        '''Retrival Augmented Generation model to generate responses for a given query 
        and a set of retieved documents
        '''
        self.max_seq_length = 4096
        self.max_new_tokens = 4096
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name,
        #             trust_remote_code=True,
        #             device_map="auto")                        
        # self.model = torch.compile(self.model, mode = "max-autotune", backend="inductor")

        max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        self.model = transformers.AutoModelForCausalLM.from_pretrained("Nexusflow/Starling-LM-7B-beta")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("Nexusflow/Starling-LM-7B-beta")




        # Do model patching and add fast LoRA weights
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(device)
        
        # Ensure the total length does not exceed max_seq_length
        print("input_ids.shape[1]:",input_ids.shape[1])
        max_length = max(self.max_seq_length, input_ids.shape[1]+1)
        print("max_length:", max_length)

        output = self.model.generate(input_ids=input_ids, max_length=max_length)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

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
