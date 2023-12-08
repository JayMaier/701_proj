from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from trl import SFTTrainer, is_xpu_available
import time
import ipdb

model_name_or_path = "models/merged_adapters_4bit" #path/to/your/model/or/name/on/hub
base_model = "models/Llama-2-7b-hf"
device = "cuda" # or "cuda" if you have a GPU
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                        )

# quantization_config = BitsAndBytesConfig(
#         load_in_8bit=True, load_in_4bit=False)

device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
torch_dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=quantization_config, device_map=device_map, torch_dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(base_model)

inputs = tokenizer.encode("### English: This movie was really good. ### French: ", return_tensors="pt").to(device)

print('inputs: ', inputs)
start = time.time()
outputs = model.generate(inputs, max_new_tokens=10)
print('inference time: ', time.time() - start)
print(tokenizer.decode(outputs[0]))
ipdb.set_trace()