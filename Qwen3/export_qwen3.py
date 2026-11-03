from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import export as exportsd

model_path = 'G:\\model\\Qwen3-4B-Instruct-2507'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                             device_map="cpu")

f = open("Qwen3-4B-Instruct-2507-fp16.dat", "wb")
exportsd.save_state_dict(model.to("cpu").state_dict(), f)
f.close()