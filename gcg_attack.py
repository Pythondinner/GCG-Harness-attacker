import time
import json
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import nanogcg
from nanogcg import GCGConfig

def run_gcg_attack(model_name,prompt,target,attack_params,return_model=False,
                   tokenizer=None,model=None)

    if tokenizer is None or model is None:
        print(f'Loading model----')
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code = True,
            padding_side = "left"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.float16,
            trust_remote_code = True,
            device_map = "auto"
        )

    else:
        print(f'model Loaded----')


    config = GCGConfig(
        num_steps = attack_params["num_steps"],
        search_width = attack_params["search_width"],
        batch_size = attack_params["batch_size"],
        topk = attack_params["topk"],
        n_replace = attack_params["n_replace"],
        seed = attack_params["seed"]
    )

    messages = [{"role":"user","content":prompt}]

    start = time.time()
    result =nanogcg.run(
        model=model,
        tokenizer = tokenizer,
        messages = messages,
        target =target,
        config=config
    )

    elapsed = time.time() - start

    result_dict = {
        "config": {
            "model": model_name,
            "prompt": prompt,
            "target": target,
            "num_steps": attack_params["num_steps"],
            "search_width": attack_params["search_width"],
            "batch_size": attack_params["batch_size"],
            "seed": attack_params["seed"],
        },
        "best_string": result.best_string,
        "best_loss": result.best_loss,
        "losses": result.losses,
        "time_seconds": elapsed,
    }
    if return_model:
        return result_dict, tokenizer, model
    return result_dict

def generate_response(text,tokenizer,model):
    messages = [{"role":"user","content":text}]
    formatted = tokenizer.apply_chat_template(messages,tokenizer=False,add_generation_prompt=True)
    inputs = tokenizer([formatted],return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 150,
            do_sample = True,
            temperature = 0.7,
            top_p = 0.9,
            min_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return response