import gcg_attack

class LocalExecutor:
    def execute(self,config:dict,tokenizer,model):
        model_name = config["model_name"]
        prompt = config["prompt"]
        target = config["target"]
        attack_params = config["attack"]

        try:
            result_dict = gcg_attack.run_gcg_attack(
                model_name,prompt,target,attack_params,
                return_model = False,
                tokenizer=  tokenizer,
                model = model
            )

            adversarial_prompt = f"{prompt} {result_dict['best_string']}"

            generated_response = gcg_attack.generate_response(
                adversarial_prompt , tokenizer , model
            )

            return {
                "best_loss": result_dict["best_loss"],
                "best_string": result_dict["best_string"],
                "time_seconds": result_dict["time_seconds"],
                "is_oom": False,
                "generated_response": generated_response,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"executor.py异常：{error_msg}")

            is_oom = "CUDA out of memory" in error_msg or "OOM" in error_msg

            return {
                "best_loss": "999999.0",
                "best_string": "",
                "time_seconds": 0,
                "is_oom": is_oom,
                "generated_response": f"[ERROR] {error_msg}",
            }