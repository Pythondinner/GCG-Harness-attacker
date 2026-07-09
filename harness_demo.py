
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import nanogcg
from nanogcg import GCGConfig

def run_gcg_attack(model_name, prompt, target, attack_params, return_model=False,
                   tokenizer=None, model=None):
    if tokenizer is None or model is None:
        print("Loading model----")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        print('Model Loaded----')

    config = GCGConfig(
        num_steps=attack_params["num_steps"],
        search_width=attack_params["search_width"],
        batch_size=attack_params["batch_size"],
        topk=attack_params["topk"],
        n_replace=attack_params["n_replace"],
        seed=attack_params["seed"]
    )

    messages = [{"role": "user", "content": prompt}]

    start = time.time()
    result = nanogcg.run(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        target=target,
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

def generate_response(text, tokenizer, model):
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            min_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return response


# ==================== 模块1：executor ====================
class LocalExecutor:
    def execute(self, config: dict, tokenizer, model):
        model_name = config["model_name"]
        prompt = config["prompt"]
        target = config["target"]
        attack_params = config["attack"]

        try:
            result_dict = run_gcg_attack(
                model_name, prompt, target, attack_params,
                return_model=False,
                tokenizer=tokenizer,
                model=model
            )

            adversarial_prompt = f"{prompt} {result_dict['best_string']}"
            generated_response = generate_response(adversarial_prompt, tokenizer, model)

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
                "best_loss": 999999.0,
                "best_string": "",
                "time_seconds": 0,
                "is_oom": is_oom,
                "generated_response": f"[ERROR] {error_msg}",
            }


# ==================== 模块2：ledger ====================
import copy

class MemoryLedger:
    def __init__(self, init_config: dict):
        self.config = copy.deepcopy(init_config)
        self.round = 0
        self.loss_trend = []
        self.best_loss_global = 999.0
        self.consecutive_ooms = 0

    def get_config(self) -> dict:
        return copy.deepcopy(self.config)

    def rewrite_parameters(self, audited_args: dict) -> dict:
        new_config = copy.deepcopy(self.config)
        if "attack" not in new_config:
            new_config["attack"] = {}

        for key, value in audited_args.items():
            if isinstance(value, int) and value < 1:
                cleaned_value = 1
            else:
                cleaned_value = value
            new_config["attack"][key] = cleaned_value

        self.config = new_config
        return new_config

    def update_history(self, current_loss: float, is_oom: bool):
        self.round += 1
        self.loss_trend.append(current_loss)
        if current_loss < self.best_loss_global:
            self.best_loss_global = current_loss
        if is_oom:
            self.consecutive_ooms += 1
        else:
            self.consecutive_ooms = 0

    def get_history_snapshot(self) -> dict:
        return {
            "round": self.round,
            "loss_trend": self.loss_trend.copy(),
            "best_loss_global": self.best_loss_global,
            "consecutive_ooms": self.consecutive_ooms
        }


# ==================== 模块3：observer ====================
def calculate_fluctuation(loss_history: list) -> str:
    if len(loss_history) < 3:
        return "insufficient"
    recent = loss_history[-3:]
    range_value = max(recent) - min(recent)
    if range_value > 0.5:
        return "high"
    elif range_value > 0.1:
        return "medium"
    else:
        return "low"

class ResultObserver:
    @staticmethod
    def analyze(current_report: dict, history: dict) -> dict:
        current_loss = current_report["best_loss"]
        is_oom = current_report["is_oom"]
        time_seconds = current_report["time_seconds"]

        loss_history = history.get("loss_trend", [])
        full_history = loss_history + [current_loss]
        fluctuation = calculate_fluctuation(full_history)

        return {
            "current_loss": current_loss,
            "time_seconds": time_seconds,
            "is_oom": is_oom,
            "fluctuation": fluctuation,
            "loss_history": full_history
        }


# ==================== 模块4：brain ====================
import random
import os
from openai import OpenAI

class DecisionBrain:
    def __init__(self, max_rounds=5, target_loss=0.05):
        self.max_rounds = max_rounds
        self.target_loss = target_loss

    def decide(self, observation: dict, history: dict) -> dict:
        current_round = history.get("round", 0)
        current_loss = observation["current_loss"]
        is_oom = observation["is_oom"]
        fluctuation = observation["fluctuation"]

        if current_loss <= self.target_loss:
            return {"action": "STOP_WIN", "reason": f"Loss达标{current_loss:.4f}"}

        if current_round >= self.max_rounds:
            return {"action": "STOP_SAFE", "reason": "到达最大轮次"}

        if history.get("consecutive_ooms", 0) >= 3:
            return {"action": "STOP_SAFE", "reason": "Out of Memory"}

        if fluctuation == "high":
            return {
                "action": "PERTURB",
                "reason": "高波动，开启强制扰动",
                "forced_params": {"num_steps": random.randint(50, 300)}
            }

        prompt = self._build_prompt(observation, history)
        try:
            new_params = self._call_llm(prompt)
            return {"action": "CONTINUE", "reason": "常规调参成功", "new_params": new_params}
        except RuntimeError as e:
            return {"action": "STOP_LLM_FAILURE", "reason": str(e)}

    def _build_prompt(self, observation: dict, history: dict) -> str:
        loss = observation["current_loss"]
        fluct = observation["fluctuation"]
        round_num = history.get("round", 0)
        return f"第 {round_num} 轮攻击结束。当前 Best Loss 为 {loss:.4f}，波动状态为 {fluct}。请调整攻击参数（num_steps, search_width, batch_size, seed）以降低 Loss。"

    def _call_llm(self, prompt: str, max_retries: int = 2) -> dict:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DeepSeek API Key，请设置环境变量 DEEPSEEK_API_KEY")

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        system_prompt = """
你是一个GCG攻击参数调优专家。请根据用户提供的攻击日志，输出下一轮攻击的优化参数。
你必须严格按照以下JSON格式输出，不要包含任何其他解释文字，保证输出内容的纯净：
{
    "num_steps": 整数 (10-300),
    "search_width": 整数 (10-512),
    "batch_size": 整数 (1-128),
    "seed": 整数 (1-9999)
}
"""
        current_prompt = prompt

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": current_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.5,
                    max_tokens=300
                )

                raw_content = response.choices[0].message.content
                params = json.loads(raw_content)

                is_valid, msg = SignalInspector.audit(params)
                if is_valid:
                    return params

                error_feedback = (
                    f"你返回的参数不符合要求，错误信息：{msg}。"
                    f"请根据以下范围重新生成："
                    f"num_steps(10-300), search_width(10-512), batch_size(1-128), seed(1-9999)。"
                    f"只返回JSON，保证JSON文本的纯净性，不要发任何其他无关的内容。"
                )
                current_prompt = f"{current_prompt}\n\n ERROR:{error_feedback}"
                print(f"第 {attempt + 1} 次参数核对失败，重试----")

            except json.JSONDecodeError:
                error_feedback = "你返回的内容不是有效的JSON格式。请只返回JSON对象。"
                current_prompt = f"{current_prompt}\n\n【错误反馈】{error_feedback}"
                print(f"第 {attempt + 1} 次JSON解析失败，正在重试...")

            except Exception as e:
                print(f"第 {attempt + 1} 次请求异常: {e}，重试----")
                continue

        raise RuntimeError("外部模型连续返回非法参数，已放弃")


# ==================== 插件1：contract ====================
class GCGContract:
    @staticmethod
    def get_schema() -> dict:
        return {
            "num_steps": {"type": "integer", "minimum": 10, "maximum": 300},
            "search_width": {"type": "integer", "minimum": 10, "maximum": 512},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 128},
            "seed": {"type": "integer", "minimum": 1, "maximum": 9999}
        }

    @staticmethod
    def get_required_fields() -> list:
        return ["num_steps", "search_width", "batch_size"]


# ==================== 插件2：inspector ====================
class SignalInspector:
    @staticmethod
    def get_schema() -> dict:
        return GCGContract.get_schema()

    @staticmethod
    def audit(raw_args: dict) -> tuple:
        schema = GCGContract.get_schema()
        required = GCGContract.get_required_fields()

        for key in required:
            if key not in raw_args:
                return False, f'缺少必要参数:{key}'
        for key, value in raw_args.items():
            if key not in schema:
                return False, f'参数不正确:{key}'
            rules = schema[key]
            if rules["type"] == "integer":
                if not isinstance(value, int):
                    return False, f"参数 {key} 应为整数，实际为 {type(value)}"
                if not (rules["minimum"] <= value <= rules["maximum"]):
                    return False, f"参数 {key} 值 {value} 超出范围 [{rules['minimum']}, {rules['maximum']}]"
        return True, "核验通过"


# ==================== 主程序 main ====================
import yaml
from datetime import datetime

print('GCG_Harness loading----')

with open("config.yaml", 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

ledger = MemoryLedger(config_data)

print('加载模型中----')
tokenizer = AutoTokenizer.from_pretrained(
    config_data["model_name"],
    trust_remote_code=True,
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    config_data["model_name"],
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto"
)
print("----模型加载完成")

executor = LocalExecutor()
observer = ResultObserver()
brain = DecisionBrain(max_rounds=5, target_loss=0.05)

audit_log_path = "audit_log.jsonl"
if os.path.exists(audit_log_path):
    os.remove(audit_log_path)
print(f'审计日志路径:{audit_log_path}')

def write_audit_log(entry: dict):
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

stop_reason = "未启动"
best_string = ""
generated_response = ""

max_rounds = 5
while ledger.round < max_rounds:
    current_round = ledger.round + 1
    print(f"\n{'=' * 40}")
    print(f"第 {current_round} 轮攻击开始")

    current_config = ledger.get_config()
    report = executor.execute(current_config, tokenizer, model)

    generated_response = report.get("generated_response", "")
    best_string = report.get("best_string", "")
    print(f'本轮结果：Loss={report["best_loss"]:.4f},耗时={report["time_seconds"]:.1f}秒')

    ledger.update_history(report["best_loss"], report["is_oom"])

    history_snapshot = ledger.get_history_snapshot()
    observation = observer.analyze(report, history_snapshot)
    print(f'分析日志:Loss={observation["current_loss"]:.4f},波动={observation["fluctuation"]}')

    decision = brain.decide(observation, history_snapshot)
    print(f'决策结果:{decision["reason"]}')

    write_audit_log({
        "round": current_round,
        "event": "decision",
        "action": decision['action'],
        "reason": decision['reason'],
        "current_loss": observation['current_loss']
    })

    if decision['action'] == "STOP_WIN":
        stop_reason = "攻击成功，Loss达标"
        print("达标！攻击成功！")
        write_audit_log({"round": current_round, "event": "stop_win", "final_loss": observation['current_loss']})
        break

    elif decision['action'] == "STOP_SAFE":
        stop_reason = decision['reason']
        print(f"安全熔断: {decision['reason']}")
        write_audit_log({"round": current_round, "event": "stop_safe", "reason": decision['reason']})
        break

    elif decision['action'] == "STOP_LLM_FAILURE":
        stop_reason = decision['reason']
        print(f"调用外部LLM失败: {decision['reason']}")
        write_audit_log({"round": current_round, "event": "stop_llm_failure", "reason": decision['reason']})
        break

    elif decision['action'] == "PERTURB":
        print(f"人工干预，执行扰动: {decision['forced_params']}")
        ledger.rewrite_parameters(decision['forced_params'])
        write_audit_log({
            "round": current_round,
            "event": "perturb",
            "forced_params": decision['forced_params']
        })
        print("继续下一轮...")
        continue

    elif decision['action'] == "CONTINUE":
        new_params = decision['new_params']
        is_valid, msg = SignalInspector.audit(new_params)
        if not is_valid:
            print(f"参数非法（Brain 校验遗漏）: {msg}")
            stop_reason = f"参数校验失败: {msg}"
            write_audit_log({
                "round": current_round,
                "event": "critical_inspector_fail",
                "invalid_params": new_params,
                "error": msg
            })
            print("强制熔断退出")
            break

        ledger.rewrite_parameters(new_params)
        print(f" 新参数已入库: {new_params}")
        write_audit_log({
            "round": current_round,
            "event": "params_updated",
            "new_params": new_params
        })
        print("继续下一轮...")
        continue

if ledger.round >= max_rounds and stop_reason == "未启动":
    stop_reason = "达到最大轮次但未达标或熔断"

print("循环结束。")
print(f"审计日志已保存至: {audit_log_path}")
print("最终账本状态：")
print(ledger.get_history_snapshot())

final_result = {
    "model_name": config_data["model_name"],
    "prompt": config_data["prompt"],
    "target": config_data["target"],
    "best_loss": ledger.best_loss_global,
    "best_string": best_string,
    "generated_response": generated_response,
    "rounds": ledger.round,
    "stop_reason": stop_reason,
    "success": ledger.best_loss_global <= 0.05
}

with open("final_result.json", "w", encoding='utf-8') as f:
    json.dump(final_result, f, indent=2, ensure_ascii=False)

print("最终结果已保存到 final_result.json")