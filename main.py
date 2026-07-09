import yaml
import json
import os
from datetime import datetime
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

from harness.ledger import MemoryLedger
from harness.executor import LocalExecutor
from harness.observer import ResultObserver
from harness.brain import DecisionBrain
from harness.inspector import SignalInspector

print(f'GCG_Harness loading----')

with open("config.yaml",'r',encoding = 'utf-8') as f:
    config_data = yaml.safe_load(f)

ledger = MemoryLedger(config_data)

print(f'加载模型中----')

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
brain = DecisionBrain(max_rounds = 5,target_loss = 0.05)

audit_log_path = "audit_log.jsonl"
if os.path.exists(audit_log_path):
    os.remove(audit_log_path)
print(f'审计日志路径:{audit_log_path}')

def write_audit_log(entry:dict):
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(audit_log_path,"a",encoding = "utf-8") as f:
        f.write(json.dumps(entry,ensure_ascii=False) + "\n")

stop_reason = "未启动"
best_string = ""
generated_response = ""

max_rounds = 5
while ledger.round<max_rounds:
    current_round = ledger.round+1
    print(f"\n{'=' * 40}")
    print(f"第 {current_round} 轮攻击开始")

    current_config = ledger.get_config()
    report = executor.execute(current_config,tokenizer,model)

    generated_response = report.get("generated_response","")
    best_string = report.get("best_string","")
    print(f'本轮结果：Loss={report["best_loss"]:.4f},耗时={report["time_seconds"]:.1f}秒')

    ledger.update_history(report["best_loss"],report["is_oom"])

    history_snapshot = ledger.get_history_snapshot()
    observation = observer.analyze(report,history_snapshot)
    print(f'分析日志:Loss={observation["current_loss"]:.4f},波动 = {observation["fluctuation"]}')

    decision = brain.decide(observation,history_snapshot)
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