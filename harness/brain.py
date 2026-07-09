import random
import os
import json
import time
from openai import OpenAI

class DecisionBrain:
    def __init__(self,max_rounds = 5 , target_loss = 0.05):
        self.max_rounds = max_rounds
        self.target_loss = target_loss

    def decide(self, observation: dict, history: dict) -> dict:
        current_round = history.get("round", 0)
        current_loss = observation["current_loss"]
        is_oom = observation["is_oom"]
        fluctuation = observation["fluctuation"]

        if current_loss <= self.target_loss:
            return {
                "action": "STOP_WIN",
                "reason": f"Loss达标{current_loss:.4f}"
            }

        if current_round >= self.max_rounds:
            return {
                "action": "STOP_SAFE",
                "reason": "到达最大轮次"
            }

        if history.get("consecutive_ooms", 0) >= 3:
            return {
                "action": "STOP_SAFE",
                "reason": "Out of Memory"
            }

        if fluctuation == "high":
            return {
                "action": "PERTURB",
                "reason": "高波动，开启强制扰动",
                "forced_params": {"num_steps": random.randint(50, 300)}
            }

        prompt = self._build_prompt(observation, history)
        try:
            new_params = self._call_llm(prompt)
            return {
                "action": "CONTINUE",
                "reason": "常规调参成功",
                "new_params": new_params
            }

        except RuntimeError as e:
            return {
                "action": "STOP_LLM_FAILURE",
                "reason": str(e)
            }

    def _build_prompt(self,observation:dict , history:dict) -> str:
        loss = observation["current_loss"]
        fluct = observation["fluctuation"]
        round_num = history.get("round", 0)
        return f"第 {round_num} 轮攻击结束。当前 Best Loss 为 {loss:.4f}，波动状态为 {fluct}。请调整攻击参数（num_steps, search_width, batch_size, seed）以降低 Loss。"

    def _call_llm(self,prompt:str , max_retries:int = 2 ) ->dict:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DeepSeek API Key，请设置环境变量 DEEPSEEK_API_KEY")

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )


        #后续可优化prompt或者更改交互方式来压缩回答概率空间
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

        for attempt in range(max_retries +1):
            try:
                response = client.chat.completions.create(
                    model = "deepseek-chat",
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

                from .inspector import SignalInspector
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
                error_feedback = f"你返回的内容不是有效的JSON格式。请只返回JSON对象。"
                current_prompt = f"{current_prompt}\n\n【错误反馈】{error_feedback}"
                print(f"第 {attempt + 1} 次JSON解析失败，正在重试...")

            except Exception as e:
                print(f"第 {attempt + 1} 次请求异常: {e}，重试----")
                continue

        raise RuntimeError("外部模型连续返回非法参数，已放弃")