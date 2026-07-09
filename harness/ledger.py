import copy

class MemoryLedger:

    def __init__(self,init_config:dict):
        self.config = copy.deepcopy(init_config)

        self.round = 0
        self.loss_trend = []
        self.best_loss_global = 999.0
        self.consecutive_ooms = 0

    def get_config(self) ->dict:
        return copy.deepcopy(self.config)

    def rewrite_parameters(self,audited_args:dict) ->dict:
        new_config = copy.deepcopy(self.config)

        if "attack" not in new_config:
            new_config["attack"] = {}

        for key , value in audited_args.items():
            if isinstance(value,int) and value < 1:
                cleaned_value =1
            else:
                cleaned_value = value

            new_config["attack"][key] = cleaned_value

        self.config = new_config
        return new_config

    def update_history(self,current_loss:float,is_oom:bool):
        self.round += 1
        self.loss_trend.append(current_loss)

        if current_loss < self.best_loss_global:
            self.best_loss_global = current_loss

        if is_oom:
            self.consecutive_ooms += 1

        else:
            self.consecutive_ooms = 0

    def get_history_snapshot(self) ->dict:
        return {
            "round": self.round,
            "loss_trend": self.loss_trend.copy(),
            "best_loss_global": self.best_loss_global,
            "consecutive_ooms": self.consecutive_ooms
        }