def calculate_fluctuation(loss_history:list) ->str:
    if len(loss_history) <3:
        return "insufficient"

    recent = loss_history[-3:]
    range_value = max(recent) - min(recent)

    if range_value >0.5:
        return "high"

    elif range_value >0.1:
        return "medium"
    else:
        return "low"

class ResultObserver:

    @staticmethod
    def analyze(current_report:dict , history:dict) ->dict:
        current_loss = current_report["best_loss"]
        is_oom = current_report["is_oom"]
        time_seconds = current_report["time_seconds"]

        loss_history = history.get("loss_trend",[])
        full_history = loss_history +[current_loss]

        fluctuation = calculate_fluctuation(full_history)

        return {
            "current_loss":current_loss,
            "time_seconds":time_seconds,
            "is_oom":is_oom,
            "fluctuation":fluctuation,
            "loss_history":full_history
        }