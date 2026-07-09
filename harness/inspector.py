from .contract import GCGContract

class SignalInspector:
    @staticmethod
    def get_schema() ->dict:
        return GCGContract.get_schema()

    @staticmethod
    def audit(raw_args:dict) ->tuple:
        schema = GCGContract.get_schema()
        required = GCGContract.get_required_fields()

        for key in required:
            if key not in raw_args:
                return False, f'缺少必要参数:{key}'
        for key,value in raw_args.items():
            if key not in schema:
                return False,f'参数不正确:{key}'

            rules = schema[key]
            if rules["type"] == "integer":
                if not isinstance(value, int):
                    return False, f"参数 {key} 应为整数，实际为 {type(value)}"
                if not (rules["minimum"] <= value <= rules["maximum"]):
                    return False, f"参数 {key} 值 {value} 超出范围 [{rules['minimum']}, {rules['maximum']}]"

        return True, "核验通过"