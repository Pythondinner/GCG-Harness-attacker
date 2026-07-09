class GCGContract:

    @staticmethod
    def get_schema() ->dict:
        return {
            "num_steps": {"type": "integer", "minimum": 10, "maximum": 300},
            "search_width": {"type": "integer", "minimum": 10, "maximum": 512},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 128},
            "seed": {"type": "integer", "minimum": 1, "maximum": 9999}
        }

    @staticmethod
    def get_required_fields() ->list:
        return ["num_steps", "search_width", "batch_size"]