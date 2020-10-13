_problem_map = {}


class Problem:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _problem_map[cls.__name__] = cls
