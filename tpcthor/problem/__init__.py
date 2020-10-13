from . import base
from . import thor


def make_problem(problem_name, kwargs):
    # TODO: Error reporting?
    return base._problem_map[problem_name](**kwargs)
