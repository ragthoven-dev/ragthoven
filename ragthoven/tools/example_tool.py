"""
Every tool is a function that takes the data row in form of dictionary in a variable `args` and can modify it.
- Every tool must return the said args in the end.
- The tools are called sequentially as specified in yaml config.
"""


def fizzbuzz(args: dict[str, any]):
    text = str(args["article"])
    n_of_spaces = text.count(" ")
    args["fizzbuzz"] = (
        "fizzbuzz"
        if n_of_spaces % 3 and n_of_spaces % 5
        else "buzz" if n_of_spaces % 3 else "fizz" if n_of_spaces % 5 else "Nothing"
    )
    return args


def count_ands(args: dict[str, any]):
    text = str(args["article"])
    args["and_countes"] = text.count("and")
    return args


def replace_text_with_length(args: dict[str, any]):
    text_length = len(str(args["article"]))
    args["article"] = text_length
    return args
