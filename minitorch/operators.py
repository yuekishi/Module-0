"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # TODO: Implement for Task 0.1.
    return x*y
    # raise NotImplementedError("Need to implement for Task 0.1")


def id(x: float) -> float:
    "$f(x) = x$"
    # TODO: Implement for Task 0.1.
    return x
    # raise NotImplementedError("Need to implement for Task 0.1")


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # TODO: Implement for Task 0.1.
    return x+y
    raise NotImplementedError("Need to implement for Task 0.1")


def neg(x: float) -> float:
    "$f(x) = -x$"
    # TODO: Implement for Task 0.1.
    return -x
    raise NotImplementedError("Need to implement for Task 0.1")


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    if x<y:
        return 1.0
    else:
        return 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    if x==y:
        return 1.0
    else:
        return 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    if x>y:
        return x
    else:
        return y
    raise NotImplementedError("Need to implement for Task 0.1")


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # TODO: Implement for Task 0.1.
    return abs(x-y)< exp(-2)
    raise NotImplementedError("Need to implement for Task 0.1")


def sigmoid(x: float) -> float:
    """
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # TODO: Implement for Task 0.1.
    # if x>=0:
    #     return 1.0/(1.0+math.exp(-x))
    # else:
    #     return math.exp(x)/(1.0+math.exp(x))
    return  1.0/(1.0+math.exp(-x)) if x>=0 else math.exp(x)/(1.0+math.exp(x))
    # raise NotImplementedError("Need to implement for Task 0.1")


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # TODO: Implement for Task 0.1.
    if x>0:
        return x
    else:
        return 0
    raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return d/(x+EPS)
    raise NotImplementedError("Need to implement for Task 0.1")


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # TODO: Implement for Task 0.1.
    return 1/x
    raise NotImplementedError("Need to implement for Task 0.1")


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return -(1/x**2)*d
    raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    if x>0:
        return 1*d
    else:
        return 0
    raise NotImplementedError("Need to implement for Task 0.1")


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # TODO: Implement for Task 0.3.
    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return apply
    
    raise NotImplementedError("Need to implement for Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # TODO: Implement for Task 0.3.
    return map(neg)(ls)
    raise NotImplementedError("Need to implement for Task 0.3")


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    def apply(ls1:Iterable[float], ls2:Iterable[float]) -> Iterable[float]:
        if len(ls1) != len(ls2):
            raise ValueError("Lists must have the same length")
        result = []
        for i in range(len(ls1)):
            result.append(fn(ls1[i], ls2[i]))
        return result
    return apply
    raise NotImplementedError("Need to implement for Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # TODO: Implement for Task 0.3.
    return zipWith(add)(ls1, ls2)
    raise NotImplementedError("Need to implement for Task 0.3")


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    def apply(input_list):
        my_list = list(input_list).copy()

        if len(my_list) == 0:
            return start
        current_value = my_list.pop()
        return fn(current_value, apply(my_list))
    return apply

    raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # TODO: Implement for Task 0.3.
    my_fn = reduce(add, 0)
    return my_fn(ls)
    raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    my_fn = reduce(mul, 1)
    return my_fn(ls)
    raise NotImplementedError("Need to implement for Task 0.3")
