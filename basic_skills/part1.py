import numpy as np
from numpy import r_, array, ndarray
from typing import Callable

def deriv(func: Callable[[ndarray],ndarray],
          in_: ndarray,
          delta: float = 0.001) -> ndarray:
    return (func(in_+delta) - func(in_-delta))/(2*delta)

y = lambda x: x**2
print(deriv(y,1))

from typing import List

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

from collections.abc import Iterable
##def chain_run(chain: Chain, a: ndarray) -> ndarray:
##    for ch in chain:
##        fun = fun(ch)
##    return fun(a)
##    if ~isinstance(chain, Iterable):
##        chain=iter(chain)
##    return chain_run(next(chain)(a),a)
def chain_run(chain):
    if isinstance(chain, Iterable):
        return chain_run(next(chain))
    return chain

def y1(a):return a**2

import pdb; pdb.set_trace();pdb.jump(28)
print(chain_run(iter([y1,y1,y1]))(2))
