from __future__ import annotations

from typing import cast

from typeguard import typechecked, check_type

from arkouda.client import generic_msg
from arkouda.dtypes import bigint, float64, int64, int_scalars, uint64
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.pdarraycreation import zeros

supported_dtypes = {uint64} # Add more as the support grows
numeric_dtypes = {int64, uint64, float64}


__all__ = ["times2", "gpuScan", "gpuSort"]

def times2(pda: pdarray) -> pdarray:
  """
    Returns a pdarray with each entry double that of the input.
    Run on the GPUs

    Parameters
    ----------
    pda : pdarray
        The array to double.

    Returns
    -------
    pdarray
        The doubled array

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    ValueError
        Raised if sort attempted on a pdarray with an unsupported dtype
        such as bool

    Notes
    -----
    Copies the array to GPU memory and launches a kernel and returns the array
    multiplied by two.

    Examples
    --------
    >>> a = ak.randint(0, 10, 10, dtype=ak.uint64)
    >>> a
    array([7 7 9 0 0 9 8 8 4 3])
    >>> ak.times2(a)
    array([14 14 18 0 0 18 16 16 8 6])
  """
  check_type(argname="times2", value=pda, expected_type=pdarray)
  if pda.dtype not in supported_dtypes:
        raise ValueError(f"ak.times2 does not support {pda.dtype}")
  repMsg = generic_msg(cmd="times2", args={"array": pda})
#   return create_pdarray(repMsg)
  return repMsg

@typechecked
def gpuScan(pda: pdarray) -> pdarray:
    """
    Returns an array which is an exclusive prefix sum (scan) of the given
    array. Only supports numeric arrays. Runs on gpu

 Parameters
    ----------
    pda : pdarray or Categorical
        The array to sort (int64, uint64, or float64)

    Returns
    -------
    pdarray, int64, uint64, or float64
        The scanned copy of pda

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    ValueError
        Raised if sort attempted on a pdarray with an unsupported dtype
        such as bool

    Notes
    -----
    Copies the array to GPU memory and launches a kernel and uses a parallel
    scan algorithm to carry out the prefix sum.

    Examples
    --------
    >>> a = ak.randint(0, 10, 10)
    >>> a
    array([5 3 7 7 5 9 0 5 9 2])
    >>> ak.gpuScan(a)
    array([0 5 8 15 22 27 36 36 41 50])
    """
    if not pda.on_gpu:
        raise ValueError("ak.gpuScan only works on GPU arrays")
    if pda.dtype not in numeric_dtypes:
        raise ValueError(f"ak.gpuScan supports int64, uint64, or float64, not {pda.dtype}")
    if pda.size == 0:
        return zeros(0, dtype=pda.dtype)
    repMsg = generic_msg(cmd="gpuScan", args={"array": pda})
    return repMsg

@typechecked
def gpuSort(pda: pdarray) -> pdarray:
    """
    Return a sorted copy of the array. Only sorts uint arrays

    Parameters
    ----------
    pda : pdarray or Categorical
        The array to sort (uint64)

    Returns
    -------
    pdarray, uint64
        The sorted copy of pda

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    ValueError
        Raised if sort attempted on a pdarray with an unsupported dtype
        such as bool

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive.

    Examples
    --------
    >>> a = ak.randint(0, 10, 10, dtype=ak.uint64)
    >>> a
    array([9 8 3 6 2 0 3 9 1 9])
    >>> ak.gpuSort(a)
    array([0 1 2 3 3 6 8 9 9 9])
    """
    if pda.dtype not in supported_dtypes:
        raise ValueError(f"ak.gpuSort supports uint64 not {pda.dtype}")
    if pda.size == 0:
        return zeros(0, dtype=pda.dtype)
    repMsg = generic_msg(cmd="gpuSort", args={"array": pda})
    return create_pdarray(cast(str, repMsg))
