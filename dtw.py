from nptyping import NDArray
from numba import jit, prange
from typing import Any, List, Tuple, Union
import numpy as np

@jit(nopython=True)
def __atleast_1d(x: Union[NDArray[np.float_], float]):
    zero = np.zeros((1,))
    
    return x + zero

@jit(nopython=True)
def __compute_region(
    global_constraint: str,
    len_x: int,
    len_y: int,
    r: int
) -> List[Tuple[int, int]]:
    """
    Compute region based on global constraint

    Parameters
    ----------
    global_constraint : {None, 'sakoe_chiba'} (default: None)
        Global constraint to restrict DTW path
    len_x : int
        Time series 1 length
    len_y : int
        Time series 2 length
    r : int
        Region radius
        
    Returns
    -------
    region : List[Tuple[int, int]]
        Region
    """
    if global_constraint is not None and global_constraint not in ('sakoe_chiba'):
        raise Exception('Invalid global_constraint')
        
    if global_constraint == 'sakoe_chiba' and r < 1:
        raise Exception('Radius for sakoe_chiba must be greater than 0')

    if global_constraint is None:
        region = [(i, j) for i in prange(len_x) for j in prange(len_y)]
    elif global_constraint == 'sakoe_chiba':
        region = []

        if len_x <= len_y:
            w = len_y - len_x + r
            
            for i in prange(len_x):
                lower = max(0, i - r)
                upper = 1 + min(i + w, len_y)
                
                for j in prange(lower, upper):
                    if j < len_y:
                        region.append((i, j))
        else:
            w = len_x - len_y + r
            
            for i in prange(len_y):
                lower = max(0, i - r)
                upper = 1 + min(i + w, len_x)
                
                for j in prange(lower, upper):
                    if j < len_x:
                        region.append((j, i))
        
    return region

@jit(nopython=True)
def __expanded_res_window(
    len_x: int,
    len_y: int,
    path: List[Tuple[int, int]],
    r: int
) -> List[Tuple[int, int]]:
    """
    Expand resolution window through warp path projected from the lower resolution

    Parameters
    ----------
    len_x : int
        Time series 1 length
    len_y : int
        Time series 2 length
    path : List[Tuple[int, int]]
        Warp path
    r : int
        Region radius
        
    Returns
    -------
    window : List[Tuple[int, int]]
        Expanded resolution window
    """
    cur_i, cur_j = 0, 0
    last_i, last_j = 2147483647, 2147483647
    window = []
    
    for i, j in path:
        if i > last_i: # move vertically
            cur_i += 2
            
        if j > last_j: # move horizontally
            cur_j += 2
            
        """
        If a diagonal move was performed, add 2 cells to the edges of the 2 blocks in the projected path to create a continuous path (path with even width...avoid a path of boxes connected only at their corners).
                            |x|x|_|_|     then mark      |_|_|x|x|
        ex: projected path: |x|x|_|_|  --2 more cells->  |_|X|x|x|
                            |_|_|x|x|        (X's)       |x|x|X|_|
                            |_|_|x|x|                    |x|x|_|_|
        """
        if i > last_i and j > last_j: # move diagonally
            window.append((cur_i - 1, cur_j))
            window.append((cur_i, cur_j - 1))
            
        for k in prange(2):
            for l in prange(2):
                window.append((cur_i + k, cur_j + l))
                
        last_i = i
        last_j = j
        
    window_ = set(window)
    
    for w_i, w_j in window:
        for i in prange(-r, 1 + r):
            for j in prange(-r, 1 + r):
                if i + w_i >= 0 and j + w_j >= 0 and i + w_i < len_x and j + w_j < len_y:
                    window_.add((i + w_i, j + w_j))
                    
    return sorted(list(window_))

@jit(nopython=True)
def __reduce_by_half(x: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Shrink a time series into a smaller time series by a factor of 2

    Parameters
    ----------
    x : NDArray[np.float_]
        Time series
        
    Returns
    -------
    result : NDArray[np.float_]
        Shrunk time series
    """
    if x.ndim == 1:
        return np.array([(x[i] + x[i + 1]) / 2 for i in range(0, len(x) - len(x) % 2, 2)])
    elif x.ndim == 2:
        result = np.zeros((1, x.shape[1]))
        
        for i in prange(0, len(x) - len(x) % 2, 2):
            result = np.append(result, np.expand_dims((x[i] + x[i + 1]) / 2, axis=0), axis=0)
            
        return result[1:]

@jit(nopython=True)
def dtw(
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    dist_order: int = 1,
    global_constraint: str = None,
    r: int = 1,
    region: List[Tuple[int, int]] = None
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Return the distance and the path between 2 time series

    Parameters
    ----------
    x : NDArray[np.float_]
        Time series 1
    y : NDArray[np.float_]
        Time series 2
    dist_order : int
        Numpy matrix or vector norm order
    global_constraint : {None, 'sakoe_chiba'} (default: None)
        Global constraint to restrict DTW path
    r : int
        Region radius
    region : List[Tuple[int, int]]
        Predefined region. Will be prioritized over global_constraint
        
    Returns
    -------
    result : Tuple[float, List[Tuple[int, int]]]
        The distance and the path between 2 time series
    """
    len_x, len_y = len(x), len(y)
    
    if x.ndim != y.ndim:
        raise Exception('x and y must have the same ndim')
        
    if x.ndim == 2 and y.ndim == 2 and x.shape[1] != y.shape[1]:
        raise Exception('x and y must have the same second dimension')
        
    if x.ndim > 2:
        raise Exception('The maximum ndim of x is 2')
                
    if y.ndim > 2:
        raise Exception('The maximum ndim of y is 2')

    cost_mat = np.full((1 + len_x, 1 + len_y), np.inf)
    cost_mat[0, 0] = 0
    path_mat = np.full((len_x, len_y, 2), np.inf, dtype=np.int16)
    
    if region is None:
        region = __compute_region(global_constraint, len_x, len_y, r)

    for i, j in region:
        dist = np.linalg.norm(__atleast_1d(x[i]) - __atleast_1d(y[j]), ord=dist_order)
        paths = np.array([
            cost_mat[1 + i, j], # deletion
            cost_mat[i, 1 + j], # insertion
            cost_mat[i, j] # match
        ])
        argmin = paths.argmin()
        cost_mat[1 + i, 1 + j] = dist + paths[argmin]
        
        if argmin == 0:
            path_mat[i, j] = (1 + i, j)
        elif argmin == 1:
            path_mat[i, j] = (i, 1 + j)
        elif argmin == 2:
            path_mat[i, j] = (i, j)
            
    i, j = len_x, len_y
    path = []
    
    while not (i == 0 and j == 0):
        path.append((i - 1, j - 1))
        i, j = path_mat[i - 1, j - 1]
        
    path.reverse()
    # print(cost_mat)
    
    return (cost_mat[len_x, len_y], path)

@jit(nopython=True)
def fastdtw(
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    dist_order: int = 1,
    r: int = 1
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Return the distance and the path between 2 time series

    Parameters
    ----------
    x : NDArray[np.float_]
        Time series 1
    y : NDArray[np.float_]
        Time series 2
    dist_order : int
        Numpy matrix or vector norm order
    r : int
        Region radius
        
    Returns
    -------
    result : Tuple[float, List[Tuple[int, int]]]
        The distance and the path between 2 time series
    """
    if r < 0:
        r = 0
        
    min_ts_size = 2 + r
    
    if len(x) < min_ts_size or len(y) < min_ts_size:
        return dtw(x, y, dist_order=dist_order)
    
    shrunk_x = __reduce_by_half(x)
    shrunk_y = __reduce_by_half(y)
    distance, path = fastdtw(shrunk_x, shrunk_y, dist_order=dist_order, r=r)
    window = __expanded_res_window(len(x), len(y), path, r)
    
    return dtw(x, y, dist_order=dist_order, region=window)
