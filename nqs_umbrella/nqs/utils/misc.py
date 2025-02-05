import torch as pt
import numpy as np

from tqdm import tqdm


def verbosify_generator(generator=None,
                        verbose: bool = False,
                        activity_descr: str = None):
    if verbose:
        print(f'We start {activity_descr}')
        generator = tqdm(generator)

    return generator


def compute_chunk_boundaries(array_len: int = None,
                             chunk_size: int = None):
    chunk_boundaries = [0]
    len_left = array_len
    while len_left > 0:
        chunk_boundaries.append(chunk_boundaries[-1] + min(chunk_size, len_left))
        len_left -= chunk_size

    return chunk_boundaries


def hermitian_matmul(a):
    outer_size = a.shape[0]
    result = pt.zeros((outer_size, outer_size), dtype=a.dtype, device=a.device)
    for idx in range(outer_size):
        result[idx, idx:] = a[idx:idx + 1, :] @ a[idx:, :].T.conj()

    return result + pt.tril(result.T.conj(), diagonal=-1)


def soft_eigvals_inv(eigvals):
    eigvals_inv = 1 / eigvals
    return pt.where(pt.isclose(eigvals, pt.zeros_like(eigvals)),
                    pt.zeros_like(eigvals),
                    eigvals_inv)


def soft_matrix_inv(a):
    u, s, v = np.linalg.svd(a.cpu().numpy(), full_matrices=True)
    u = pt.from_numpy(u).to(a.device)
    s = pt.from_numpy(s).to(a.device)
    v = pt.from_numpy(v).to(a.device)
    s_inv = soft_eigvals_inv(s)

    return v.T.conj() @ pt.diag(s_inv.type(a.dtype)) @ u.T.conj()
