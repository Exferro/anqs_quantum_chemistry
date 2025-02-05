import torch as pt

from .bf_quantum_state import BFQuantumState


def compute_log_jacobians(wf, sampled_indices):
    base_named_params = dict(wf.named_parameters())
    name_idx2name = {name_idx: name for name_idx, name in enumerate(base_named_params.keys())}

    def params2amps(*params):
        cur_named_params = {name_idx2name[name_idx]: param for name_idx, param in enumerate(params)}

        return pt.view_as_real(pt.log(pt.conj(pt.func.functional_call(wf, cur_named_params, sampled_indices))))

    return pt.autograd.functional.jacobian(params2amps,
                                           tuple(base_named_params[param_name] for param_name in base_named_params),
                                           vectorize=True)


def compute_log_grad(log_jacobians):
    log_grads = []
    for jacobian in log_jacobians:
        log_grads.append(pt.reshape(jacobian, (jacobian.shape[0], -1)))

    return pt.cat(log_grads, dim=-1)
