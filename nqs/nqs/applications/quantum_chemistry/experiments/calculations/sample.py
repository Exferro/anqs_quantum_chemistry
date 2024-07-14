import torch as pt

from .....infrastructure.nested_data import Config

from .....stochastic.ansatzes.anqs import AbstractANQS


class SamplingConfig(Config):
    FIELDS = (
        'sample_indices',
        'sample_num',
        'sample_precisely',
        'upscale_factor',
        'downscale_factor',
        'couple_spin_flip',
    )

    def __init__(self,
                 *args,
                 sample_indices: bool = True,
                 sample_num: int = 10000,
                 sample_precisely: bool = False,
                 upscale_factor: float = 3.0,
                 downscale_factor: float = 2.0,
                 couple_spin_flip: bool = False,
                 **kwargs):
        self.sample_indices = sample_indices
        self.sample_num = sample_num

        self.sample_precisely = sample_precisely
        self.upscale_factor = upscale_factor
        self.downscale_factor = downscale_factor

        self.couple_spin_flip = couple_spin_flip

        super().__init__(*args, **kwargs)


class SamplingResult:
    def __init__(self,
                 *args,
                 indices: pt.Tensor = None,
                 counts: pt.Tensor = None,
                 **kwargs):
        self.indices = indices
        self.counts = counts

        super().__init__(*args, **kwargs)


def sample(wf: AbstractANQS = None,
           config: SamplingConfig = None,
           starting_sample_num: int = None,
           verbose: bool = False):
    repetition_num = 1
    next_rep_sample_num = starting_sample_num
    if config.sample_indices:
        indices, counts = wf.sample_indices_gumbel(sample_num=config.sample_num)
        next_rep_sample_num = config.sample_num
        actual_unq_num = indices.shape[0]
    else:
        if config.sample_precisely:
            while True:
                indices, counts = wf.sample_stats(sample_num=next_rep_sample_num)
                if indices.shape[0] > config.sample_num:
                    if (next_rep_sample_num / config.downscale_factor) > config.sample_num:
                        next_rep_sample_num /= config.downscale_factor
                    break
                elif indices.shape[0] == config.sample_num:
                    break
                else:
                    repetition_num += 1
                    next_rep_sample_num *= config.upscale_factor
            actual_unq_num = indices.shape[0]
            counts, sorted_counts_inv = pt.sort(counts.real, descending=True)
            indices = indices[sorted_counts_inv[:config.sample_num]]
        else:
            indices, counts = wf.sample_stats(sample_num=config.sample_num)
            next_rep_sample_num = config.sample_num
            actual_unq_num = indices.shape[0]

    counts = counts.type(wf.cdtype).to(wf.device)

    if verbose:
        print(f'We have sampled {indices.shape[0]} indices')
        print(f'They amounted for a total of {counts.sum()} samples')
        print(
            f'Which is a {counts.sum() / config.sample_num} fraction of the required number of samples')

    if config.couple_spin_flip:
        with pt.no_grad():
            spin_flip_indices = wf.spin_flip_base_idx(indices)
            indices = pt.cat((indices, spin_flip_indices), dim=0)
            indices, _ = wf.hilbert_space.compute_unique_indices(indices)
            if verbose:
                print(f'After complementing them with spin flip partners we have {indices.shape[0]}')
            counts = wf.amplitude(indices)
            counts = pt.conj(counts) * counts
            counts = counts / pt.sum(counts)

    return SamplingResult(indices=indices, counts=counts), actual_unq_num, repetition_num, next_rep_sample_num
