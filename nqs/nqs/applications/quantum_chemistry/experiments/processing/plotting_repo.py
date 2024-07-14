DESCR_FIELD_TO_LABEL = {
    'sample_num': r'$N_{\rm S}$',
    'sampling_mode': 'Sampling mode',
    'masking_depth': 'Masking depth',
    'depth': r'$D_{\rho}$',
    'width': r'$W_{\rho}$',
    'activation_combo': 'Activation combo',
    'phase_depth': r'$D_{\varphi}$',
    'phase_width': r'$W_{\varphi}$',
    'phase_activation_combo': 'Phase activation combo',
    'device': 'Device',
    'rng_seed': 'RNG seed',
    'ansatz': 'Ansatz',
    'symmetry_level': 'Symmetry level',
    'use_sr': 'Use SR',
    'max_sr_indices': 'Max SR indices',
    'use_reg_sr': 'Use reg-SR',
    'sr_reg': 'SR reg',
    'renorm_grad': 'Renorm grad',
    'clip_grad_norm': 'clip_grad_norm',
}
COL_TO_LABEL = {
    'iter_idx': r'Iteration',
    'energy': r'$\langle E_{\rm loc} \rangle$ [Ha]',
    'energy_diff': r'$E - E_{\rm FCI}$ [Ha]',
    'ste_est': r'$\widehat{\rm SE}E_{\rm loc}$',
    'iter_time': r'Time, s',
    'cumtime': r'Time, s',
    'counts_sum': 'Number of samples',
    'wcounts_sum': 'Weighted number of samples',
    'unique_num': 'Number of sampled unique indices',
    'var_energy': r'Var $\langle E_{\rm loc} \rangle$ [Ha]',
    'var_energy_diff': r'Var $E - E_{\rm FCI}$ [Ha]',
    'var_ste_est': r'Var $\widehat{\rm SE}E_{\rm loc}$',
}

COL_TO_STE_COL = {
    'energy': 'ste_est',
    'energy_diff': 'ste_est',
    'var_energy': 'var_ste_est',
    'var_energy_diff': 'var_ste_est',
}

METHOD_TO_COLOR = {
        'hf': 'xkcd:periwinkle',
        'chem_acc': 'xkcd:pink',
        'ccsd': 'xkcd:peach',
        'ccsd_t': 'xkcd:royal blue',
        'cisd': 'xkcd:magenta',
}
METHOD_TO_LABEL = {
        'hf': 'HF',
        'chem_acc': 'Chem. acc.',
        'ccsd': 'CCSD',
        'cisd': 'CISD',
        'ccsd_t': 'CCSD(T)',
}
