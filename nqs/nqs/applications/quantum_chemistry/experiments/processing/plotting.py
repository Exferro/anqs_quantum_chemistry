import os
import pandas as pd
import pickle

import numpy as np
import scipy

from matplotlib import pyplot as plt

from typing import Union
from typing import Tuple, List, Dict, Set
from typing import Callable

from nqs.applications.quantum_chemistry.experiments.preparation.infrastructure import MolDescr, ExpDescr, SeedlessExpDescr
from nqs.applications.quantum_chemistry.experiments.preparation.infrastructure import MOL_DESCR_FIELDS, EXP_DESCR_FIELDS
from nqs.applications.quantum_chemistry.experiments.preparation.infrastructure import create_mol

from nqs.applications.quantum_chemistry import EXPERIMENTS_ROOT_DIR, RESULTS_ROOT_DIR
from nqs.applications.quantum_chemistry import CHEMICAL_ACCURACY

from nqs.applications.quantum_chemistry.experiments.processing.plotting_repo import DESCR_FIELD_TO_LABEL, COL_TO_LABEL, COL_TO_STE_COL
from nqs.applications.quantum_chemistry.experiments.processing.plotting_repo import METHOD_TO_COLOR, METHOD_TO_LABEL


def get_result(exp_descr: ExpDescr = None,
               mols_dir: str = None) -> Union[pd.DataFrame, None]:
    mol_dir = os.path.join(mols_dir,
                           *[f'{descr_field}={getattr(exp_descr, descr_field)}'
                             for descr_field in MOL_DESCR_FIELDS])
    result_filename = os.path.join(mol_dir,
                                   EXPERIMENTS_ROOT_DIR,
                                   *[f'{descr_field}={getattr(exp_descr, descr_field)}'
                                     for descr_field in EXP_DESCR_FIELDS
                                     if descr_field not in MOL_DESCR_FIELDS],
                                   RESULTS_ROOT_DIR,
                                   'result')

    if os.path.exists(result_filename):
        dummy_df = pd.read_csv(result_filename)

        return pd.read_csv(result_filename, usecols=list(range(1, len(dummy_df.columns))))
    else:
        return None


def get_results(mol_name: str = None,
                exp_series_suffix: str = None,
                mols_dir: str = None) -> Dict[ExpDescr, pd.DataFrame]:
    exp_series_name = f'{mol_name}_{exp_series_suffix}'
    exp_series_dir = os.path.join(mols_dir,
                                  'exp_series',
                                  exp_series_name)
    exp_descrs_filename = os.path.join(exp_series_dir, 'exp_descrs.pickle')
    if os.path.exists(exp_descrs_filename):
        with open(exp_descrs_filename, 'rb') as f:
            exp_descrs = pickle.load(f)
    else:
        raise RuntimeError(f'No exp_descrs.pickle file exists at {exp_descrs_filename}')

    results = {}
    for exp_descr in exp_descrs:
        result = get_result(exp_descr,
                            mols_dir)
        if result is not None:
            results[exp_descr] = result

    return results


def get_processed_results(results: Dict[ExpDescr, pd.DataFrame] = None,
                          mols_dir: str = None,
                          df_process_lambda: Callable = None,
                          filter_lambda: Callable = None) -> Tuple[Dict[ExpDescr, pd.DataFrame], Dict[MolDescr, Dict[str, float]]]:
    processed_results = {}
    mol_to_diffs = {}
    for exp_descr, df in results.items():
        if (filter_lambda is None) or (filter_lambda(exp_descr) is True):
            mol_descr = MolDescr(**{field: getattr(exp_descr, field) for field in MOL_DESCR_FIELDS})
            mol, mol_dir = create_mol(mol_descr=mol_descr,
                                      mols_dir=mols_dir)
            fci_energy = mol.fci_energy if mol.fci_energy is not None else mol.ccsd_t_energy

            df['energy_diff'] = df['energy'] - fci_energy
            df['var_energy_diff'] = df['var_energy'] - fci_energy
            df['cumtime'] = df['iter_time'].cumsum()
            processed_results[exp_descr] = df if df_process_lambda is None else df_process_lambda(exp_descr, df)

            if mol_descr not in mol_to_diffs:
                mol_to_diffs[mol_descr] = {}
                for method in ['hf', 'cisd', 'ccsd', 'ccsd_t']:
                    mol_to_diffs[mol_descr][method] = getattr(mol, f'{method}_energy') - fci_energy
                mol_to_diffs[mol_descr]['chem_acc'] = CHEMICAL_ACCURACY

    return processed_results, mol_to_diffs


def concat_over_seeds(results: Dict[ExpDescr, pd.DataFrame] = None) -> Dict[SeedlessExpDescr, pd.DataFrame]:
    seedless_results = {}
    for exp_decsr, df in results.items():
        seedless_exp_descr = SeedlessExpDescr(**{descr_field: getattr(exp_decsr, descr_field)
                                                 for descr_field in EXP_DESCR_FIELDS[:-1]})
        if seedless_exp_descr not in seedless_results:
            seedless_results[seedless_exp_descr] = df
        else:
            seedless_results[seedless_exp_descr] = pd.concat([seedless_results[seedless_exp_descr],
                                                              df],
                                                             ignore_index=True)

    return seedless_results


def results_to_hier_exp_descrs(results: Dict[ExpDescr, pd.DataFrame] = None,
                               lvl_1: str = None,
                               lvl_2: str = None,
                               filter_lambda: Callable = None) -> Tuple[Dict[str, Dict[str, List[ExpDescr]]], Dict[str, Set[MolDescr]]]:
    hier_exp_descrs = {}
    hier_mol_descrs = {}
    for exp_descr in results:
        if (filter_lambda is None) or (filter_lambda(exp_descr) is True):
            lvl_1_val = getattr(exp_descr, lvl_1)
            lvl_2_val = getattr(exp_descr, lvl_2)
            if lvl_1_val not in hier_exp_descrs:
                hier_exp_descrs[lvl_1_val] = {}
            if lvl_2_val not in hier_exp_descrs[lvl_1_val]:
                hier_exp_descrs[lvl_1_val][lvl_2_val] = []
            hier_exp_descrs[lvl_1_val][lvl_2_val].append(exp_descr)

            mol_descr = MolDescr(**{field: getattr(exp_descr, field) for field in MOL_DESCR_FIELDS})
            if lvl_1_val not in hier_mol_descrs:
                hier_mol_descrs[lvl_1_val] = set()
            hier_mol_descrs[lvl_1_val].add(mol_descr)

    return hier_exp_descrs, hier_mol_descrs


def plot_levels(results: Dict[ExpDescr, pd.DataFrame] = None,
                filter_lambda: Callable = None,
                lvl_1: str = None,
                lvl_2: str = None,
                x_col: str = None,
                y_col: str = None,
                x_agg_f: Callable = scipy.mean,
                y_agg_f: Callable = scipy.mean,
                legend_fields: List[str] = None,
                figs: Union[List[plt.Figure], None] = None,
                axes: Union[List[plt.Axes], None] = None,
                plot_ste: bool = False,
                plot_min_max: bool = False,
                yscale: str = 'log',
                alpha: float = 0.5,
                ste_alpha: float = 0.25,
                fs: int = None):
    hier_exp_descrs, hier_mol_descrs = results_to_hier_exp_descrs(results=results,
                                                                  lvl_1=lvl_1,
                                                                  lvl_2=lvl_2,
                                                                  filter_lambda=filter_lambda)
    assert ((figs is None) and (axes is None)) or ((figs is not None) and (axes is not None))
    if figs is not None:
        assert len(figs) == len(hier_exp_descrs.keys())
        assert len(axes) == len(figs)
    else:
        figs = []
        axes = []

    for fig_ax_idx, lvl_1_val in enumerate(hier_exp_descrs.keys()):
        if len(figs) <= fig_ax_idx:
            fig, ax = plt.subplots()
            ax.grid()

            figs.append(fig)
            axes.append(ax)

        fig = figs[fig_ax_idx]
        ax = axes[fig_ax_idx]

        assert len(hier_mol_descrs[lvl_1_val]) == 1
        mol_descr = next(iter(hier_mol_descrs[lvl_1_val]))
        ax.set_title(f'{mol_descr.mol_name}, ' + DESCR_FIELD_TO_LABEL[lvl_1] + ' = ' + f'{lvl_1_val}')
        ax.set_xlabel(COL_TO_LABEL[x_col], fontsize=fs)
        ax.set_ylabel(COL_TO_LABEL[y_col], fontsize=fs)

        for lvl_2_val in hier_exp_descrs[lvl_1_val].keys():
            for exp_descr in hier_exp_descrs[lvl_1_val][lvl_2_val]:
                df = results[exp_descr]
                gb = df.groupby('iter_idx')
                ax.plot(gb[x_col].apply(x_agg_f),
                        gb[y_col].apply(y_agg_f),
                        label=f', '.join(
                            [f'{DESCR_FIELD_TO_LABEL[descr_field]}' + ' = ' + f'{getattr(exp_descr, descr_field)}'
                             for descr_field in legend_fields]),
                        alpha=alpha)
                assert not ((plot_ste is True) and (plot_min_max is True))
                if plot_ste:
                    ax.fill_between(df[x_col],
                                    df[y_col] - df[COL_TO_STE_COL[y_col]],
                                    df[y_col] + df[COL_TO_STE_COL[y_col]],
                                    alpha=ste_alpha)
                if plot_min_max:
                    ax.fill_between(gb[x_col].mean(),
                                    gb[y_col].min(),
                                    gb[y_col].max(),
                                    alpha=ste_alpha)
        ax.set_yscale(yscale)
        ax.legend(loc='best')

    return figs, axes


def add_ref_energies(mol_to_diffs: Dict[MolDescr, Dict[str, float]] = None,
                     axes: List[plt.Axes] = None,
                     yscale: str = 'log',
                     linewidth: int = 1,
                     alpha: float = 0.5) -> List[plt.Axes]:
    assert len(mol_to_diffs.keys()) == 1

    for ax in axes:
        x_lim = ax.get_xlim()
        point_num = 10
        x_axis = np.linspace(0, x_lim[1], point_num)
        for method, diff in list(mol_to_diffs.values())[0].items():
            ax.plot(x_axis,
                    [diff] * point_num,
                    linestyle='--',
                    color=METHOD_TO_COLOR[method],
                    label=METHOD_TO_LABEL[method],
                    linewidth=linewidth,
                    alpha=alpha)

            ax.set_yscale(yscale)
            ax.legend(loc='best')

    return axes
