from ...molecule import MolConfig, MolInitConfig, Molecule


def create_mol(config: MolConfig = None,
               init_config: MolInitConfig = None,
               mols_root_dir: str = None):

    return Molecule.create(config=config,
                           init_config=init_config,
                           mols_root_dir=mols_root_dir)
