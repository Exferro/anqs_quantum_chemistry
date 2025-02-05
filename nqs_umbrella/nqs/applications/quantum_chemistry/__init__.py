CELLS_ROOT_DIR = './cells'
CELL_PICKLE_FILENAME = 'cell.pickle'

MOLS_ROOT_DIR = './../mols'
GEOM_TYPES = ('custom', 'carleo', 'pubchem', 'curve', 'paper')

OF_HAM_PICKLE_FILENAME = 'of_ham.pickle'

SPLIT_SIZES_ROOT_DIR = 'split_sizes'
SPLIT_SIZE_FILENAME = 'split_size'

STATE_DICTS_ROOT_DIR = 'state_dicts'
STATE_DICT_FILENAME = 'state_dict'

EXPERIMENTS_ROOT_DIR = 'experiments'
RESULTS_ROOT_DIR = 'results'

CHEMICAL_ACCURACY = 1.6e-3

SYMMETRY_LEVELS = ('no_sym',
                   'e_num_spin',
                   'quasimom',
                   'z2')
SAMPLING_MODES = ('vanilla',
                  'masked_logits',
                  'masked_part_base_vecs',
                  'perseverant',
                  'renorm')
