from .embeddings import load_glove, load_word2vec, geometry_matrix
from .norms import load_mcrae, load_binder, truth_matrix
from .criterion import (
    augmented_rank, projector, defect, defect_table, holdout_defect,
    separability, principal_angles, random_geometry, admissible_dependences,
)
from .relations import bilinear_defect, identity_defect, equivalence_defect
from .parallel import parallelogram_defects
from .sweep import pca_geometry, ridge_holdout, probe_auc, forced_zero_angles, dimension_sweep
from .calib import monotone_defects
from .calib import nonlinear_defect
from .train import truth_floor, dist_retained, price_of_exactness, als_train, per_predicate_defect, sweep as train_sweep
