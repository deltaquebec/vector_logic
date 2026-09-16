# Faithful compression in a vector logic for formal semantics

This repository accompanies the paper:

> **Faithful compression in a vector logic for formal semantics**
> Daniel Quigley
> [arXiv link](https://www.arxiv.org/abs/2512.06205)

## Overview


A vector logic for formal semantics in [Quigley (2025)](https://link.springer.com/article/10.1007/s10849-025-09443-x) and [Quigley (2026)](https://arxiv.org/abs/2602.02940) prove that the typed models of Montague semantics embed into vector spaces by a family of injections for each semantic type, such that every semantic function lifts to a linear map on the free carriers, and composition is preserved. For a primitive domain, the construction assigns each element its own basis vector. This is a one-hot representation: distinct entities are linearly independent, and every pair of distinct entities has the same Euclidean distance. The construction guarantees exact preservation, but it does not resemble the dense, learned geometries produced by distributional training.

We follow [Sebastian Raschka](https://sebastianraschka.com/faq/docs/embedding-linear-onehot.html), in treating an embedding layer that maps a token identifier to a row of a learned matrix as equivalent, on one-hot inputs, to a bias-free linear layer with weight matrix. An embedding layer can, consequently, be understood as the free carrier supplied by the extensional construction, followed by the linear map. The free carrier supplies the full extensional space; an empirical embedding occupies only its image under that matrix. Passing from the carrier to the embedding imposes the linear dependences on the entity vectors; a geometry with distinct columns still determines every predicate on a finite domain by a lookup, so what the dependences decide is which predicates remain linear readouts.

Laconically:

- Problem of interest here is when can compressed embeddings preserve a lexicon’s truth conditions through linear readouts.
- We derived an exact row-space criterion and minimum dimension, then tested pretrained and supervised geometries.
- We show that many predicates were linearly separable, but none was exactly affine-recoverable from the pretrained embeddings. Supervision achieved numerical exactness; approximate recovery retained most pretrained variance.

## Repository contents

Note: need GloVe and word2vec in code root directory

```

├── paper/
│   └── quigley_daniel_compression.pdf    main paper
├── code/
│   ├── make_figures.py                   figures from the results csvs
│   ├── run_diagnostics.py                measurements on fixed geometry
│   ├── run_separability.py               strict separability per predicate by linear feasibility
│   ├── run_train.py                      training under mixed objective
│   ├── vlcompress/                       modules
│   └── CONCS_FEATS_concstats_brm.txt     McRae et al. 2005 feature norms
└── README.md
```

## Data

- GloVe text file, e.g. `glove.6B.300d.txt`.
- word2vec GoogleNews binary (`gensim` required).
- McRae et al. 2005 `CONCS_FEATS_concstats_brm.txt`.
- Binder et al. 2016 `WordSet1_Ratings.xlsx` (optional; ratings binarized at 3.0).

## Runs for reproducibility

```
python run_diagnostics.py --glove G.txt --w2v W.bin --mcrae CONCS_FEATS_concstats_brm.txt \
    --min_true 15 --calibrate --nonlinear --no_sweep --out results_mcrae_cal
python run_diagnostics.py --glove G.txt --w2v W.bin --wordnet 20000 --wn_min_members 30 \
    --min_true 30 --calibrate --no_sweep --out results_wn_cal
python run_train.py --glove G.txt --w2v W.bin --mcrae CONCS_FEATS_concstats_brm.txt \
    --min_true 15 --iters 40 --out results_train
python run_train.py --glove G.txt --w2v W.bin --wordnet 20000 --min_true 30 \
    --iters 40 --out results_train
python run_separability.py --glove G.txt --w2v W.bin --mcrae CONCS_FEATS_concstats_brm.txt
python run_separability.py --glove G.txt --wordnet 20000 --min_true 30
python run_separability.py --w2v W.bin --wordnet 20000 --min_true 30
python make_figures.py --train results_train --diag results_mcrae_cal results_wn_cal --out figures
```

## Module map

`embeddings.py` loads vectors and builds `H` over a list of entities. `norms.py` builds long-form lexicons from McRae, Binder, and WordNet, and `truth_matrix` turns one into `T` with thresholds on positives and negatives. `criterion.py` holds the rank, projector, defect, principal angles, and separability. `sweep.py` holds ridge readouts, probes, forced zero angles, and the PCA dimension sweep. `calib.py` holds the monotone and two-layer readouts, `relations.py` the bilinear residuals, `parallel.py` the parallelogram statistic, and `train.py` the alternating least squares and the closed forms.

## Acknowledgments

The observation that an embedding layer is a linear map on one-hot inputs, and its pedagogical framing, are due entirely to Sebastian Raschka, which set this paper in motion.

## Citation
```bibtex
@misc{quigley2026compression,
      title={Faithful compression in a vector logic for formal semantics}, 
      author={Daniel Quigley},
      year={2026},
      eprint={2512.06205},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.06205}, 
}
```
