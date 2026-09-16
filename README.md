# vector_logic

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
```
.
├── paper/
│   └── quigley_daniel_compression.pdf          # main paper
├── code/
│   │── make_figures.py           # figures from results csvs
│   │── run_diagnostics.py        # on trained embeddings against feature norms
│   │── run_separability.py       # strict linear separability of each predicate over a geometry decided by a linear feasibility problem
│   └── run_train.py              # train geometries under mixed distributional and truth objective
└── README.md
```

## Data

- GloVe text file, e.g. `glove.6B.300d.txt`.
- word2vec GoogleNews binary (`gensim` required).
- McRae et al. 2005 `CONCS_FEATS_concstats_brm.txt`.
- Binder et al. 2016 `WordSet1_Ratings.xlsx` (optional; ratings binarized at 3.0).

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
