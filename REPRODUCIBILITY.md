# Reproducibility notes

The experiments were run on Kaggle GPUs, notebook by notebook. While turning them into this repository,
the code was checked against the thesis text. This file lists every difference found, so that anyone
rerunning the experiments knows what to expect. The numbers in the thesis and in `results/*.csv` are the
original ones and have not been changed.

## 0. Traceability of the reported results

The experiments ran on Kaggle, and most notebooks were exported without their cell outputs. This table
shows which reported numbers can be checked against a file in this repository.

| Thesis result | Evidence in this repository | Status |
|---|---|---|
| Tables 1–2, dataset statistics | recomputed by `scripts/dataset_stats.py` | reproduced exactly (Yelp: see §3) |
| Table 9, GMF on Yelp | saved output of `notebooks/04_ncf_yelp.ipynb` | matches exactly |
| Table 11, MLP on Yelp | saved output of `notebooks/04_ncf_yelp.ipynb` | matches exactly |
| Table 13, NeuMF on Yelp | saved output of `notebooks/04_ncf_yelp.ipynb` | matches exactly |
| Figs. 20–21, NeuMF curves on Yelp | logged history hard-coded in `04_ncf_yelp.ipynb` | matches the saved run |
| Tables 4–5, ItemKNN | code in `01_itemknn.ipynb`, no saved test output | not traceable; the scoring issue is in §1 |
| Tables 6–7, MF-BPR | no notebook (not exported) | not traceable |
| Tables 8, 10, 12, 16, MovieLens NCF models | code in notebooks 02–03, no saved output | not traceable |
| Table 14, ConvNCF on MovieLens | code in `05_convncf_movielens.ipynb`, no saved output | not traceable |
| Table 15, ConvNCF on Yelp | the local notebook differs from the reported run (§4) | not traceable |
| Table 17, NeuMF-pre on Yelp | the learning-rate grid was not exported (§5) | not traceable |

"Not traceable" means the code that produced the number is present (except where noted), but its
output was not saved, so the value cannot be confirmed from this repository alone. The numbers are
reported as they appear in the thesis.

## 1. ItemKNN scoring and tie-breaking (affects Tables 4, 5, 18, 19)

The thesis defines the ItemKNN score as the sum of similarities between the candidate item and its
neighbours in the user's history (Section 4.1.1). `notebooks/01_itemknn.ipynb` uses a normalised weighted
average instead:

```python
pred = sum(s * r) / sum(abs(s))       # r = 1 for implicit feedback
```

With plain cosine similarity every `s ≥ 0`, so `pred = 1.0` for **every** candidate that shares at least
one user with the history. The evaluator then ranks the true item by counting only strictly higher
scores (`rank = np.sum(preds > gt_score)`), which breaks every tie in favour of the held-out item. The
MovieLens part uses adjusted cosine, where similarities can be negative, so it is less affected, but the
scoring rule is still not the one described in the thesis.

`src/cfcompare/knn.py` implements the formula from the thesis, and `src/cfcompare/metrics.py` breaks ties
pessimistically. Re-evaluated with the same split and protocol (`python scripts/run_experiment.py --model itemknn`;
full outputs in `results/reproduced/`). The default K grid is the thesis grid (up to 200). The extended grid
row uses `--k-grid 5 10 20 50 100 200 500 1000`:

| Dataset | Selected K | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Thesis HR@10 / NDCG@10 |
|---|---|---|---|---|---|---|
| MovieLens-1M | 50 | 0.6570 | 0.3848 | 0.8217 | 0.4266 | 0.5204 / 0.3022 |
| Yelp, thesis grid (K ≤ 200) | 200 | 0.6666 | 0.4762 | 0.6951 | 0.4836 | 0.7301 / 0.4747 |
| Yelp, extended grid (K ≤ 1000) | 1000 | 0.7433 | 0.5113 | 0.8520 | 0.5389 | |

Consequences for the conclusions: on Yelp ItemKNN remains the best model with the thesis grid (NDCG@10
0.476 vs 0.430 for NeuMF-pre), and by a larger margin once K may exceed 200. On MovieLens a correctly scored ItemKNN is roughly level with GMF (NDCG@10
0.385 vs 0.385) instead of trailing the neural models. That strengthens the thesis's overall conclusion
that well-tuned classical baselines are hard to beat.

## 2. NeuMF-pre output-layer initialisation (Tables 16, 17)

`get_weights()` in notebooks 02–04 builds the NeuMF prediction layer as
`concat([α·h_GMF, (1-α)·h_MLP])`. In `build_NEUMF` the fused vector is `concat([mlp_vector, gmf_vector])`,
MLP (8 dims) first. The shapes still match (72 × 1), so Keras raises no error, but the pretrained output
weights start misaligned with their inputs. Fine-tuning partly compensates.
`src/cfcompare/ncf.py::load_pretrained` uses the correct order.

## 3. Yelp data version (Table 2)

`yelp.rating`, the preprocessed file of He et al. (2016), contains 731,671 rows from 25,677 users and
25,815 items, including 34,806 duplicated (user, item) pairs. Table 2 reports 24,930 users, 25,799 items
and 690,381 interactions, which is the file after de-duplication and keeping users with at least 10
distinct items. All Yelp experiments (notebooks 01, 04, 06) were trained on the **raw** file. Their logs
show 25,677 users and 680,317 training rows. `python scripts/dataset_stats.py` prints both versions.

## 4. ConvNCF details

* **Filters.** Section 4.1.6 describes three convolutional layers with {32, 32, 64} filters. All ConvNCF
  notebooks use `conv_filters=(32, 64, 64)`, with 2 × 2 kernels, 2 × 2 max-pooling after every
  convolution and a 128-unit dense layer.
* **Interaction-map normalisation.** The text says the outer-product map is normalised before the CNN. No
  notebook contains such a step. The code feeds the raw outer product to the first convolution.
* **Yelp run.** The local `06_convncf_yelp.ipynb` is a later edit: it uses `Adam(0.05)` and a fixed
  L2 = 1e-4, while §5.6.2 describes `Adagrad(0.05)` with an L2 grid. The L2 value the grid selected is
  not recorded, so `src` defaults to the MovieLens optimum (0.1 / 0.1 / 1e-3). Override it with
  `--l2-emb/--l2-conv/--l2-out`.

## 5. Experiments without a notebook in this repository

The final versions of these runs were Kaggle notebooks that were not exported:

* **MF-BPR K-analysis** (Tables 6–7): use `scripts/run_experiment.py --model mf-bpr`.
* **NeuMF-pre learning-rate grid on Yelp** (Table 17, best lr = 1e-4): the `src` config uses 1e-4.
* **Dataset statistics** (Tables 1–2): reproduced by `scripts/dataset_stats.py`.

## 6. Smaller differences between the notebooks and `src`

| | Notebooks | `src/cfcompare` |
|---|---|---|
| Candidate negatives | re-sampled for each K and each evaluation | sampled once per split and seed, shared across K and models |
| Ties in ranking | depends on the notebook | always pessimistic |
| Point-wise negatives | sampled once before training | same (kept for fidelity) |
| Random seeds | mostly unset | fixed (`--seed`, default 42) |

Because negatives are sampled, expect small run-to-run variation in all metrics.
