# Data

The datasets are not included in this repository because their licences do not allow redistribution.
Download them yourself and place them in this folder under these names:

| File | Dataset | Source |
|---|---|---|
| `ml-1m.ratings.dat` | MovieLens-1M: 1,000,209 ratings, 6,040 users, 3,706 movies | [GroupLens](https://grouplens.org/datasets/movielens/1m/). Download `ml-1m.zip` and rename `ratings.dat`. |
| `yelp.rating` | Yelp: 731,671 interactions, 25,677 users, 25,815 businesses | Preprocessed Yelp file from He et al. (SIGIR 2016), derived from the [Yelp Open Dataset](https://www.yelp.com/dataset). Available in the authors' repository [hexiangnan/sigir16-eals](https://github.com/hexiangnan/sigir16-eals) under `data/`. |

```
data/
├── ml-1m.ratings.dat   # user::item::rating::timestamp
└── yelp.rating         # user<TAB>item<TAB>rating<TAB>timestamp  (0-based ids)
```

Download commands:

```bash
curl -L -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -p ml-1m.zip ml-1m/ratings.dat > data/ml-1m.ratings.dat
curl -L -o data/yelp.rating https://raw.githubusercontent.com/hexiangnan/sigir16-eals/master/data/yelp.rating
```

MD5 checksums of the files used in the thesis. The Yelp file is byte-identical to the one in the
sigir16-eals repository:

```
a89aa3591bc97d6d4e0c89459ff39362  ml-1m.ratings.dat
5d73d508fbbb5bb2f93fdaad1ef613c0  yelp.rating
```

## Preprocessing

All ratings are converted to implicit feedback (`y_ui = 1` if user *u* rated item *i*). For each user,
interactions are sorted by timestamp:

* **test**: the most recent interaction;
* **validation**: the second most recent interaction;
* **train**: all remaining interactions.

At evaluation time the held-out item is ranked against 99 (test) or 100 (validation) sampled items the
user has not interacted with. See `src/cfcompare/data.py` and `src/cfcompare/metrics.py`.

**Yelp duplicates.** `yelp.rating` contains 34,806 repeated (user, item) pairs. Table 2 of the thesis
reports the statistics after removing them and keeping users with at least 10 distinct items
(24,930 users, 25,799 items, 690,381 interactions). The experiments ran on the raw file.
`python scripts/dataset_stats.py` prints both versions.
