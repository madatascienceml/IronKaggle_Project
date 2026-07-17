# IronKaggle — King County House Price Prediction

## Overview

This project builds a regression model to predict house sale prices in King County (Seattle area, WA) based on property characteristics — size, location, condition, and sale details. It's part of the Ironhack Data Science & Machine Learning Bootcamp (Week 5 team project). This notebook covers the **Random Forest Regressor** track, compared against Linear Regression and KNN as baselines.

**Dataset:** [King County House Sales](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa) — 21,613 property sales from May 2014 to May 2015, 21 original features.

---

## Project Structure

```
IronKaggle_Project/
├── data/
│   └── king_country_houses_aa.csv
├── IronKaggle.ipynb
└── README.md
```

## How to Run

1. Place `king_country_houses_aa.csv` inside the `data/` folder.
2. Open `IronKaggle.ipynb` in Jupyter and run all cells in order.
3. Required libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`.

---

## Workflow

1. **Load & explore** — checked shape, dtypes, and summary statistics to get a first read on the data.
2. **Clean** — investigated every suspicious value found in `.describe()` before deciding what to do with it (see *Key Decisions* below). No missing values or duplicates were present.
3. **Feature engineering** — extracted `sale_year`/`sale_month` from `date`, converted `yr_renovated` into a clearer binary + numeric signal, and encoded `zipcode`.
4. **EDA & correlation analysis** — explored relationships between features and price, and among features, to catch multicollinearity and understand what actually drives price.
5. **Modeling** — trained Linear Regression and KNN as baselines, then Random Forest as the main model, with hyperparameter tuning via GridSearchCV.
6. **Evaluation & comparison** — compared all models on R² and MAE, and reviewed predicted-vs-actual plots.
7. **Feature importance** — extracted and interpreted which features drive the final model's predictions.

---

## Key Decisions

Every outlier or unusual value was investigated the same way: **cross-check against a related variable, weigh statistical evidence against real-world plausibility, and document the reasoning** — never dropped just because a number looked unusual in isolation.

- **`bedrooms = 33`**: physically impossible for a 1,620 sqft home (confirmed against `sqft_living`) → row dropped.
- **`bedrooms = 0`** (13 rows): initially hypothesized as studios/lofts, but `sqft_living` for these rows ranged up to 4,810 sqft, contradicting that theory → rows dropped (0.06% of the data, negligible impact).
- **`bathrooms`** (fractional values, max of 8): confirmed as a standard real-estate convention and a plausible size trend, not errors → kept as-is.
- **`grade`**: external documentation stated a 1–11 scale, but the data itself showed a consistent 3–13 range (102 rows above 11), supported by clear correlation with price and living area → treated the observed data as authoritative over outdated documentation.
- **`yr_renovated`**: mixing real years with zeros ("never renovated") would have introduced a false extreme value → replaced with `was_renovated` (binary) and `years_since_renovation`, original column dropped.
- **`sqft_above` / `sqft_basement`**: mathematically redundant with `sqft_living` (sum matches exactly in 100% of rows), but empirically tested — removing them slightly *worsened* the model (R² 0.882 → 0.879) → kept both.
- **`id`**: unique identifier, no predictive value → dropped.
- **`zipcode`**: Label Encoded rather than One-Hot, since tree-based models don't require encoding that preserves distance; `lat`/`long` were kept alongside it to retain continuous geographic signal.
- **`price` (target)**: log-transformed (`np.log1p`) before training the final model, based on the team's finding that this was the most effective preprocessing step across models — it improved R² and reduced overfitting on Random Forest as well.

---

## Results

| Model | R² | MAE |
|---|---|---|
| Linear Regression | 0.710 | $126,343.52 |
| KNN Regressor | 0.790 | $93,666.91 |
| Random Forest (Base) | 0.882 | $68,356.85 |
| Random Forest (Tuned) | 0.880 | $68,120.24 |
| **Random Forest (Tuned + log price)** | **0.887** | $68,493.53 |

**Final model: Random Forest (Tuned + log price)** — following the team's finding that log-transforming `price` (`np.log1p`, reverted with `np.expm1` before evaluation) was the most effective preprocessing step across models, we applied it to our tuned Random Forest. It improved R² (0.880 → 0.887) and reduced the train-test overfitting gap (0.098 → 0.085), at the cost of a small increase in dollar-term MAE — log-transforming optimizes for proportional error, trading some mid-range accuracy for better performance on the underrepresented high-value segment.

KNN outperforming Linear Regression during baseline testing was an early signal that the price–feature relationship isn't purely linear — which Random Forest's stronger performance later confirmed.

### Top predictive features
1. `sqft_living`
2. `grade`
3. `lat` — notably more influential than its weak linear correlation with price would suggest, revealing a non-linear relationship between location and price that Random Forest captures but simple correlation cannot.

---

## Limitations & Next Steps

- All models — Random Forest included — tend to underpredict high-value properties (>$2.5M), likely due to their underrepresentation in the dataset.
- Some overfitting persists in Random Forest (train R² ≈0.98 vs test R² ≈0.88); further regularization (e.g., stricter `max_depth`) could be explored.
- `zipcode_encoded` ranked low in importance despite representing location — worth testing alternative encodings (e.g., grouping by broader region) in future iterations.
