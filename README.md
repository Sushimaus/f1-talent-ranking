# F1 Driver Talent Ranking

Replication and extension of **Eichenberger & Stadelmann (2009)**: *"Who is the best Formula 1 driver? An economic approach to evaluating talent"*.

The model ranks all F1 drivers from 1950–2025 by raw talent, controlling for car quality and era via OLS with fixed effects.

---

## How to Run

```bash
pip install requests pandas numpy statsmodels tqdm
python3 f1_replication.py
```

On first run the script fetches ~76 years of race data from the [Jolpica API](https://api.jolpi.ca/) (~12 min). Results are cached to `ergast_raw.json` — subsequent runs are instant.

---

## Output Files

| File | Description |
|------|-------------|
| `ergast_raw.json` | Raw API cache (driver-race rows, 1950–2025) |
| `f1_dataset_clean.csv` | Processed dataset with dropout classifications and counterfactual positions |
| `f1_talent_ranking.csv` | Full ranked list of drivers (176 drivers with 40+ starts) |
| `reconstruction.csv` | Per-race actual position, model-predicted position, and residual |

---

## Methodology

### 1. Data
- Source: Jolpica API (successor to the now-offline Ergast API)
- Unit of observation: one row per driver per race entry (~25,873 rows)

### 2. Dropout Classification
Each race result is classified as:
- **Finished** — actual finishing position is used
- **Technical dropout** (engine, gearbox, hydraulics, etc.) — assigned a counterfactual position; flagged with `is_tech_dropout = 1`
- **Human dropout** (accident, collision, disqualification) — assigned the same counterfactual position, no flag

Counterfactual position: `last_finisher_pos + n_dropouts / 2`

### 3. OLS Regression

```
classification_it = Σ α_i · DriverDummy_i
                  + Σ γ_s · CarYearDummy_s
                  + β₁ · n_finished_t
                  + β₂ · is_tech_dropout_it
                  + ε_it
```

- **Driver dummies** (`α_i`): the talent estimates — lower = finishes better after controlling for car
- **Car-year dummies** (`γ_s`): one per constructor-season (e.g. `red_bull_2023`) — absorbs car quality
- **n_finished**: controls for field size variation across eras
- **is_tech_dropout**: controls for mechanical retirements beyond the driver's control
- 1,552 variables on 24,929 observations; **R² = 0.60**

### 4. Ranking
Drivers sorted ascending by `α_i` (lower = more talented). Only drivers with 40+ race starts are ranked.

---

## Top 10 Results

| Rank | Driver | Coef | Races | Era |
|------|--------|------|-------|-----|
| 1 | Juan Fangio | −6.38 | 58 | 1950–1958 |
| 2 | Max Verstappen | −6.13 | 233 | 2015–2025 |
| 3 | Jim Clark | −5.56 | 73 | 1960–1968 |
| 4 | Charles Leclerc | −5.49 | 173 | 2018–2025 |
| 5 | Lando Norris | −5.45 | 152 | 2019–2025 |
| 6 | Emerson Fittipaldi | −5.35 | 146 | 1970–1980 |
| 7 | Jackie Stewart | −5.19 | 100 | 1965–1973 |
| 8 | Fernando Alonso | −5.06 | 428 | 2001–2025 |
| 12 | Alain Prost | −4.75 | 202 | 1980–1993 |
| 20 | Lewis Hamilton | −4.36 | 380 | 2007–2025 |

---

## Limitations

- **Partial API data**: rate limiting during fetch may have dropped a small number of races for high-volume seasons; results are directionally reliable but not perfectly complete
- **Cross-era comparability**: car-year dummies control for within-dataset car quality but cannot fully account for overall field competitiveness differences across decades
- **40-race threshold**: drivers with fewer starts are excluded even if individually dominant
- **Relative coefficients**: all values are relative to the omitted reference driver — absolute values are not meaningful, only the ordering and differences
