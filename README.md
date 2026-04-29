# F1 Driver Talent Ranking

Replication and extension of **Eichenberger & Stadelmann (2009)**: *"Who is the best Formula 1 driver? An economic approach to evaluating talent"*.

The model ranks all F1 drivers by raw talent, controlling for car quality and era via OLS with fixed effects. Two rankings are produced:
- **1950–2006** — same period as the original study (for direct comparison)
- **1950–2025** — full extension to the present day

---

## How to Run

```bash
pip install requests pandas numpy statsmodels tqdm
python3 f1_replication.py
```

On first run the script fetches ~76 years of race data from the [Jolpica API](https://api.jolpi.ca/) (~12 min). Results are cached to `ergast_raw.json` — subsequent runs are instant.

The script prints both rankings and a side-by-side top-20 comparison table to the console.

---

## Output Files

| File | Description |
|------|-------------|
| `ergast_raw.json` | Raw API cache (driver-race rows, 1950–2025) |
| `f1_dataset_clean.csv` | Processed dataset with dropout classifications and counterfactual positions |
| `f1_talent_ranking_1950_2006.csv` | Full ranked list — **1950–2006 replication** (original study period) |
| `f1_talent_ranking.csv` | Full ranked list — **1950–2025 extension** |
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

### 4. Ranking
Drivers sorted ascending by `α_i` (lower = more talented). Only drivers with 40+ race starts and at least one top-10 finish are ranked. The regression is run separately for each period so coefficients are internally consistent.

---

## Top 20 Comparison

> **Note on the original E&S (2009) column:** Rankings are as reported in Eichenberger & Stadelmann (2009). Drivers marked with `—` had fewer than 40 starts under their threshold and were not ranked.

| Rank | Original E&S (2009) | 1950–2006 Replication (coef) | 1950–2025 Extension (coef) |
|-----:|---------------------|------------------------------|----------------------------|
| 1 | Juan Fangio | Juan Fangio (−6.37) | Juan Fangio (−6.38) |
| 2 | Jim Clark | Jim Clark (−5.55) | Max Verstappen (−6.13) |
| 3 | Stirling Moss | Emerson Fittipaldi (−5.36) | Jim Clark (−5.56) |
| 4 | Jackie Stewart | Jackie Stewart (−5.21) | Charles Leclerc (−5.49) |
| 5 | Emerson Fittipaldi | Mike Hawthorn (−4.95) | Lando Norris (−5.45) |
| 6 | Jochen Rindt | Fernando Alonso (−4.95) | Emerson Fittipaldi (−5.35) |
| 7 | Alain Prost | Alain Prost (−4.72) | Jackie Stewart (−5.19) |
| 8 | Niki Lauda | Dan Gurney (−4.68) | Fernando Alonso (−5.06) |
| 9 | Ayrton Senna | Michael Schumacher (−4.53) | Mike Hawthorn (−4.94) |
| 10 | Nelson Piquet | Bruce McLaren (−4.29) | Carlos Sainz (−4.84) |
| 11 | Mike Hawthorn | Jochen Rindt (−4.25) | Felipe Nasr (−4.77) |
| 12 | Dan Gurney | Ronnie Peterson (−4.12) | Alain Prost (−4.75) |
| 13 | Bruce McLaren | Elio de Angelis (−4.11) | Sebastian Vettel (−4.70) |
| 14 | Ronnie Peterson | John Watson (−3.96) | Dan Gurney (−4.67) |
| 15 | Carlos Reutemann | Denny Hulme (−3.96) | Nico Rosberg (−4.57) |
| 16 | Gilles Villeneuve | Carlos Reutemann (−3.87) | Alexander Albon (−4.45) |
| 17 | Mario Andretti | Stirling Moss (−3.86) | Oscar Piastri (−4.43) |
| 18 | Jack Brabham | Jacky Ickx (−3.86) | Kamui Kobayashi (−4.41) |
| 19 | Denny Hulme | James Hunt (−3.84) | Daniel Ricciardo (−4.38) |
| 20 | Michael Schumacher | Jean-Pierre Beltoise (−3.77) | Lewis Hamilton (−4.36) |

---

## Key Observations

- **Fangio dominates both eras** — his coefficient is the most negative in the dataset despite only 58 races, a testament to near-perfect results in under-represented equipment
- **Modern drivers rank higher in the full extension** (e.g. Verstappen #2, Leclerc #4) because the 2007–2025 era adds more data and more competitive teammates to anchor the car-year controls
- **Cross-era comparability** is limited: car-year dummies control for within-dataset car quality but cannot fully equalise field strength differences across decades

---

## Limitations

- **Partial API data**: rate limiting during fetch may have dropped a small number of races for high-volume seasons; results are directionally reliable but not perfectly complete
- **Points-finish filter**: the "at least one top-10 finish" proxy is an approximation — the original paper used era-accurate points rules (top 6 before 2003)
- **Counterfactual tie**: all dropouts in a race receive the same counterfactual position; the original paper may have distinguished their ordinal rank among retirees
- **40-race threshold**: drivers with fewer starts are excluded even if individually dominant
- **Relative coefficients**: all values are relative to the omitted reference driver — absolute values are not meaningful, only the ordering and differences
