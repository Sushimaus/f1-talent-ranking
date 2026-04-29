"""
EFB350 - F1 Driver Talent Replication & Extension
Replicates Eichenberger & Stadelmann (2009) and extends to 2025.

HOW TO RUN:
    pip install requests pandas numpy statsmodels tqdm
    python f1_replication.py

The script will:
    1. Pull all race results from Ergast API (1950-2025)
    2. Build the dataset (classifications, dropouts, car-year dummies)
    3. Run the OLS regression with fixed effects
    4. Print the driver talent ranking
"""

import requests
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm
import time
import json
import os

# ─────────────────────────────────────────────
# STEP 1: PULL DATA FROM ERGAST API
# ─────────────────────────────────────────────
# Ergast gives us every race result ever — driver, constructor, position, status
# We loop year by year and race by race to build our dataset.

CACHE_FILE = "ergast_raw.json"  # saves to disk so you don't re-download every time

def fetch_all_results(start_year=1950, end_year=2025):
    """Fetch all race results from Ergast API, with local caching."""
    
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        with open(CACHE_FILE) as f:
            return json.load(f)
    
    all_rows = []
    print(f"Fetching F1 results {start_year}–{end_year} from Ergast API...")
    
    for year in tqdm(range(start_year, end_year + 1)):
        offset = 0
        limit = 100  # Ergast max per request
        
        while True:
            url = (
                f"https://api.jolpi.ca/ergast/f1/{year}/results.json"
                f"?limit={limit}&offset={offset}"
            )
            try:
                for attempt in range(5):
                    resp = requests.get(url, timeout=15)
                    if resp.status_code == 429:
                        time.sleep(3 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    break
                data = resp.json()
            except Exception as e:
                print(f"  Error fetching {year} offset {offset}: {e}")
                break
            
            races = data["MRData"]["RaceTable"]["Races"]
            if not races:
                break
            
            for race in races:
                race_name  = race["raceName"]
                round_num  = int(race["round"])
                circuit    = race["Circuit"]["circuitId"]
                
                for result in race["Results"]:
                    driver_id    = result["Driver"]["driverId"]
                    driver_name  = (result["Driver"]["givenName"] + " "
                                    + result["Driver"]["familyName"])
                    nationality  = result["Driver"]["nationality"]
                    constructor  = result["Constructor"]["constructorId"]
                    grid         = int(result.get("grid", 0))
                    
                    # Finished position (None if DNF)
                    pos_str = result.get("position", None)
                    finished_pos = int(pos_str) if pos_str and pos_str.isdigit() else None
                    
                    # Status tells us WHY they stopped
                    status = result.get("status", "")
                    
                    all_rows.append({
                        "year":         year,
                        "round":        round_num,
                        "race_name":    race_name,
                        "circuit":      circuit,
                        "driver_id":    driver_id,
                        "driver_name":  driver_name,
                        "nationality":  nationality,
                        "constructor":  constructor,
                        "grid":         grid,
                        "finished_pos": finished_pos,
                        "status":       status,
                    })
            
            offset += limit
            total = int(data["MRData"]["total"])
            if offset >= total:
                break
            
            time.sleep(0.2)  # be nice to the API
    
    print(f"Fetched {len(all_rows)} driver-race observations.")
    with open(CACHE_FILE, "w") as f:
        json.dump(all_rows, f)
    
    return all_rows


# ─────────────────────────────────────────────
# STEP 2: CLASSIFY DROPOUTS
# ─────────────────────────────────────────────
# The paper distinguishes:
#   - Finished: use actual classification
#   - Technical dropout: engine, gearbox, hydraulics etc. → dummy control
#   - Human dropout: accident, collision, disqualified → counterfactual position

# These keywords in the "status" field = technical dropout
TECHNICAL_KEYWORDS = [
    "engine", "gearbox", "transmission", "hydraulics", "electrical",
    "brakes", "suspension", "wheel", "tyre", "puncture", "fuel",
    "overheating", "oil", "cooling", "clutch", "driveshaft", "throttle",
    "exhaust", "fire", "mechanical", "power", "turbo", "alternator",
    "battery", "electronics", "water", "vibrations", "retired"
]

HUMAN_KEYWORDS = [
    "accident", "collision", "spun off", "disqualified", "withdrew",
    "did not qualify", "did not start", "dnf"
]

def classify_status(status: str) -> str:
    """Return 'finished', 'technical', or 'human' dropout."""
    s = status.lower()
    if s == "finished" or s.startswith("+"):
        return "finished"
    for kw in TECHNICAL_KEYWORDS:
        if kw in s:
            return "technical"
    for kw in HUMAN_KEYWORDS:
        if kw in s:
            return "human"
    return "technical"  # default unknown to technical (conservative)


# ─────────────────────────────────────────────
# STEP 3: BUILD THE ANALYSIS DATASET
# ─────────────────────────────────────────────

def build_dataset(raw_rows):
    """
    Turn raw API data into the regression-ready dataset.
    Key variable: classification (lower = better, worst = last + penalty)
    """
    df = pd.DataFrame(raw_rows)
    df["dropout_type"] = df["status"].apply(classify_status)
    
    # ── Per-race calculations ──
    race_groups = df.groupby(["year", "round"])
    
    processed = []
    
    for (year, round_num), race_df in race_groups:
        race_df = race_df.copy()
        
        n_total    = len(race_df)
        finishers  = race_df[race_df["dropout_type"] == "finished"]
        n_finished = len(finishers)
        n_dropouts = n_total - n_finished
        
        # Counterfactual for human dropouts (paper's formula):
        # last_finisher_pos + n_total_dropouts / 2
        # If no one finished (very rare), use n_total as base
        last_pos = n_finished if n_finished > 0 else n_total
        human_dropout_pos = last_pos + n_dropouts / 2
        
        for _, row in race_df.iterrows():
            dtype = row["dropout_type"]
            
            if dtype == "finished":
                classification = row["finished_pos"]
                is_tech_dropout = 0
            elif dtype == "technical":
                # Technical dropouts: assign counterfactual + flag dummy
                classification = human_dropout_pos  # treated same positionally
                is_tech_dropout = 1
            else:  # human
                classification = human_dropout_pos
                is_tech_dropout = 0
            
            # Car-year dummy key: e.g. "ferrari_2010"
            car_year = f"{row['constructor']}_{year}"
            
            processed.append({
                "year":           year,
                "round":          round_num,
                "driver_id":      row["driver_id"],
                "driver_name":    row["driver_name"],
                "nationality":    row["nationality"],
                "constructor":    row["constructor"],
                "car_year":       car_year,
                "classification": float(classification),
                "n_finished":     n_finished,
                "is_tech_dropout": is_tech_dropout,
                "dropout_type":   dtype,
            })
    
    df_out = pd.DataFrame(processed)
    
    # ── Filter: only drivers with at least 1 points finish ──
    # Approximation: at least 1 top-10 finish (modern points) or top-6 (old)
    # Simpler: keep drivers with at least 1 "finished" entry in top 10
    top10 = df_out[(df_out["dropout_type"] == "finished") & 
                   (df_out["classification"] <= 10)]
    drivers_with_points = set(top10["driver_id"].unique())
    df_out = df_out[df_out["driver_id"].isin(drivers_with_points)]
    
    # ── Filter: only drivers with 40+ race starts (paper's threshold) ──
    race_counts = df_out.groupby("driver_id")["round"].count()
    qualified_drivers = race_counts[race_counts >= 40].index
    
    print(f"\nDataset summary:")
    print(f"  Total driver-race rows: {len(df_out)}")
    print(f"  Drivers with points:    {len(drivers_with_points)}")
    print(f"  Drivers with 40+ races: {len(qualified_drivers)}")
    
    return df_out, qualified_drivers


# ─────────────────────────────────────────────
# STEP 4: RUN THE OLS REGRESSION
# ─────────────────────────────────────────────
# The model is:
#   classification = Σ(α_i × driver_dummy_i)
#                  + Σ(γ_s × car_year_dummy_s)
#                  + β1×n_finished + β2×is_tech_dropout
#                  + error
#
# We use statsmodels OLS with dummy variables.
# Driver dummies are the KEY output — their coefficients = talent estimates.
# Lower coefficient = finishes better on average = more talented.

def run_regression(df, qualified_drivers):
    """Run OLS with driver and car-year fixed effects."""
    
    # Work only with 40+ race drivers for the ranking,
    # but keep all observations (including teammates) for identification
    df_reg = df.copy()
    
    # Create dummy variables
    print("\nCreating dummy variables (this may take a moment)...")
    
    # Driver dummies
    driver_dummies = pd.get_dummies(df_reg["driver_id"], prefix="D", drop_first=False)
    
    # Car-year dummies
    car_year_dummies = pd.get_dummies(df_reg["car_year"], prefix="CY", drop_first=True)
    
    # Combine
    X = pd.concat([driver_dummies, car_year_dummies, 
                   df_reg[["n_finished", "is_tech_dropout"]]], axis=1)
    y = df_reg["classification"]
    
    # Drop one driver dummy to avoid perfect multicollinearity
    # (we drop the last alphabetically — doesn't affect relative rankings)
    driver_cols = [c for c in X.columns if c.startswith("D_")]
    X = X.drop(columns=[driver_cols[-1]])  # drop reference driver
    
    print(f"  Running OLS with {X.shape[1]} variables on {len(y)} observations...")
    X = X.astype(float)

    import statsmodels.api as sm
    model = sm.OLS(y, X).fit()
    
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  Adj R² = {model.rsquared_adj:.4f}")
    
    return model, driver_cols, X


# ─────────────────────────────────────────────
# STEP 5: EXTRACT AND PRINT THE TALENT RANKING
# ─────────────────────────────────────────────

def build_ranking(model, driver_cols, df, qualified_drivers):
    """Extract driver talent coefficients and build the ranking table."""
    
    # Get all driver coefficients from the model
    results = []
    for col in driver_cols[:-1]:  # last one was dropped (reference)
        driver_id = col[2:]  # strip "D_" prefix
        if col not in model.params.index:
            continue
        coef = model.params[col]
        se   = model.bse[col]
        results.append({
            "driver_id": driver_id,
            "talent_coef": coef,
            "std_error":   se,
        })
    
    ranking_df = pd.DataFrame(results)
    
    # Add driver names and race counts
    driver_info = df.groupby("driver_id").agg(
        driver_name=("driver_name", "first"),
        nationality=("nationality", "first"),
        n_races=("round", "count"),
        years_active=("year", lambda x: f"{x.min()}–{x.max()}")
    ).reset_index()
    
    ranking_df = ranking_df.merge(driver_info, on="driver_id", how="left")
    
    # Filter to 40+ race drivers
    ranking_df = ranking_df[ranking_df["driver_id"].isin(qualified_drivers)]
    
    # Sort: lower coefficient = better
    ranking_df = ranking_df.sort_values("talent_coef").reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1
    
    return ranking_df


# ─────────────────────────────────────────────
# STEP 6: SPLIT-ERA COMPARISON
# ─────────────────────────────────────────────
# Extension: compare original era (1950-2006) vs modern era (2007-2025)

def era_comparison(df, qualified_drivers, model, driver_cols):
    """Show how rankings differ across eras."""
    
    original_drivers = df[df["year"] <= 2006]["driver_id"].unique()
    modern_drivers   = df[df["year"] >= 2007]["driver_id"].unique()
    
    print("\n" + "="*60)
    print("ERA COMPARISON")
    print("="*60)
    print(f"Drivers active 1950–2006: {len(original_drivers)}")
    print(f"Drivers with data 2007+:  {len(modern_drivers)}")
    print(f"Drivers spanning both:    "
          f"{len(set(original_drivers) & set(modern_drivers))}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # 1. Fetch data
    raw = fetch_all_results(start_year=1950, end_year=2025)
    
    # 2. Build dataset
    df, qualified_drivers = build_dataset(raw)
    
    # Save the cleaned dataset for inspection
    df.to_csv("f1_dataset_clean.csv", index=False)
    print("\nClean dataset saved to f1_dataset_clean.csv")
    
    # 3. Run regression
    model, driver_cols, X = run_regression(df, qualified_drivers)

    # 3b. Reconstruction & residuals
    X_arr = X.values
    betas = model.params.values

    y_hat = X_arr @ betas
    resid = df.loc[X.index, "classification"].values - y_hat

    df.loc[X.index, "predicted_pos"] = y_hat
    df.loc[X.index, "residual"]      = resid

    df[["driver_name", "year", "round", "classification",
        "predicted_pos", "residual"]].to_csv("reconstruction.csv", index=False)
    print("\nReconstruction saved to reconstruction.csv")

    # 4. Build ranking
    ranking = build_ranking(model, driver_cols, df, qualified_drivers)
    
    # 5. Print TOP 30
    print("\n" + "="*60)
    print("ALL-TIME F1 DRIVER TALENT RANKING (1950–2025)")
    print("Lower coefficient = better driver")
    print("="*60)
    
    top30 = ranking.head(30)
    for _, row in top30.iterrows():
        print(f"  #{int(row['rank']):>3}  {row['driver_name']:<25} "
              f"({row['nationality']:<15})  "
              f"coef={row['talent_coef']:>7.3f}  "
              f"SE={row['std_error']:.3f}  "
              f"races={int(row['n_races'])}  "
              f"{row['years_active']}")
    
    # 6. Save full ranking to CSV
    ranking.to_csv("f1_talent_ranking.csv", index=False)
    print(f"\nFull ranking saved to f1_talent_ranking.csv")
    
    # 7. Modern era spotlight (2007+)
    print("\n" + "="*60)
    print("MODERN ERA SPOTLIGHT (drivers active 2007–2025)")
    print("="*60)
    modern_ids = set(df[df["year"] >= 2007]["driver_id"].unique())
    modern_ranking = ranking[ranking["driver_id"].isin(modern_ids)].head(20)
    for _, row in modern_ranking.iterrows():
        print(f"  #{int(row['rank']):>3} (all-time)  {row['driver_name']:<25}  "
              f"coef={row['talent_coef']:>7.3f}  {row['years_active']}")
    
    era_comparison(df, qualified_drivers, model, driver_cols)

    # ─────────────────────────────────────────────
    # REPLICATION: 1950–2006 (original study period)
    # ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("REPLICATION: 1950–2006 (original study period)")
    print("="*60)

    raw_06 = [r for r in raw if r["year"] <= 2006]
    df_06, qd_06 = build_dataset(raw_06)
    model_06, dc_06, X_06 = run_regression(df_06, qd_06)
    ranking_06 = build_ranking(model_06, dc_06, df_06, qd_06)

    print("\n" + "="*60)
    print("F1 DRIVER TALENT RANKING — 1950–2006 REPLICATION")
    print("Lower coefficient = better driver")
    print("="*60)
    for _, row in ranking_06.head(30).iterrows():
        print(f"  #{int(row['rank']):>3}  {row['driver_name']:<25} "
              f"({row['nationality']:<15})  "
              f"coef={row['talent_coef']:>7.3f}  "
              f"SE={row['std_error']:.3f}  "
              f"races={int(row['n_races'])}  "
              f"{row['years_active']}")

    ranking_06.to_csv("f1_talent_ranking_1950_2006.csv", index=False)
    print(f"\n1950–2006 ranking saved to f1_talent_ranking_1950_2006.csv")

    # ─────────────────────────────────────────────
    # SIDE-BY-SIDE COMPARISON: 1950–2006 vs 1950–2025 (top 20)
    # ─────────────────────────────────────────────
    print("\n" + "="*80)
    print("TOP 20 COMPARISON: 1950–2006 replication  vs  1950–2025 extension")
    print("="*80)
    print(f"  {'Rank':<5}  {'1950–2006 Driver':<25}  {'Coef':>7}    "
          f"{'Rank':<5}  {'1950–2025 Driver':<25}  {'Coef':>7}")
    print(f"  {'-'*4}  {'-'*24}  {'-'*7}    {'-'*4}  {'-'*24}  {'-'*7}")

    top20_06  = ranking_06.head(20).reset_index(drop=True)
    top20_all = ranking.head(20).reset_index(drop=True)
    for i in range(20):
        r06  = top20_06.iloc[i]
        rall = top20_all.iloc[i]
        print(f"  #{int(r06['rank']):<4}  {r06['driver_name']:<25}  "
              f"{r06['talent_coef']:>7.3f}    "
              f"#{int(rall['rank']):<4}  {rall['driver_name']:<25}  "
              f"{rall['talent_coef']:>7.3f}")

    return ranking, ranking_06, model, model_06


if __name__ == "__main__":
    ranking, ranking_06, model, model_06 = main()
