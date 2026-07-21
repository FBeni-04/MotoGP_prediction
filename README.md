# MotoGP and Moto2 Race Result Prediction Project

This project builds two related machine learning pipelines for predicting upcoming motorcycle Grand Prix race results:

- **MotoGP prediction model**
- **Moto2 prediction model**

Both models use historical race results, current-season data, rider form, team/bike information, DNF risk, and circuit-specific history to estimate predicted finishing order for upcoming races.

The project is designed for both:

- data analysis
- visual race prediction output in HTML or PNG format

---

# 1. Project Goal

The goal is to generate predicted finishing orders for selected upcoming races.

The current prediction target includes the next five races:

- German GP / Sachsenring
- British GP / Silverstone
- Aragon GP
- San Marino GP / Misano
- Austrian GP / Red Bull Ring

For each race, the model estimates:

- predicted rank
- expected finishing score
- predicted position if the rider finishes
- probability of finishing
- DNF probability
- DNF risk level
- recent and season form indicators

---

# 2. MotoGP Model

## MotoGP Dataset

The main MotoGP dataset is:

```text
motogp_dataset_from_Race_plus_2026_current.csv
```

This dataset is based on historical MotoGP race results and has been extended with current 2026 race results.

The 2026 results currently include:

```text
THA, BRA, USA, SPA, FRA, CAT, ITA, HUN, CZE, NED
```

The dataset contains rider-level race records with information such as:

- year
- round
- circuit ID
- rider ID
- rider name
- team
- constructor
- race finishing position
- sprint finishing position
- points
- DNF status
- season form
- career form
- circuit history
- constructor/circuit history

## MotoGP Rider List

The MotoGP prediction input can be restricted to a manually selected rider list, for example:

- Marco Bezzecchi
- Jorge Martin
- Raúl Fernández
- Ai Ogura
- Francesco Bagnaia
- Marc Márquez
- Fermín Aldeguer (he did not start in the last race)
- Alex Márquez
- Franco Morbidelli
- Fabio Di Giannantonio
- Johann Zarco (currently he has an injury)
- Diogo Moreira
- Luca Marini
- Joan Mir
- Brad Binder
- Pedro Acosta
- Maverick Vinales
- Enea Bastianini
- Fabio Quartararo
- Alex Rins
- Toprak Razgatlioglu
- Jack Miller

## Temporary Riders
- Cal Crutchlow (Currently he substitutes Zarco)
- Augusto Fernández
- Iker Lecuona
- Michele Pirro
- Lorenzo Savadori

Rider images, country flags, and team images can be configured through dictionaries in the notebook or script.

## MotoGP Constructor Handling

MotoGP uses motorcycle constructors, for example:

```text
Ducati
Aprilia
KTM
Honda
Yamaha
```

The model can use `constructor` as a categorical feature and also for constructor/circuit historical averages.

Typical categorical features for MotoGP:

```python
categorical_features = [
    "circuit_id",
    "rider_id",
    "constructor",
    "team",
]
```

## MotoGP Sprint Handling

MotoGP includes sprint races, so sprint-related features can be useful.

Example sprint features:

```text
sprint_avg_last3
sprint_avg_last5
```

These are used as rider form indicators, especially when recent race data is limited.

---

# 3. Moto2 Model

## Moto2 Dataset

The main Moto2 dataset is:

```text
moto2_dataset_from_Race2_plus_2026_current.csv
```

This dataset is based on historical Moto2 race results and has been extended with current 2026 race results.

The 2026 results currently include:

```text
THA, BRA, USA, SPA, FRA, CAT, ITA, HUN, CZE, NED
```

The dataset contains rider-level race records with information such as:

- year
- round
- circuit ID
- rider ID
- rider name
- team
- bike/chassis
- final position
- classified finish position
- points
- DNF status
- season form
- career form
- circuit-specific historical form
- bike/chassis circuit performance

## Moto2 Rider List

The Moto2 prediction input can be restricted to a manually selected rider list, for example:

- Manuel González
- Izan Guevara
- Senna Agius
- David Alonso
- Celestino Vietti
- Daniel Holgado
- Iván Ortolá
- Filip Salač
- Alonso López
- Collin Veijer
- Daniel Muñoz
- Álex Escrig
- Tony Arbolino
- Barry Baltus
- Joe Roberts
- Adrián Huertas
- José Antonio Rueda
- Deniz Öncü
- Alberto Fernández
- Arón Canet
- Zonta van den Goorbergh
- Luca Lunetta
- Ayumu Sasaki
- Sergio García
- Taiyo Furusato
- Ángel Piqueras
- Jorge Navarro
- Xabi Zurutuza
- Jacob Roulstone (temporary rider. He substitutes Aji)
- Milan Pawelec

> **_NOTE:_** During season Jorge Navarro changed team from Forward to Kalex and Xabi Zurutuza replaced him.

Rider images, country flags, and team/bike images can be configured through dictionaries in the notebook or script.

## Moto2 Bike / Chassis Handling

Moto2 uses chassis manufacturers instead of MotoGP-style constructors.

Typical values include:

```text
Kalex
Boscoscuro
Forward
```

Therefore, the Moto2 model uses the `bike` column instead of `constructor`.

Typical categorical features for Moto2:

```python
categorical_features = [
    "circuit_id",
    "rider_id",
    "bike",
    "team",
]
```

For Moto2, manufacturer/circuit averages are based on `bike`, not `constructor`.

Example:

```python
df_feat["mfr_circuit_classified_avg"] = df_feat.groupby(
    ["circuit_id", "bike"]
)["finish_position_classified"].transform(
    lambda s: s.shift(1).expanding().mean()
)
```

## Moto2 Sprint Handling

Moto2 does not use sprint races.

Therefore, sprint-related features are intentionally removed from the Moto2 model.

Do not use:

```text
sprint_pos
sprint_avg_last3
sprint_avg_last5
```

The Moto2 form score should be based only on race results, season form, career form, circuit history, and bike/chassis history.

---

# 4. Dataset Logic

Both MotoGP and Moto2 datasets use careful handling of non-standard race results.

DNF, Ret, DSQ, DNS, WD and similar statuses are not treated as normal finishing positions.

Instead:

- `finish_position_classified` is only filled when the rider is classified.
- non-classified results are handled through DNF/risk features.
- `final_position` may contain a technical fallback value for machine learning compatibility.
- DNF risk is modelled separately from race pace.

This prevents retired riders from being incorrectly interpreted as simply finishing last.

---

# 5. Model Structure

Both MotoGP and Moto2 models use the same two-stage structure.

## Stage 1: Finish Probability Classifier

The first stage predicts the probability that a rider will be classified at the finish.

It uses an ensemble of classifiers:

- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier

Output:

```text
prob_finish
prob_dnf
risk_level
```

## Stage 2: Finishing Position Regressor

The second stage predicts the expected finishing position assuming the rider finishes the race.

It uses an ensemble of regressors:

- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

Output:

```text
predicted_position_if_finished
```

---

# 6. Final Ranking Score

The final ranking is based on a combined expected position score.

The model combines:

- predicted position if classified
- recent rider form
- season form
- career form
- circuit-specific history
- constructor or bike/chassis circuit history
- DNF risk

A simplified version of the scoring logic is:

```python
pure_pace_score = (
    0.60 * predicted_position_if_finished
    + 0.40 * classified_form_score
)

dnf_risk_penalty = (
    4.0 * prob_dnf
    + 1.2 * season_dnf_rate_before
    + 0.8 * dnf_rate_last5
)

expected_position_score = pure_pace_score + dnf_risk_penalty
```

Lower `expected_position_score` means a better predicted result.

---

# 7. Feature Engineering

The models create rolling and historical features without data leakage.

Common features include:

```text
classified_avg_last3
classified_avg_last5
classified_avg_last10
dnf_rate_last3
dnf_rate_last5
dnf_rate_last10
top10_rate_last5
top10_rate_last10
season_classified_avg_before
season_dnf_rate_before
season_points_before_calc
career_classified_avg_before
career_dnf_rate_before
circuit_classified_avg
mfr_circuit_classified_avg
```

MotoGP may also include:

```text
sprint_avg_last3
sprint_avg_last5
```

Moto2 should not include sprint features.

Circuit and constructor/bike averages are shifted before use, so the current race result is not leaked into its own prediction.

---

# 8. Validation

The project validates the model using historical seasons.

Typical validation setup:

```text
Training years: up to 2024
Validation year: 2025
```

## Results of validation:

### MotoGP
| stage                           | model  | validation_year | rows | accuracy | brier_loss | roc_auc | mae_positions_only_finishers |
|---------------------------------|--------|-----------------|------|----------|------------|---------|------------------------------|
| classifier_finish_probability   | RF_cls | 2025            | 410  | 0.798    | 0.174      | 0.595   | NaN                          |
| classifier_finish_probability   | ET_cls | 2025            | 410  | 0.751    | 0.188      | 0.594   | NaN                          |
| classifier_finish_probability   | GB_cls | 2025            | 410  | 0.780    | 0.165      | 0.620   | NaN                          |
| regressor_position_if_finished  | RF_reg | 2025            | 323  | NaN      | NaN        | NaN     | 3.207                        |
| regressor_position_if_finished  | ET_reg | 2025            | 323  | NaN      | NaN        | NaN     | 3.348                        |
| regressor_position_if_finished  | GB_reg | 2025            | 323  | NaN      | NaN        | NaN     | 3.273                        |


### Moto2
| stage                           | model  | validation_year | rows | accuracy | brier_loss | roc_auc | mae_positions_only_finishers |
|---------------------------------|--------|-----------------|------|----------|------------|---------|------------------------------|
| classifier_finish_probability   | RF_cls | 2025            | 410  | 0.777    | 0.179      | 0.522   | NaN                          |
| classifier_finish_probability   | ET_cls | 2025            | 410  | 0.704    | 0.202      | 0.508   | NaN                          |
| classifier_finish_probability   | GB_cls | 2025            | 410  | 0.831    | 0.139      | 0.555   | NaN                          |
| regressor_position_if_finished  | RF_reg | 2025            | 323  | NaN      | NaN        | NaN     | 4.369                        |
| regressor_position_if_finished  | ET_reg | 2025            | 323  | NaN      | NaN        | NaN     | 4.391                        |
| regressor_position_if_finished  | GB_reg | 2025            | 323  | NaN      | NaN        | NaN     | 4.245                        |


The classifier is evaluated with:

- accuracy
- Brier loss
- ROC AUC

The finishing-position regressor is evaluated with:

- mean absolute error on classified finishers only

---


# 9. Output Columns

The main prediction table usually contains:

```text
race_name
circuit_id
round
predicted_rank
rider_full_name
constructor or bike
team
expected_position_score
pure_pace_score
predicted_position_if_finished
prob_finish
prob_dnf
dnf_risk_penalty
risk_level
season_points_before_calc
champ_pos_before
season_classified_avg_before
classified_avg_last3
classified_avg_last5
classified_avg_last10
circuit_classified_avg
mfr_circuit_classified_avg
```

For MotoGP, the output may include:

```text
constructor
sprint_avg_last5
```

For Moto2, the output should include:

```text
bike
```

and should not include sprint columns.

---

# 10. HTML Output

The notebook can generate an HTML prediction table.

The HTML output can include:

- rider images
- rider names
- country flags
- team/bike images
- team names
- expected score
- pace score
- finish probability
- DNF probability
- risk level

Recommended filenames:

```text
motogp_next5_predictions.html
moto2_next5_predictions.html
```

Example:

```python
HTML_OUT = Path("moto2_next5_predictions.html")
HTML_OUT.write_text(html, encoding="utf-8")
```

---

# 11. PNG Export

The final HTML can be converted into PNG images using Playwright.

A recommended approach is to export each race separately, producing one PNG per Grand Prix.  
This avoids creating one very large image and makes the result easier to share.

Example output folders:

```text
motogp_next5_png/
moto2_next5_png/
```

---

# 12. Moto3 Background Data

Moto3 data is not currently merged into the main Moto2 model.

It was considered for rookie riders, but the current Moto2 version remains based on Moto2 data only.  
This keeps the dataset cleaner and avoids over-weighting junior category results.

A future version could add Moto3 history as separate contextual features, such as:

```text
moto3_adjusted_career_form
moto3_win_rate
moto3_podium_rate
moto3_top10_rate
moto3_dnf_rate
```

These should be used as background rider features, not as direct Moto2 race results.

---

# 13. Notes and Limitations

This project is a prediction model, not a deterministic race simulator.

The model does not directly know about:

- weather
- injuries
- qualifying results
- penalties
- practice pace
- tyre allocation
- crashes in previous sessions
- late rider substitutions
- mechanical updates

Because of this, predictions should be interpreted as form-based estimates rather than guaranteed results.

The model can be improved later by adding:

- qualifying results
- practice/session pace
- grid position
- Moto3 career background features for Moto2 rookies
- manual injury or substitution flags
- weather and circuit condition data

---

# 14. Requirements

Typical Python dependencies:

```text
numpy
pandas
scikit-learn
IPython
beautifulsoup4
playwright
```

Install them with:

```bash
pip install numpy pandas scikit-learn beautifulsoup4 playwright
playwright install chromium
```

In Google Colab or some notebook environments, Playwright may also require:

```bash
playwright install-deps chromium
```

---

# 15. How to Run

## MotoGP

1. Place the MotoGP dataset CSV in the working directory:

```text
motogp_dataset_from_Race_plus_2026_current.csv
```

2. Set the input path:

```python
INPUT = Path("motogp_dataset_from_Race_plus_2026_current.csv")
```

3. Keep MotoGP-specific fields:

```text
constructor
sprint_avg_last3
sprint_avg_last5
```

4. Run the notebook or script.

## Moto2

1. Place the Moto2 dataset CSV in the working directory:

```text
moto2_dataset_from_Race2_plus_2026_current.csv
```

2. Set the input path:

```python
INPUT = Path("moto2_dataset_from_Race2_plus_2026_current.csv")
```

3. Use Moto2-specific fields:

```text
bike
```

4. Remove sprint features.

5. Run the notebook or script.

---

# 16. Current Project Status

Current status:

```text
MotoGP dataset prepared
MotoGP 2026 current results appended
MotoGP two-stage model working
MotoGP HTML/PNG output supported

Moto2 dataset prepared
Moto2 2026 current results appended
Moto2 sprint features removed
Moto2 bike/chassis handling adapted
Moto2 two-stage model working
Moto2 HTML/PNG output supported
```

The next useful improvement would be to add qualifying/session data once available for each upcoming race.

# 17. Pictures of the results
<img width="1615" height="1174" alt="german_gp_sachsenring (1)" src="https://github.com/user-attachments/assets/a4fe9759-9e4c-4b3c-830e-8a19f6df83dd" />
<img width="1558" height="1108" alt="german_gp_sachsenring (2)" src="https://github.com/user-attachments/assets/ca6f33df-7cff-4d64-b274-99c46d93ac52" />

> **_NOTE:_** These are PNG outputs. Now you can only see the top 5 riders of the next race (German GP/Sachsenring).


