

# 🏎️ F1 Winner Mexico GP 2025 Prediction

This repository contains an **end-to-end machine learning pipeline** for predicting Formula 1 race winners, using real race data from the **OpenF1 API**.
It automatically fetches the latest 2025 season data up to the **Austin Grand Prix**, builds structured features for each driver and team, trains a predictive model, and outputs **probabilities for the next race — the Mexico GP**.


## 🧠 Workflow Overview

### 1. Data Fetching — `load_dataset.py`

Fetches all race data from the 2025 season **up to Austin (COTA)** and stores it locally:

```bash
python load_dataset.py
```

This script:

* Calls OpenF1 API endpoints for sessions, results, laps, pits, stints, and weather.
* Saves data under `/openf1_2025_until_austin/`.
* Creates a clean dataset of race-level features:
  `features_up_to_austin_for_mexico.csv`

---

### 2. Model Training & Prediction — `model_predict.py`

Trains a predictive model on all past races and forecasts who will win the next race (Mexico GP):

```bash
python model_predict.py
```

Outputs:

* `prediction_mexico_probs.csv` → ranked probability of winning for each driver.
* `feature_importance.csv` → feature weights for interpretability.

Example output:

```
Top predicted winners for Mexico GP:
 driver_number  driver_name      team_name   pred_win_proba
             4  Lando Norris     McLaren           0.682
             1  Max Verstappen   Red Bull          0.666
            16  Charles Leclerc  Ferrari           0.023
```

---

## ⚙️ Installation

### Requirements

* Python ≥ 3.9
* Dependencies:

  ```bash
  pip install pandas numpy requests python-dateutil scikit-learn xgboost pyarrow
  ```

---

## 📂 Folder Structure

```
F1WINNERS_AUSTIN 2025/
│
├── openf1_2025_until_austin/       
│   ├── drivers_upto_austin.csv
│   ├── laps_upto_austin.csv
│   ├── pit_upto_austin.csv
│   ├── results_upto_austin.csv
│   ├── sessions_race_upto_austin.csv
│   ├── stints_upto_austin.csv
│   ├── weather_upto_austin.csv
│
├── feature_importance.csv          
├── features_up_to_austin_for_mexico.csv   
├── load_dataset.py                 
├── model_predict.py                
├── prediction_mexico_probs.csv      
└── README.md                       

```

---

## 🧩 Data Sources

* **OpenF1 API**

  * Website: [https://openf1.org](https://openf1.org)
  * API: [https://api.openf1.org/v1/](https://api.openf1.org/v1/)
  * Provides open telemetry, results, and timing data for all Formula 1 sessions.

---

## 🧪 Model Logic

The model uses historical patterns up to the last race (Austin) to estimate each driver's winning chance for Mexico GP.
Main predictive features include:

* Recent race performance (average & best finish, DNF rate)
* Grid position trends
* Lap pace and median lap times
* Team performance average
* Weather averages
* Circuit characteristics (e.g., altitude, overtaking difficulty)

---

## 🧾 License

This project is open-sourced under the **MIT License**.
All Formula 1 data used is publicly available via [OpenF1](https://openf1.org).