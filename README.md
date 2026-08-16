# NFL Game Predictor

Starter Streamlit interface for an NFL weekly game prediction project.

## What is included

- Weekly game cards
- Win probabilities
- Predicted winner
- Expandable matchup analytics
- Metric definitions
- Explanation of baseline vs final prediction
- Placeholder model performance section
- Sample data only for the first UI prototype

## Run locally

Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py
```

## Next steps

1. Pull nflverse play-by-play and schedules with `nflreadpy`.
2. Build pregame team features.
3. Train a logistic regression model on 2016-2025 games.
4. Backtest on a held-out season.
5. Save the trained model with `joblib`.
6. Replace sample UI data with real weekly predictions.
