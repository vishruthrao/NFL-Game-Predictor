from pathlib import Path

import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TRAIN_SEASONS = list(range(2016, 2025))
TEST_SEASON = 2025

RATINGS_FILE = Path(
    "data/pregame_team_ratings_2016_2025.csv"
)

MODEL_FOLDER = Path("models")

MODEL_FILE = MODEL_FOLDER / "game_predictor.joblib"


# ---------------------------------------------------------
# MODEL VERSIONS
# ---------------------------------------------------------

MODEL_FEATURES = {

    "Model A - Simple": [
        "off_epa_diff",
        "def_epa_diff",
        "rest_diff",
    ],

    "Model B - Current": [
        "off_epa_diff",
        "def_epa_diff",
        "pass_epa_diff",
        "rush_epa_diff",
        "success_rate_diff",
        "pace_diff",
        "rest_diff",
    ],

    "Model C - Efficiency": [
        "pass_epa_diff",
        "rush_epa_diff",
        "def_epa_diff",
        "success_rate_diff",
        "rest_diff",
    ],
}


ALL_FEATURES = sorted(
    {
        feature
        for feature_list in MODEL_FEATURES.values()
        for feature in feature_list
    }
)


# ---------------------------------------------------------
# LOAD PRE-GAME TEAM RATINGS
# ---------------------------------------------------------

def load_ratings():

    if not RATINGS_FILE.exists():

        raise FileNotFoundError(
            f"Could not find {RATINGS_FILE}. "
            "Run src/build_features.py first."
        )

    return pd.read_csv(
        RATINGS_FILE
    )


# ---------------------------------------------------------
# LOAD HISTORICAL SCHEDULES
# ---------------------------------------------------------

def load_historical_schedules():

    seasons = list(
        range(2016, 2026)
    )

    print(
        "Downloading historical schedules..."
    )

    schedule = nfl.load_schedules(
        seasons
    ).to_pandas()

    schedule = schedule[
        schedule["game_type"] == "REG"
    ].copy()

    schedule["gameday"] = pd.to_datetime(
        schedule["gameday"]
    )

    return schedule


# ---------------------------------------------------------
# REST DAYS
# ---------------------------------------------------------

def calculate_rest_if_needed(
    schedule
):

    schedule = schedule.copy()

    if (
        "home_rest" in schedule.columns
        and "away_rest" in schedule.columns
    ):

        schedule["home_rest"] = pd.to_numeric(
            schedule["home_rest"],
            errors="coerce",
        )

        schedule["away_rest"] = pd.to_numeric(
            schedule["away_rest"],
            errors="coerce",
        )

        return schedule

    home = schedule[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team": "team"
        }
    )

    away = schedule[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "away_team",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team": "team"
        }
    )

    team_schedule = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    team_schedule = team_schedule.sort_values(
        [
            "team",
            "gameday",
        ]
    )

    team_schedule["previous_game"] = (
        team_schedule
        .groupby("team")["gameday"]
        .shift(1)
    )

    team_schedule["rest"] = (
        team_schedule["gameday"]
        - team_schedule["previous_game"]
    ).dt.days

    rest_lookup = team_schedule[
        [
            "game_id",
            "team",
            "rest",
        ]
    ]

    home_rest = rest_lookup.rename(
        columns={
            "team": "home_team",
            "rest": "home_rest",
        }
    )

    schedule = schedule.merge(
        home_rest,
        on=[
            "game_id",
            "home_team",
        ],
        how="left",
    )

    away_rest = rest_lookup.rename(
        columns={
            "team": "away_team",
            "rest": "away_rest",
        }
    )

    schedule = schedule.merge(
        away_rest,
        on=[
            "game_id",
            "away_team",
        ],
        how="left",
    )

    return schedule


# ---------------------------------------------------------
# BUILD ONE ROW PER GAME
# ---------------------------------------------------------

def build_game_dataset(
    ratings,
    schedule,
):

    rating_columns = [
        "game_id",
        "team",
        "pregame_off_epa",
        "pregame_def_epa",
        "pregame_pass_epa",
        "pregame_rush_epa",
        "pregame_success_rate",
        "pregame_pace",
    ]

    ratings = ratings[
        rating_columns
    ].copy()

    home_ratings = ratings.rename(
        columns={
            "team": "home_team",
            "pregame_off_epa":
                "home_off_epa",
            "pregame_def_epa":
                "home_def_epa",
            "pregame_pass_epa":
                "home_pass_epa",
            "pregame_rush_epa":
                "home_rush_epa",
            "pregame_success_rate":
                "home_success_rate",
            "pregame_pace":
                "home_pace",
        }
    )

    games = schedule.merge(
        home_ratings,
        on=[
            "game_id",
            "home_team",
        ],
        how="left",
    )

    away_ratings = ratings.rename(
        columns={
            "team": "away_team",
            "pregame_off_epa":
                "away_off_epa",
            "pregame_def_epa":
                "away_def_epa",
            "pregame_pass_epa":
                "away_pass_epa",
            "pregame_rush_epa":
                "away_rush_epa",
            "pregame_success_rate":
                "away_success_rate",
            "pregame_pace":
                "away_pace",
        }
    )

    games = games.merge(
        away_ratings,
        on=[
            "game_id",
            "away_team",
        ],
        how="left",
    )

    # Positive means home-team advantage.

    games["off_epa_diff"] = (
        games["home_off_epa"]
        - games["away_off_epa"]
    )

    games["def_epa_diff"] = (
        games["home_def_epa"]
        - games["away_def_epa"]
    )

    games["pass_epa_diff"] = (
        games["home_pass_epa"]
        - games["away_pass_epa"]
    )

    games["rush_epa_diff"] = (
        games["home_rush_epa"]
        - games["away_rush_epa"]
    )

    games["success_rate_diff"] = (
        games["home_success_rate"]
        - games["away_success_rate"]
    )

    games["pace_diff"] = (
        games["home_pace"]
        - games["away_pace"]
    )

    games["rest_diff"] = (
        games["home_rest"]
        - games["away_rest"]
    )

    games = games[
        games["home_score"].notna()
        & games["away_score"].notna()
    ].copy()

    games = games[
        games["home_score"]
        != games["away_score"]
    ].copy()

    games["home_win"] = (
        games["home_score"]
        > games["away_score"]
    ).astype(int)

    return games


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------

def create_model():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------
# COMPARE MODELS
# ---------------------------------------------------------

def compare_models(
    games
):

    # Use the same games for every model.
    # This makes the comparison fair.

    model_data = games.dropna(
        subset=ALL_FEATURES + ["home_win"]
    ).copy()

    train = model_data[
        model_data["season"].isin(
            TRAIN_SEASONS
        )
    ].copy()

    test = model_data[
        model_data["season"]
        == TEST_SEASON
    ].copy()

    y_train = train["home_win"]
    y_test = test["home_win"]

    print()
    print("=" * 70)

    print(
        f"Training games: "
        f"{len(train):,}"
    )

    print(
        f"2025 test games: "
        f"{len(test):,}"
    )

    results = []

    trained_models = {}

    for model_name, features in MODEL_FEATURES.items():

        X_train = train[
            features
        ]

        X_test = test[
            features
        ]

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        loss = log_loss(
            y_test,
            probabilities,
        )

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "roc_auc": auc,
                "log_loss": loss,
            }
        )

        trained_models[
            model_name
        ] = {
            "model": model,
            "features": features,
        }

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        "log_loss",
        ascending=True,
    ).reset_index(drop=True)

    print()
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.6f}".format,
                "roc_auc":
                    "{:.6f}".format,
                "log_loss":
                    "{:.6f}".format,
            },
        )
    )

    return (
        comparison,
        trained_models,
        model_data,
        test,
    )

def walk_forward_test(
    games
):

    test_seasons = [
        2022,
        2023,
        2024,
        2025,
    ]

    model_data = games.dropna(
        subset=ALL_FEATURES + ["home_win"]
    ).copy()

    results = []
    prediction_rows = []

    print()
    print("WALK-FORWARD TEST")
    print("=" * 70)

    for test_season in test_seasons:

        train = model_data[
            model_data["season"] < test_season
        ].copy()

        test = model_data[
            model_data["season"] == test_season
        ].copy()

        y_train = train["home_win"]
        y_test = test["home_win"]

        for model_name, features in MODEL_FEATURES.items():

            X_train = train[
                features
            ]

            X_test = test[
                features
            ]

            model = create_model()

            model.fit(
                X_train,
                y_train,
            )

            probabilities = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

            predictions = (
                probabilities >= 0.50
            ).astype(int)

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            auc = roc_auc_score(
                y_test,
                probabilities,
            )

            loss = log_loss(
                y_test,
                probabilities,
            )

            brier = brier_score_loss(
                y_test,
                probabilities,
            )

            results.append(
                {
                    "test_season": test_season,
                    "model": model_name,
                    "accuracy": accuracy,
                    "roc_auc": auc,
                    "log_loss": loss,
                    "brier_score": brier,
                    "games": len(test),
                }
            )

            # Save every out-of-sample prediction so calibration
            # can be assessed using only games the model had not seen.
            season_predictions = test[
                [
                    "game_id",
                    "season",
                    "week",
                    "gameday",
                    "away_team",
                    "home_team",
                    "home_win",
                ]
            ].copy()

            season_predictions["model"] = model_name
            season_predictions["home_win_probability"] = probabilities
            season_predictions["predicted_home_win"] = predictions

            prediction_rows.append(
                season_predictions
            )

    walk_forward = pd.DataFrame(
        results
    )

    walk_forward_predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    print(
        walk_forward.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.6f}".format,
                "roc_auc":
                    "{:.6f}".format,
                "log_loss":
                    "{:.6f}".format,
                "brier_score":
                    "{:.6f}".format,
            },
        )
    )

    # -----------------------------------------------------
    # AVERAGE PERFORMANCE ACROSS TEST SEASONS
    # -----------------------------------------------------

    averages = (
        walk_forward
        .groupby("model", as_index=False)
        .agg(
            avg_accuracy=(
                "accuracy",
                "mean",
            ),
            avg_roc_auc=(
                "roc_auc",
                "mean",
            ),
            avg_log_loss=(
                "log_loss",
                "mean",
            ),
            avg_brier_score=(
                "brier_score",
                "mean",
            ),
        )
        .sort_values(
            "avg_log_loss"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("WALK-FORWARD AVERAGES")
    print("=" * 70)

    print(
        averages.to_string(
            index=False,
            formatters={
                "avg_accuracy":
                    "{:.6f}".format,
                "avg_roc_auc":
                    "{:.6f}".format,
                "avg_log_loss":
                    "{:.6f}".format,
                "avg_brier_score":
                    "{:.6f}".format,
            },
        )
    )

    # -----------------------------------------------------
    # CALIBRATION FOR THE BEST WALK-FORWARD MODEL
    # -----------------------------------------------------

    best_model_name = averages.iloc[0]["model"]

    best_predictions = walk_forward_predictions[
        walk_forward_predictions["model"] == best_model_name
    ].copy()

    best_brier = brier_score_loss(
        best_predictions["home_win"],
        best_predictions["home_win_probability"],
    )

    prob_true, prob_pred = calibration_curve(
        best_predictions["home_win"],
        best_predictions["home_win_probability"],
        n_bins=8,
        strategy="quantile",
    )

    calibration = pd.DataFrame(
        {
            "mean_predicted_home_win_probability": prob_pred,
            "actual_home_win_rate": prob_true,
        }
    )

    # Also create a fan-friendly calibration table based on
    # the model's favorite, regardless of whether that team
    # is home or away.
    best_predictions["favorite_probability"] = np.maximum(
        best_predictions["home_win_probability"],
        1 - best_predictions["home_win_probability"],
    )

    best_predictions["favorite_is_home"] = (
        best_predictions["home_win_probability"] >= 0.50
    )

    best_predictions["favorite_won"] = np.where(
        best_predictions["favorite_is_home"],
        best_predictions["home_win"] == 1,
        best_predictions["home_win"] == 0,
    ).astype(int)

    confidence_bins = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        1.000001,
    ]

    confidence_labels = [
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80-85%",
        "85-90%",
        "90%+",
    ]

    best_predictions["confidence_bucket"] = pd.cut(
        best_predictions["favorite_probability"],
        bins=confidence_bins,
        labels=confidence_labels,
        right=False,
        include_lowest=True,
    )

    favorite_calibration = (
        best_predictions
        .groupby(
            "confidence_bucket",
            observed=False,
        )
        .agg(
            games=(
                "favorite_won",
                "size",
            ),
            avg_model_confidence=(
                "favorite_probability",
                "mean",
            ),
            actual_favorite_win_rate=(
                "favorite_won",
                "mean",
            ),
        )
        .reset_index()
    )

    favorite_calibration = favorite_calibration[
        favorite_calibration["games"] > 0
    ].copy()

    print()
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(
        f"Best walk-forward model: {best_model_name}"
    )
    print(
        f"Out-of-sample Brier score: {best_brier:.6f}"
    )

    print()
    print("FAVORITE CONFIDENCE CALIBRATION")
    print("=" * 70)

    print(
        favorite_calibration.to_string(
            index=False,
            formatters={
                "avg_model_confidence":
                    "{:.3f}".format,
                "actual_favorite_win_rate":
                    "{:.3f}".format,
            },
        )
    )

    walk_forward.to_csv(
        "data/walk_forward_results.csv",
        index=False,
    )

    averages.to_csv(
        "data/walk_forward_averages.csv",
        index=False,
    )

    walk_forward_predictions.to_csv(
        "data/walk_forward_predictions.csv",
        index=False,
    )

    calibration.to_csv(
        "data/calibration_curve.csv",
        index=False,
    )

    favorite_calibration.to_csv(
        "data/favorite_calibration.csv",
        index=False,
    )

    return (
        walk_forward,
        averages,
        walk_forward_predictions,
        calibration,
        favorite_calibration,
        best_brier,
    )


# ---------------------------------------------------------
# BASELINES
# ---------------------------------------------------------

def show_baselines(
    model_data
):

    train = model_data[
        model_data["season"].isin(
            TRAIN_SEASONS
        )
    ].copy()

    test = model_data[
        model_data["season"]
        == TEST_SEASON
    ].copy()

    y_train = train["home_win"]
    y_test = test["home_win"]

    always_home_predictions = np.ones(
        len(y_test),
        dtype=int,
    )

    home_accuracy = accuracy_score(
        y_test,
        always_home_predictions,
    )

    fifty_fifty = np.full(
        len(y_test),
        0.50,
    )

    fifty_loss = log_loss(
        y_test,
        fifty_fifty,
    )

    historical_home_rate = (
        y_train.mean()
    )

    historical_probabilities = np.full(
        len(y_test),
        historical_home_rate,
    )

    historical_loss = log_loss(
        y_test,
        historical_probabilities,
    )

    print()
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"Always-home accuracy: "
        f"{home_accuracy:.3f}"
    )

    print(
        f"50/50 log loss:       "
        f"{fifty_loss:.3f}"
    )

    print(
        f"Historical-home loss: "
        f"{historical_loss:.3f}"
    )


# ---------------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------------

def train_production_model(
    games,
    walk_forward_averages,
    calibration_brier,
):

    # -----------------------------------------------------
    # CHOOSE MODEL USING MULTI-SEASON WALK-FORWARD RESULTS
    # -----------------------------------------------------

    best_model_name = (
        walk_forward_averages
        .sort_values(
            "avg_log_loss",
            ascending=True,
        )
        .iloc[0]["model"]
    )

    features = MODEL_FEATURES[
        best_model_name
    ]

    # -----------------------------------------------------
    # RETRAIN ON ALL COMPLETED MODELING SEASONS
    # 2016 THROUGH 2025
    # -----------------------------------------------------

    production_data = games[
        games["season"].between(
            2016,
            2025,
        )
    ].dropna(
        subset=features + ["home_win"]
    ).copy()

    X = production_data[
        features
    ]

    y = production_data[
        "home_win"
    ]

    production_model = create_model()

    production_model.fit(
        X,
        y,
    )

    best_metrics = (
        walk_forward_averages[
            walk_forward_averages["model"]
            == best_model_name
        ]
        .iloc[0]
    )

    MODEL_FOLDER.mkdir(
        exist_ok=True
    )

    joblib.dump(
        {
            "model":
                production_model,

            "features":
                features,

            "model_name":
                best_model_name,

            "selection_method":
                "Lowest average walk-forward log loss",

            "walk_forward_avg_accuracy":
                float(
                    best_metrics[
                        "avg_accuracy"
                    ]
                ),

            "walk_forward_avg_roc_auc":
                float(
                    best_metrics[
                        "avg_roc_auc"
                    ]
                ),

            "walk_forward_avg_log_loss":
                float(
                    best_metrics[
                        "avg_log_loss"
                    ]
                ),

            "walk_forward_avg_brier_score":
                float(
                    best_metrics[
                        "avg_brier_score"
                    ]
                ),

            "calibration_brier_score":
                float(
                    calibration_brier
                ),

            "trained_through_season":
                2025,

            "training_games":
                int(
                    len(production_data)
                ),
        },
        MODEL_FILE,
    )

    print()
    print("PRODUCTION MODEL")
    print("=" * 70)

    print(
        f"Selected model: {best_model_name}"
    )

    print(
        "Selected by: lowest average "
        "walk-forward log loss"
    )

    print(
        f"Features: {', '.join(features)}"
    )

    print(
        f"Training games: {len(production_data):,}"
    )

    print(
        "Retrained on all completed seasons: "
        "2016-2025"
    )

    print(
        f"Saved production model to: "
        f"{MODEL_FILE}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    ratings = load_ratings()

    schedule = (
        load_historical_schedules()
    )

    schedule = (
        calculate_rest_if_needed(
            schedule
        )
    )

    # -----------------------------------------------------
    # BUILD GAME-LEVEL MODEL DATASET
    # -----------------------------------------------------

    games = build_game_dataset(
        ratings,
        schedule,
    )

    games.to_csv(
        "data/model_games_2016_2025.csv",
        index=False,
    )

    print()
    print(
        "Saved modeling dataset to: "
        "data/model_games_2016_2025.csv"
    )

    # -----------------------------------------------------
    # SINGLE-SEASON 2025 COMPARISON
    # -----------------------------------------------------

    (
        comparison,
        trained_models,
        model_data,
        test,
    ) = compare_models(
        games
    )

    # -----------------------------------------------------
    # MULTI-SEASON WALK-FORWARD + CALIBRATION
    # -----------------------------------------------------

    (
        walk_forward,
        walk_forward_averages,
        walk_forward_predictions,
        calibration,
        favorite_calibration,
        calibration_brier,
    ) = walk_forward_test(
        games
    )

    # -----------------------------------------------------
    # BASELINE COMPARISON
    # -----------------------------------------------------

    show_baselines(
        model_data
    )

    comparison.to_csv(
        "data/model_comparison.csv",
        index=False,
    )

    # -----------------------------------------------------
    # FINAL PRODUCTION MODEL
    # -----------------------------------------------------

    train_production_model(
        games,
        walk_forward_averages,
        calibration_brier,
    )