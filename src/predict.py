from pathlib import Path

import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd

from src.build_features import build_team_game_metrics


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SEASON = 2026

MODEL_FILE = Path(
    "models/game_predictor.joblib"
)

HISTORICAL_TEAM_GAMES_FILE = Path(
    "data/team_game_metrics_2015_2025.csv"
)

OUTPUT_FILE = Path(
    "data/predictions_2026.csv"
)

SPAN = 6
TRANSITION_GAMES = 4

TEAM_METRICS = [
    "off_epa",
    "def_epa",
    "pass_epa",
    "rush_epa",
    "success_rate",
    "pace",
]


# ---------------------------------------------------------
# LOAD PRODUCTION MODEL
# ---------------------------------------------------------

def load_model_bundle():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Could not find {MODEL_FILE}. "
            "Run train_model.py first."
        )

    return joblib.load(
        MODEL_FILE
    )


# ---------------------------------------------------------
# LOAD REAL 2026 SCHEDULE
# ---------------------------------------------------------

def load_2026_schedule():

    print(
        "Downloading current 2026 NFL schedule..."
    )

    schedule = nfl.load_schedules(
        [SEASON]
    ).to_pandas()

    schedule = schedule[
        (schedule["season"] == SEASON)
        & (schedule["game_type"] == "REG")
    ].copy()

    schedule["gameday"] = pd.to_datetime(
        schedule["gameday"]
    )

    # The model was trained using nflverse's rest columns.
    required_rest_columns = [
        "home_rest",
        "away_rest",
    ]

    missing_rest = [
        column
        for column in required_rest_columns
        if column not in schedule.columns
    ]

    if missing_rest:

        raise ValueError(
            "The current schedule data is missing "
            f"these model inputs: {missing_rest}"
        )

    schedule["home_rest"] = pd.to_numeric(
        schedule["home_rest"],
        errors="coerce",
    )

    schedule["away_rest"] = pd.to_numeric(
        schedule["away_rest"],
        errors="coerce",
    )

    return schedule


# ---------------------------------------------------------
# DETERMINE CURRENT / NEXT PREDICTION WEEK
# ---------------------------------------------------------

def get_prediction_week(
    schedule
):

    unplayed = schedule[
        schedule["home_score"].isna()
        | schedule["away_score"].isna()
    ].copy()

    if unplayed.empty:

        return int(
            schedule["week"].max()
        )

    return int(
        unplayed["week"].min()
    )


# ---------------------------------------------------------
# LOAD HISTORICAL TEAM-GAME METRICS
# ---------------------------------------------------------

def load_historical_team_games():

    if not HISTORICAL_TEAM_GAMES_FILE.exists():

        raise FileNotFoundError(
            f"Could not find "
            f"{HISTORICAL_TEAM_GAMES_FILE}. "
            "Run src/build_features.py first."
        )

    historical = pd.read_csv(
        HISTORICAL_TEAM_GAMES_FILE
    )

    historical = historical[
        historical["season"] == 2025
    ].copy()

    return historical


# ---------------------------------------------------------
# LOAD COMPLETED 2026 REGULAR-SEASON PERFORMANCE
# ---------------------------------------------------------

def load_current_season_team_games():

    try:

        print(
            "Checking for completed 2026 "
            "regular-season play-by-play..."
        )

        pbp = nfl.load_pbp(
            [SEASON]
        ).to_pandas()

    except Exception as exc:

        print(
            "No usable 2026 play-by-play "
            "is available yet."
        )

        print(
            f"Reason: {exc}"
        )

        return pd.DataFrame()

    if "season_type" in pbp.columns:

        pbp = pbp[
            pbp["season_type"] == "REG"
        ].copy()

    if pbp.empty:

        print(
            "No 2026 regular-season plays "
            "have been completed yet."
        )

        return pd.DataFrame()

    team_games = build_team_game_metrics(
        pbp
    )

    if team_games.empty:

        print(
            "No completed 2026 team-game "
            "metrics are available yet."
        )

        return pd.DataFrame()

    print(
        f"Loaded {len(team_games):,} "
        "2026 team-game rows."
    )

    return team_games


# ---------------------------------------------------------
# EWM HELPER
# ---------------------------------------------------------

def calculate_final_ewm(
    values,
    alpha,
):

    state = None

    for value in values:

        if pd.isna(value):
            continue

        value = float(value)

        if state is None:

            state = value

        else:

            state = (
                alpha * value
                + (1 - alpha) * state
            )

    return state


# ---------------------------------------------------------
# BUILD CURRENT TEAM RATINGS
# ---------------------------------------------------------

def build_current_team_ratings(
    historical_2025,
    current_2026,
    schedule,
):

    alpha = 2 / (SPAN + 1)

    # -----------------------------------------------------
    # END-OF-2025 PRIOR RATINGS
    # -----------------------------------------------------

    prior_ratings = {}

    for team, group in historical_2025.groupby(
        "team"
    ):

        group = group.sort_values(
            [
                "week",
                "game_id",
            ]
        )

        prior_ratings[team] = {}

        for metric in TEAM_METRICS:

            prior_ratings[team][metric] = (
                calculate_final_ewm(
                    group[metric],
                    alpha,
                )
            )

    # -----------------------------------------------------
    # CURRENT 2026 EWM RATINGS
    # -----------------------------------------------------

    current_ratings = {}
    games_played = {}

    if not current_2026.empty:

        for team, group in current_2026.groupby(
            "team"
        ):

            group = group.sort_values(
                [
                    "week",
                    "game_id",
                ]
            )

            games_played[team] = len(
                group
            )

            current_ratings[team] = {}

            for metric in TEAM_METRICS:

                current_ratings[team][metric] = (
                    calculate_final_ewm(
                        group[metric],
                        alpha,
                    )
                )

    # -----------------------------------------------------
    # BLEND PREVIOUS SEASON WITH CURRENT SEASON
    #
    # 0 games: 100% previous season
    # 1 game:   75% previous / 25% current
    # 2 games:  50% previous / 50% current
    # 3 games:  25% previous / 75% current
    # 4+ games: 100% current
    # -----------------------------------------------------

    schedule_teams = set(
        schedule["home_team"].dropna()
    ) | set(
        schedule["away_team"].dropna()
    )

    rows = []

    for team in sorted(
        schedule_teams
    ):

        prior = prior_ratings.get(
            team,
            {}
        )

        current = current_ratings.get(
            team,
            {}
        )

        team_games_played = games_played.get(
            team,
            0,
        )

        current_weight = min(
            team_games_played
            / TRANSITION_GAMES,
            1.0,
        )

        prior_weight = (
            1.0 - current_weight
        )

        row = {
            "team": team,
            "games_played_2026":
                team_games_played,
            "current_season_weight":
                current_weight,
        }

        for metric in TEAM_METRICS:

            prior_value = prior.get(
                metric
            )

            current_value = current.get(
                metric
            )

            if (
                current_value is None
                or pd.isna(current_value)
            ):

                rating = prior_value

            elif (
                prior_value is None
                or pd.isna(prior_value)
            ):

                rating = current_value

            else:

                rating = (
                    prior_weight
                    * prior_value
                    + current_weight
                    * current_value
                )

            row[
                f"rating_{metric}"
            ] = rating

        rows.append(
            row
        )

    ratings = pd.DataFrame(
        rows
    )

    return ratings


# ---------------------------------------------------------
# BUILD WEEK MATCHUPS
# ---------------------------------------------------------

def build_prediction_matchups(
    schedule,
    ratings,
    prediction_week,
):

    games = schedule[
        schedule["week"] == prediction_week
    ].copy()

    # Only predict games that have not been completed.
    games = games[
        games["home_score"].isna()
        | games["away_score"].isna()
    ].copy()

    rating_columns = [
        "team",
        "games_played_2026",
        "current_season_weight",
    ] + [
        f"rating_{metric}"
        for metric in TEAM_METRICS
    ]

    ratings = ratings[
        rating_columns
    ].copy()

    # -----------------------------------------------------
    # HOME RATINGS
    # -----------------------------------------------------

    home = ratings.rename(
        columns={
            "team":
                "home_team",

            "games_played_2026":
                "home_games_played_2026",

            "current_season_weight":
                "home_current_season_weight",

            "rating_off_epa":
                "home_off_epa",

            "rating_def_epa":
                "home_def_epa",

            "rating_pass_epa":
                "home_pass_epa",

            "rating_rush_epa":
                "home_rush_epa",

            "rating_success_rate":
                "home_success_rate",

            "rating_pace":
                "home_pace",
        }
    )

    games = games.merge(
        home,
        on="home_team",
        how="left",
    )

    # -----------------------------------------------------
    # AWAY RATINGS
    # -----------------------------------------------------

    away = ratings.rename(
        columns={
            "team":
                "away_team",

            "games_played_2026":
                "away_games_played_2026",

            "current_season_weight":
                "away_current_season_weight",

            "rating_off_epa":
                "away_off_epa",

            "rating_def_epa":
                "away_def_epa",

            "rating_pass_epa":
                "away_pass_epa",

            "rating_rush_epa":
                "away_rush_epa",

            "rating_success_rate":
                "away_success_rate",

            "rating_pace":
                "away_pace",
        }
    )

    games = games.merge(
        away,
        on="away_team",
        how="left",
    )

    # -----------------------------------------------------
    # MODEL FEATURES
    # Positive means home-team advantage.
    # -----------------------------------------------------

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

    return games


# ---------------------------------------------------------
# GENERATE PROBABILITIES
# ---------------------------------------------------------

def generate_predictions(
    games,
    model_bundle,
):

    model = model_bundle[
        "model"
    ]

    features = model_bundle[
        "features"
    ]

    missing_features = [
        feature
        for feature in features
        if feature not in games.columns
    ]

    if missing_features:

        raise ValueError(
            "Prediction data is missing "
            f"model features: {missing_features}"
        )

    missing_rows = games[
        features
    ].isna().any(
        axis=1
    )

    if missing_rows.any():

        problem_games = games.loc[
            missing_rows,
            [
                "away_team",
                "home_team",
            ],
        ]

        print()
        print(
            "WARNING: Some games are missing "
            "required model inputs:"
        )

        print(
            problem_games.to_string(
                index=False
            )
        )

    prediction_games = games[
        ~missing_rows
    ].copy()

    X = prediction_games[
        features
    ]

    home_probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    prediction_games[
        "home_win_probability"
    ] = home_probabilities

    prediction_games[
        "away_win_probability"
    ] = (
        1
        - home_probabilities
    )

    prediction_games[
        "predicted_winner"
    ] = np.where(
        prediction_games[
            "home_win_probability"
        ] >= 0.50,
        prediction_games[
            "home_team"
        ],
        prediction_games[
            "away_team"
        ],
    )

    prediction_games[
        "predicted_winner_probability"
    ] = np.maximum(
        prediction_games[
            "home_win_probability"
        ],
        prediction_games[
            "away_win_probability"
        ],
    )

    prediction_games[
        "model_name"
    ] = model_bundle.get(
        "model_name",
        "Unknown model",
    )

    prediction_games[
        "prediction_generated_at"
    ] = pd.Timestamp.now().isoformat()

    return prediction_games


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

def print_predictions(
    predictions,
    prediction_week,
):

    print()
    print(
        f"2026 NFL WEEK "
        f"{prediction_week} PREDICTIONS"
    )

    print(
        "=" * 78
    )

    display = predictions[
        [
            "away_team",
            "home_team",
            "away_win_probability",
            "home_win_probability",
            "predicted_winner",
        ]
    ].copy()

    display[
        "away_win_probability"
    ] = (
        display[
            "away_win_probability"
        ]
        * 100
    )

    display[
        "home_win_probability"
    ] = (
        display[
            "home_win_probability"
        ]
        * 100
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "away_win_probability":
                    "{:.1f}%".format,
                "home_win_probability":
                    "{:.1f}%".format,
            },
        )
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    model_bundle = (
        load_model_bundle()
    )

    print(
        "Production model:"
    )

    print(
        model_bundle.get(
            "model_name"
        )
    )

    print(
        "Model features:"
    )

    print(
        ", ".join(
            model_bundle[
                "features"
            ]
        )
    )

    schedule = (
        load_2026_schedule()
    )

    prediction_week = (
        get_prediction_week(
            schedule
        )
    )

    print()
    print(
        f"Current prediction week: "
        f"{prediction_week}"
    )

    historical_2025 = (
        load_historical_team_games()
    )

    current_2026 = (
        load_current_season_team_games()
    )

    current_ratings = (
        build_current_team_ratings(
            historical_2025,
            current_2026,
            schedule,
        )
    )

    matchups = (
        build_prediction_matchups(
            schedule,
            current_ratings,
            prediction_week,
        )
    )

    predictions = (
        generate_predictions(
            matchups,
            model_bundle,
        )
    )

    OUTPUT_FILE.parent.mkdir(
        exist_ok=True
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_predictions(
        predictions,
        prediction_week,
    )

    print()
    print(
        f"Saved predictions to: "
        f"{OUTPUT_FILE}"
    )