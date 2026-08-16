from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd


def load_play_by_play(season):
    """
    Load one season of nflverse play-by-play data.
    """

    print(f"Downloading {season} play-by-play data...")

    pbp = nfl.load_pbp([season]).to_pandas()

    # Regular season only
    if "season_type" in pbp.columns:
        pbp = pbp[
            pbp["season_type"] == "REG"
        ].copy()

    print(
        f"Loaded {len(pbp):,} plays."
    )

    return pbp


def build_team_game_metrics(pbp):
    """
    Convert play-by-play data into one row
    per team per game.
    """

    required_columns = [
        "season",
        "week",
        "game_id",
        "posteam",
        "defteam",
        "play_type",
        "epa",
        "success",
    ]

    missing = [
        column
        for column in required_columns
        if column not in pbp.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Keep normal pass and run plays.
    plays = pbp[
        pbp["play_type"].isin(
            ["pass", "run"]
        )
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & pbp["epa"].notna()
    ].copy()

    # Separate passing and rushing EPA.
    plays["pass_epa_value"] = np.where(
        plays["play_type"] == "pass",
        plays["epa"],
        np.nan,
    )

    plays["rush_epa_value"] = np.where(
        plays["play_type"] == "run",
        plays["epa"],
        np.nan,
    )

    # -----------------------------------------------------
    # OFFENSE
    # -----------------------------------------------------

    offense = (
        plays.groupby(
            [
                "season",
                "week",
                "game_id",
                "posteam",
            ],
            as_index=False,
        )
        .agg(
            off_epa=(
                "epa",
                "mean",
            ),
            pass_epa=(
                "pass_epa_value",
                "mean",
            ),
            rush_epa=(
                "rush_epa_value",
                "mean",
            ),
            success_rate=(
                "success",
                "mean",
            ),
            offensive_plays=(
                "epa",
                "size",
            ),
        )
        .rename(
            columns={
                "posteam": "team"
            }
        )
    )

    # For our first version:
    # pace = offensive plays per game
    offense["pace"] = (
        offense["offensive_plays"]
    )

    # -----------------------------------------------------
    # DEFENSE
    # -----------------------------------------------------

    defense = (
        plays.groupby(
            [
                "season",
                "week",
                "game_id",
                "defteam",
            ],
            as_index=False,
        )
        .agg(
            opponent_epa=(
                "epa",
                "mean",
            ),
            defensive_plays=(
                "epa",
                "size",
            ),
        )
        .rename(
            columns={
                "defteam": "team"
            }
        )
    )

    # Flip the sign so:
    #
    # higher defensive EPA = better defense
    #
    # Example:
    # offense produces -0.10 EPA/play
    # defense receives +0.10 defensive EPA
    defense["def_epa"] = (
        -defense["opponent_epa"]
    )

    # -----------------------------------------------------
    # COMBINE
    # -----------------------------------------------------

    team_games = offense.merge(
        defense[
            [
                "season",
                "week",
                "game_id",
                "team",
                "def_epa",
                "defensive_plays",
            ]
        ],
        on=[
            "season",
            "week",
            "game_id",
            "team",
        ],
        how="inner",
    )

    team_games = team_games[
        [
            "season",
            "week",
            "game_id",
            "team",
            "off_epa",
            "def_epa",
            "pass_epa",
            "rush_epa",
            "success_rate",
            "pace",
            "offensive_plays",
            "defensive_plays",
        ]
    ]

    team_games = team_games.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team",
        ]
    ).reset_index(drop=True)

    return team_games

def build_pregame_ratings(
    team_games,
    span=6,
    transition_games=4,
):
    """
    Build pregame team ratings with exponentially weighted
    recent performance.

    Important:
    The current game's statistics are NEVER used to predict
    that same game.

    Early in a season, ratings are blended with the team's
    final rating from the previous season.
    """

    metrics = [
        "off_epa",
        "def_epa",
        "pass_epa",
        "rush_epa",
        "success_rate",
        "pace",
    ]

    df = team_games.copy()

    df = df.sort_values(
        [
            "season",
            "team",
            "week",
            "game_id",
        ]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # STEP 1
    # Calculate each team's END-OF-SEASON rating.
    #
    # These become the prior for the following season.
    # -----------------------------------------------------

    alpha = 2 / (span + 1)

    end_of_season_ratings = {}

    for (season, team), group in df.groupby(
        ["season", "team"]
    ):

        group = group.sort_values(
            ["week", "game_id"]
        )

        current_values = {}

        for metric in metrics:

            ewm_value = None

            for value in group[metric]:

                if pd.isna(value):
                    continue

                if ewm_value is None:
                    ewm_value = value

                else:
                    ewm_value = (
                        alpha * value
                        + (1 - alpha) * ewm_value
                    )

            current_values[metric] = ewm_value

        end_of_season_ratings[
            (season, team)
        ] = current_values

    # -----------------------------------------------------
    # STEP 2
    # Calculate rating BEFORE every game.
    # -----------------------------------------------------

    output_rows = []

    for (season, team), group in df.groupby(
        ["season", "team"]
    ):

        group = group.sort_values(
            ["week", "game_id"]
        )

        previous_season_rating = (
            end_of_season_ratings.get(
                (season - 1, team),
                {},
            )
        )

        # Current-season EWM state.
        current_ewm = {
            metric: None
            for metric in metrics
        }

        games_played = 0

        for _, row in group.iterrows():

            result = row.to_dict()

            # ---------------------------------------------
            # Current season weight
            #
            # Week/game progression:
            #
            # Before game 1: 0%
            # Before game 2: 25%
            # Before game 3: 50%
            # Before game 4: 75%
            # Before game 5+: 100%
            # ---------------------------------------------

            current_weight = min(
                games_played / transition_games,
                1.0,
            )

            prior_weight = (
                1.0 - current_weight
            )

            for metric in metrics:

                prior_value = (
                    previous_season_rating.get(
                        metric
                    )
                )

                current_value = (
                    current_ewm[metric]
                )

                # -----------------------------------------
                # First game of season
                # -----------------------------------------

                if current_value is None:

                    pregame_value = (
                        prior_value
                    )

                # -----------------------------------------
                # No prior-season value available
                # -----------------------------------------

                elif (
                    prior_value is None
                    or pd.isna(prior_value)
                ):

                    pregame_value = (
                        current_value
                    )

                # -----------------------------------------
                # Blend prior season + current form
                # -----------------------------------------

                else:

                    pregame_value = (
                        prior_weight
                        * prior_value
                        + current_weight
                        * current_value
                    )

                result[
                    f"pregame_{metric}"
                ] = pregame_value

            result[
                "games_played_before"
            ] = games_played

            result[
                "current_season_weight"
            ] = current_weight

            output_rows.append(
                result
            )

            # ---------------------------------------------
            # AFTER saving the pregame rating,
            # update the current-season EWM using this game.
            #
            # Doing this AFTER is what prevents leakage.
            # ---------------------------------------------

            for metric in metrics:

                value = row[metric]

                if pd.isna(value):
                    continue

                if current_ewm[metric] is None:

                    current_ewm[metric] = (
                        value
                    )

                else:

                    current_ewm[metric] = (
                        alpha * value
                        + (1 - alpha)
                        * current_ewm[metric]
                    )

            games_played += 1

    ratings = pd.DataFrame(
        output_rows
    )

    ratings = ratings.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team",
        ]
    ).reset_index(drop=True)

    return ratings
if __name__ == "__main__":

    SEASONS = list(range(2015, 2026))

    all_team_games = []

    print("\nBUILDING HISTORICAL NFL DATASET")
    print("=" * 70)

    for season in SEASONS:

        print(f"\nProcessing {season}...")

        pbp = load_play_by_play(
            season
        )

        season_metrics = build_team_game_metrics(
            pbp
        )

        all_team_games.append(
            season_metrics
        )

        print(
            f"{season}: "
            f"{len(season_metrics):,} team-game rows created."
        )

    team_games = pd.concat(
        all_team_games,
        ignore_index=True,
    )

    team_games = team_games.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team",
        ]
    ).reset_index(drop=True)

    print("\n" + "=" * 70)

    print(
        f"Total team-game rows: "
        f"{len(team_games):,}"
    )

    print(
        f"Seasons: "
        f"{team_games['season'].min()} "
        f"through "
        f"{team_games['season'].max()}"
    )

    print("\nSample:")

    print(
        team_games.head(20).to_string(
            index=False
        )
    )

    data_folder = Path("data")

    data_folder.mkdir(
        exist_ok=True
    )

    output_file = (
        data_folder
        / "team_game_metrics_2015_2025.csv"
    )

    team_games.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved historical metrics to: "
        f"{output_file}"
    )
        # -----------------------------------------------------
    # BUILD PRE-GAME RATINGS
    # -----------------------------------------------------

    print(
        "\nBuilding pregame team ratings..."
    )

    pregame_ratings = build_pregame_ratings(
        team_games,
        span=6,
        transition_games=4,
    )

    # We loaded 2015 mainly as a warm-up season.
    # Our modeling dataset begins in 2016.

    pregame_ratings = pregame_ratings[
        pregame_ratings["season"] >= 2016
    ].copy()

    ratings_file = (
        data_folder
        / "pregame_team_ratings_2016_2025.csv"
    )

    pregame_ratings.to_csv(
        ratings_file,
        index=False,
    )

    print(
        f"\nSaved pregame ratings to: "
        f"{ratings_file}"
    )

    print(
        "\nSample pregame ratings:"
    )

    display_columns = [
        "season",
        "week",
        "team",
        "games_played_before",
        "current_season_weight",
        "pregame_off_epa",
        "pregame_def_epa",
        "pregame_pass_epa",
        "pregame_rush_epa",
        "pregame_success_rate",
        "pregame_pace",
    ]

    print(
        pregame_ratings[
            display_columns
        ].head(30).to_string(
            index=False
        )
    )