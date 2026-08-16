from pathlib import Path

import nflreadpy as nfl


def load_schedule(season=2026):
    """
    Load the real NFL regular-season schedule from nflverse.
    """

    schedules = nfl.load_schedules([season]).to_pandas()

    regular_season = schedules[
        (schedules["season"] == season)
        & (schedules["game_type"] == "REG")
    ].copy()

    columns_we_want = [
        "game_id",
        "season",
        "week",
        "gameday",
        "weekday",
        "gametime",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "result",
    ]

    existing_columns = [
        column
        for column in columns_we_want
        if column in regular_season.columns
    ]

    regular_season = regular_season[existing_columns]

    regular_season = regular_season.sort_values(
        ["week", "gameday", "gametime"]
    ).reset_index(drop=True)

    return regular_season


if __name__ == "__main__":
    schedule = load_schedule(2026)

    print("\n2026 NFL REGULAR-SEASON SCHEDULE")
    print("=" * 70)

    print(
        schedule[
            [
                "week",
                "gameday",
                "away_team",
                "home_team",
            ]
        ].to_string(index=False)
    )

    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    output_file = data_folder / "schedule_2026.csv"

    schedule.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved schedule to: {output_file}"
    )