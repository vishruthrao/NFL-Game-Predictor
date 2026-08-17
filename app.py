from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="wide",
)


# ---------------------------------------------------------
# TEAM NAMES
# ---------------------------------------------------------

TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

TEAM_COLORS = {
    "ARI": "#97233F",
    "ATL": "#A71930",
    "BAL": "#241773",
    "BUF": "#C60C30",
    "CAR": "#0085CA",
    "CHI": "#C83803",
    "CIN": "#FB4F14",
    "CLE": "#311D00",
    "DAL": "#A5ACAF",
    "DEN": "#FB4F14",
    "DET": "#0076B6",
    "GB": "#FFB612",
    "HOU": "#A71930",
    "IND": "#002C5F",
    "JAX": "#006778",
    "KC": "#E31837",
    "LA": "#FFD100",
    "LAC": "#0080C6",
    "LV": "#A5ACAF",
    "MIA": "#008E97",
    "MIN": "#4F2683",
    "NE": "#002244",
    "NO": "#D3BC8D",
    "NYG": "#0B2265",
    "NYJ": "#125740",
    "PHI": "#004C54",
    "PIT": "#FFB612",
    "SEA": "#69BE28",
    "SF": "#AA0000",
    "TB": "#D50A0A",
    "TEN": "#4B92DB",
    "WAS": "#5A1414",
}


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

SCHEDULE_FILE = Path(
    "data/schedule_2026.csv"
)

PREDICTIONS_FILE = Path(
    "data/predictions_2026.csv"
)

MODEL_FILE = Path(
    "models/game_predictor.joblib"
)

CALIBRATION_FILE = Path(
    "data/favorite_calibration.csv"
)

LAST_UPDATED_FILE = Path(
    "data/last_updated.txt"
)


TEAM_GRAPHICS_URL = (
    "https://github.com/nflverse/nflverse-data/"
    "releases/download/teams/teams_colors_logos.csv"
)


# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background: #14171A;
            color: #F5F7FA;
        }

        [data-testid="stHeader"] {
            background: #14171A;
        }

        /* Hide Streamlit's automatic header permalink icons everywhere. */
        [data-testid="stHeaderActionElements"],
        [data-testid="stHeadingWithActionElements"] > a,
        .stMarkdown a.header-anchor {
            display: none !important;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .site-title {
            color: #F5F7FA;
            font-size: 2.55rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .site-subtitle {
            color: #A7ADB5;
            margin-bottom: 0.20rem;
        }

        .last-updated {
            color: #7F8790;
            font-size: 0.82rem;
            margin-bottom: 1.4rem;
        }


        .week-nav-title {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 2.55rem;
            color: #F5F7FA;
            font-size: 1.25rem;
            font-weight: 800;
            text-align: center;
        }

        div[data-testid="stButton"] > button {
            border-radius: 12px;
            border: 1px solid #343A40;
            background: #1C2025;
            color: #F5F7FA;
            font-size: 1.2rem;
            font-weight: 800;
            min-height: 2.75rem;
            transition:
                transform 160ms ease,
                border-color 160ms ease,
                background 160ms ease;
        }

        div[data-testid="stButton"] > button:hover:not(:disabled) {
            border-color: #59616A;
            background: #252A30;
            transform: translateY(-1px);
        }

        div[data-testid="stButton"] > button:disabled {
            opacity: 0.35;
            cursor: default;
        }

        .bye-week-banner {
            background: #1C2025;
            border: 1px solid #343A40;
            border-radius: 14px;
            padding: 0.80rem 1rem;
            margin: 0.55rem 0 1.1rem 0;
            color: #F5F7FA;
            font-size: 0.95rem;
        }

        .bye-week-label {
            font-weight: 800;
            margin-right: 0.35rem;
        }

        .bye-week-teams {
            color: #A7ADB5;
            font-weight: 600;
        }

        .record-row {
            display: flex;
            justify-content: flex-end;
            gap: 0.65rem;
            margin: 0.55rem 0 0.90rem 0;
        }

        .record-pill {
            display: inline-flex;
            align-items: baseline;
            gap: 0.45rem;
            background: #1C2025;
            border: 1px solid #343A40;
            border-radius: 12px;
            padding: 0.55rem 0.80rem;
        }

        .record-label {
            color: #A7ADB5;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .record-value {
            color: #F5F7FA;
            font-size: 1.05rem;
            font-weight: 800;
        }

        .final-score {
            color: #F5F7FA;
            font-size: 1rem;
            font-weight: 800;
            margin: -0.30rem 0 0.90rem 0;
        }

        .game-card {
            background: #1C2025;
            border: 1px solid #343A40;
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
        }

        .game-info {
            color: #A7ADB5;
            font-size: 0.90rem;
            margin-bottom: 0.8rem;
        }

        .team-name {
            font-size: 1.15rem;
            font-weight: 700;
        }

        .team-abbr {
            color: #A7ADB5;
            font-size: 0.86rem;
        }

        .probability {
            font-size: 1.70rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        .versus {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 700;
            padding-top: 0.9rem;
        }

        .winner-label {
            color: #A7ADB5;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.8rem;
        }

        .winner-name {
            font-size: 1.10rem;
            font-weight: 800;
        }

        .metric-note {
            color: #A7ADB5;
            font-size: 0.84rem;
            margin-top: 0.55rem;
        }

        .matchup-accent {
            height: 5px;
            border-radius: 999px;
            margin: -0.15rem 0 0.95rem 0;
            animation: accentReveal 700ms ease-out;
            transform-origin: left;
        }

        .team-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 3.15rem;
            padding: 0.24rem 0.60rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            margin-bottom: 0.45rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        }


        .team-logo-wrap {
            height: 86px;
            display: flex;
            align-items: center;
            margin-bottom: 0.35rem;
        }

        .team-logo {
            width: 78px;
            height: 78px;
            object-fit: contain;
            filter: drop-shadow(0 7px 12px rgba(0,0,0,0.24));
            animation: logoEnter 600ms cubic-bezier(.2,.8,.2,1) both;
            transition:
                transform 180ms ease,
                filter 180ms ease;
        }

        .team-logo:hover {
            transform: translateY(-2px) scale(1.04);
            filter: drop-shadow(0 10px 16px rgba(0,0,0,0.30));
        }

        .winner-logo {
            width: 25px;
            height: 25px;
            object-fit: contain;
            vertical-align: middle;
            margin-right: 0.38rem;
            filter: drop-shadow(0 3px 5px rgba(0,0,0,0.20));
        }

        .prediction-bar {
            display: flex;
            width: 100%;
            height: 13px;
            overflow: hidden;
            border-radius: 999px;
            margin-top: 0.85rem;
            background: rgba(128,128,128,0.14);
            animation: barReveal 750ms cubic-bezier(.2,.8,.2,1);
            transform-origin: left;
        }

        .prediction-segment {
            height: 100%;
        }

        .winner-chip {
            display: inline-block;
            padding: 0.38rem 0.75rem;
            border-radius: 999px;
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.25rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
            animation: winnerPop 550ms cubic-bezier(.2,.9,.25,1.2);
        }

        .prediction-confidence {
            color: #A7ADB5;
            font-size: 0.82rem;
            margin-top: 0.30rem;
        }

        .game-card {
            transition:
                transform 180ms ease,
                box-shadow 180ms ease,
                border-color 180ms ease;
            animation: cardEnter 500ms ease both;
        }

        .game-card:hover {
            transform: translateY(-2px);
            border-color: #4A5159;
            box-shadow: 0 10px 28px rgba(0,0,0,0.24);
        }

        @keyframes cardEnter {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes accentReveal {
            from {
                opacity: 0;
                transform: scaleX(0.25);
            }
            to {
                opacity: 1;
                transform: scaleX(1);
            }
        }

        @keyframes barReveal {
            from {
                opacity: 0;
                transform: scaleX(0.10);
            }
            to {
                opacity: 1;
                transform: scaleX(1);
            }
        }

        @keyframes winnerPop {
            from {
                opacity: 0;
                transform: scale(0.92);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }


        @keyframes logoEnter {
            from {
                opacity: 0;
                transform: translateY(7px) scale(0.94);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .game-card,
            .matchup-accent,
            .prediction-bar,
            .winner-chip,
            .team-logo {
                animation: none !important;
                transition: none !important;
            }

            .game-card:hover {
                transform: none;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_schedule():

    if not SCHEDULE_FILE.exists():
        return None

    schedule = pd.read_csv(
        SCHEDULE_FILE
    )

    schedule["gameday"] = pd.to_datetime(
        schedule["gameday"]
    )

    return schedule


@st.cache_data
def load_predictions():

    if not PREDICTIONS_FILE.exists():
        return None

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    if "gameday" in predictions.columns:
        predictions["gameday"] = pd.to_datetime(
            predictions["gameday"]
        )

    return predictions


@st.cache_resource
def load_model_bundle():

    if not MODEL_FILE.exists():
        return None

    return joblib.load(
        MODEL_FILE
    )


@st.cache_data
def load_calibration():

    if not CALIBRATION_FILE.exists():
        return None

    return pd.read_csv(
        CALIBRATION_FILE
    )


@st.cache_data
def load_last_updated():

    if not LAST_UPDATED_FILE.exists():
        return None

    try:
        timestamp = pd.to_datetime(
            LAST_UPDATED_FILE.read_text(
                encoding="utf-8"
            ).strip(),
            utc=True,
        )
    except Exception:
        return None

    return timestamp.tz_convert(
        "America/New_York"
    )



@st.cache_data(ttl=86400)
def load_team_graphics():
    """
    Load current NFL team logos from the nflverse team graphics dataset.

    If the remote logo file cannot be reached, the app still runs
    and simply omits the logos.
    """

    try:
        teams = pd.read_csv(
            TEAM_GRAPHICS_URL
        )
    except Exception:
        return {}

    teams = teams[
        teams["team_abbr"].isin(
            TEAM_NAMES.keys()
        )
    ].copy()

    graphics = {}

    for _, row in teams.iterrows():

        logo = None

        for logo_column in [
            "team_logo_espn",
            "team_logo_wikipedia",
            "team_logo_squared",
        ]:

            if (
                logo_column in row.index
                and pd.notna(
                    row[logo_column]
                )
                and str(
                    row[logo_column]
                ).strip()
            ):
                logo = str(
                    row[logo_column]
                ).strip()
                break

        graphics[
            row["team_abbr"]
        ] = {
            "logo": logo,
        }

    return graphics


schedule = load_schedule()
predictions = load_predictions()
model_bundle = load_model_bundle()
calibration = load_calibration()
last_updated = load_last_updated()
team_graphics = load_team_graphics()


if schedule is None:

    st.error(
        "Could not find data/schedule_2026.csv. "
        "Run src/load_data.py first."
    )

    st.stop()


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def team_name(
    abbreviation
):

    return TEAM_NAMES.get(
        abbreviation,
        abbreviation,
    )



def team_color(
    abbreviation
):

    return TEAM_COLORS.get(
        abbreviation,
        "#4B5563",
    )



def team_logo(
    abbreviation
):

    team_data = team_graphics.get(
        abbreviation,
        {}
    )

    return team_data.get(
        "logo"
    )


def logo_html(
    abbreviation,
    team_full_name,
    css_class="team-logo",
):

    logo = team_logo(
        abbreviation
    )

    if not logo:
        return ""

    safe_name = (
        str(team_full_name)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return (
        f'<img class="{css_class}" '
        f'src="{logo}" '
        f'alt="{safe_name} logo" '
        f'loading="lazy">'
    )


def readable_text_color(
    hex_color
):

    color = hex_color.lstrip("#")

    if len(color) != 6:
        return "#FFFFFF"

    red = int(
        color[0:2],
        16,
    )

    green = int(
        color[2:4],
        16,
    )

    blue = int(
        color[4:6],
        16,
    )

    brightness = (
        red * 299
        + green * 587
        + blue * 114
    ) / 1000

    return (
        "#111827"
        if brightness > 160
        else "#FFFFFF"
    )


def fmt_epa(
    value
):

    if pd.isna(value):
        return "—"

    return f"{value:+.3f}"


def fmt_percent(
    value
):

    if pd.isna(value):
        return "—"

    return f"{value:.1%}"


def fmt_number(
    value
):

    if pd.isna(value):
        return "—"

    return f"{value:.1f}"


def fmt_days(
    value
):

    if pd.isna(value):
        return "—"

    return f"{int(value)} days"



def format_game_time(
    gametime
):

    if pd.isna(gametime):
        return "Time TBD"

    try:
        eastern_time = pd.to_datetime(
            str(gametime),
            format="%H:%M"
        )
    except ValueError:
        return str(gametime)

    central_time = (
        eastern_time
        - pd.Timedelta(hours=1)
    )

    eastern_text = eastern_time.strftime(
        "%I:%M %p"
    ).lstrip("0")

    central_text = central_time.strftime(
        "%I:%M %p"
    ).lstrip("0")

    return (
        f"{eastern_text} ET • "
        f"{central_text} CT"
    )


def advantage(
    away_value,
    home_value,
    away_team,
    home_team,
    higher_is_better=True,
):

    if (
        pd.isna(away_value)
        or pd.isna(home_value)
    ):
        return "—"

    if abs(
        home_value - away_value
    ) < 1e-12:
        return "Even"

    if higher_is_better:

        return (
            home_team
            if home_value > away_value
            else away_team
        )

    return (
        home_team
        if home_value < away_value
        else away_team
    )


def build_prediction_results(
    schedule_df,
    predictions_df,
):
    # Match saved predictions to completed games.

    empty_results = pd.DataFrame(
        columns=[
            "game_id",
            "week",
            "predicted_winner",
            "actual_winner",
            "correct",
        ]
    )

    if (
        predictions_df is None
        or predictions_df.empty
    ):
        return empty_results

    required_prediction_columns = {
        "game_id",
        "predicted_winner",
    }

    required_schedule_columns = {
        "game_id",
        "week",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
    }

    if (
        not required_prediction_columns.issubset(
            predictions_df.columns
        )
        or not required_schedule_columns.issubset(
            schedule_df.columns
        )
    ):
        return empty_results

    saved_predictions = (
        predictions_df[
            [
                "game_id",
                "predicted_winner",
            ]
        ]
        .drop_duplicates(
            subset="game_id",
            keep="last",
        )
        .copy()
    )

    results = saved_predictions.merge(
        schedule_df[
            [
                "game_id",
                "week",
                "away_team",
                "home_team",
                "away_score",
                "home_score",
            ]
        ],
        on="game_id",
        how="left",
    )

    results["week"] = pd.to_numeric(
        results["week"],
        errors="coerce",
    )

    results["away_score"] = pd.to_numeric(
        results["away_score"],
        errors="coerce",
    )

    results["home_score"] = pd.to_numeric(
        results["home_score"],
        errors="coerce",
    )

    completed = (
        results["away_score"].notna()
        & results["home_score"].notna()
    )

    non_ties = (
        results["away_score"]
        != results["home_score"]
    )

    results = results[
        completed & non_ties
    ].copy()

    results["actual_winner"] = (
        results["away_team"]
    )

    home_wins = (
        results["home_score"]
        > results["away_score"]
    )

    results.loc[
        home_wins,
        "actual_winner",
    ] = results.loc[
        home_wins,
        "home_team",
    ]

    results["correct"] = (
        results["predicted_winner"]
        == results["actual_winner"]
    )

    return results


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="site-title">'
    'NFL Game Predictor'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="site-subtitle">
        Weekly NFL win probabilities with transparent
        matchup analytics.
    </div>
    """,
    unsafe_allow_html=True,
)

if last_updated is not None:

    updated_date = (
        last_updated.strftime(
            "%B %d, %Y"
        ).replace(
            " 0",
            " ",
        )
    )

    updated_time = (
        last_updated.strftime(
            "%I:%M %p"
        ).lstrip(
            "0"
        )
    )

    st.markdown(
        f"""
        <div class="last-updated">
            Last updated: {updated_date} at {updated_time} ET
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# WEEK SELECTOR
# ---------------------------------------------------------

weeks = sorted(
    schedule[
        "week"
    ].dropna().astype(
        int
    ).unique()
)


upcoming = schedule[
    schedule["home_score"].isna()
    | schedule["away_score"].isna()
]

prediction_week = (
    int(upcoming["week"].min())
    if not upcoming.empty
    else int(schedule["week"].max())
)


if (
    "selected_week" not in st.session_state
    or st.session_state.selected_week not in weeks
):

    st.session_state.selected_week = (
        prediction_week
        if prediction_week in weeks
        else weeks[0]
    )


current_week_index = weeks.index(
    st.session_state.selected_week
)


left_col, week_col, right_col = st.columns(
    [0.55, 2.2, 0.55],
    vertical_alignment="center",
)


with left_col:

    previous_clicked = st.button(
        "←",
        key="previous_week",
        use_container_width=True,
        disabled=current_week_index == 0,
        help="Previous week",
    )


with week_col:

    st.markdown(
        f"""
        <div class="week-nav-title">
            2026 NFL Week {st.session_state.selected_week}
        </div>
        """,
        unsafe_allow_html=True,
    )


with right_col:

    next_clicked = st.button(
        "→",
        key="next_week",
        use_container_width=True,
        disabled=current_week_index == len(weeks) - 1,
        help="Next week",
    )


if previous_clicked:

    st.session_state.selected_week = weeks[
        current_week_index - 1
    ]

    st.rerun()


if next_clicked:

    st.session_state.selected_week = weeks[
        current_week_index + 1
    ]

    st.rerun()


selected_week = st.session_state.selected_week


# ---------------------------------------------------------
# WEEK + SEASON RECORD
# ---------------------------------------------------------

prediction_results = build_prediction_results(
    schedule,
    predictions,
)

week_results = prediction_results[
    prediction_results["week"] == selected_week
].copy()

week_correct = int(
    week_results["correct"].sum()
)

week_total = int(
    len(week_results)
)

season_correct = int(
    prediction_results["correct"].sum()
)

season_total = int(
    len(prediction_results)
)

st.markdown(
    f"""
    <div class="record-row">
        <div class="record-pill">
            <span class="record-label">
                Week
            </span>
            <span class="record-value">
                {week_correct}/{week_total}
            </span>
        </div>
        <div class="record-pill">
            <span class="record-label">
                Season
            </span>
            <span class="record-value">
                {season_correct}/{season_total}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# PRESEASON / DATA NOTE
# ---------------------------------------------------------

if (
    selected_week == 1
    and prediction_week == 1
):

    st.info(
        "Week 1 model note: no 2026 regular-season "
        "play-by-play is available yet, so current team "
        "strength is based on exponentially weighted ending "
        "2025 performance. The model does not yet explicitly "
        "adjust for 2026 offseason roster, quarterback, "
        "coaching, or injury changes."
    )


# ---------------------------------------------------------
# WEEK DATA
# ---------------------------------------------------------

week_schedule = schedule[
    schedule["week"] == selected_week
].copy()


# ---------------------------------------------------------
# BYE WEEK SUMMARY
# ---------------------------------------------------------

all_teams = set(
    TEAM_NAMES.keys()
)

teams_playing = set(
    week_schedule[
        "away_team"
    ].dropna()
).union(
    set(
        week_schedule[
            "home_team"
        ].dropna()
    )
)

bye_teams = sorted(
    all_teams - teams_playing,
    key=lambda abbr: team_name(
        abbr
    ),
)

if bye_teams:

    bye_text = ", ".join(
        team_name(
            abbr
        )
        for abbr in bye_teams
    )

else:

    bye_text = "None"


st.markdown(
    f"""
    <div class="bye-week-banner">
        <span class="bye-week-label">
            Teams with Bye Week:
        </span>
        <span class="bye-week-teams">
            {bye_text}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


if predictions is not None:

    week_predictions = predictions[
        predictions["week"] == selected_week
    ].copy()

else:

    week_predictions = pd.DataFrame()


prediction_lookup = {}

if not week_predictions.empty:

    prediction_lookup = (
        week_predictions
        .set_index("game_id")
        .to_dict("index")
    )


# ---------------------------------------------------------
# DISPLAY GAMES
# ---------------------------------------------------------

for _, game in week_schedule.iterrows():

    game_id = game["game_id"]

    away = game["away_team"]
    home = game["home_team"]

    away_name = team_name(
        away
    )

    home_name = team_name(
        home
    )

    away_color = team_color(
        away
    )

    home_color = team_color(
        home
    )

    away_text_color = readable_text_color(
        away_color
    )

    home_text_color = readable_text_color(
        home_color
    )

    away_logo_html = logo_html(
        away,
        away_name,
    )

    home_logo_html = logo_html(
        home,
        home_name,
    )

    gameday = pd.to_datetime(
        game["gameday"]
    )

    date_text = gameday.strftime(
        "%A, %B %d"
    )

    gametime = format_game_time(
        game.get(
            "gametime"
        )
    )

    prediction = prediction_lookup.get(
        game_id
    )

    st.markdown(
        f"""
        <div
            class="matchup-accent"
            style="
                background:
                    linear-gradient(
                        90deg,
                        {away_color} 0%,
                        {away_color} 48%,
                        {home_color} 52%,
                        {home_color} 100%
                    );
            "
        ></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="game-info">
            {date_text} • {gametime}
        </div>
        """,
        unsafe_allow_html=True,
    )

    away_score = pd.to_numeric(
        game.get(
            "away_score"
        ),
        errors="coerce",
    )

    home_score = pd.to_numeric(
        game.get(
            "home_score"
        ),
        errors="coerce",
    )

    if (
        pd.notna(away_score)
        and pd.notna(home_score)
    ):

        st.markdown(
            f"""
            <div class="final-score">
                FINAL: {away} {int(away_score)}
                • {home} {int(home_score)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    away_col, middle_col, home_col = st.columns(
        [1, 0.30, 1]
    )

    with away_col:

        st.markdown(
            f"""
            <div class="team-logo-wrap">
                {away_logo_html}
            </div>
            <div
                class="team-badge"
                style="
                    background: {away_color};
                    color: {away_text_color};
                "
            >
                {away}
            </div>
            <div
                class="team-name"
                style="color: {away_color};"
            >
                {away_name}
            </div>
            <div class="team-abbr">
                Away
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prediction is not None:

            st.markdown(
                f"""
                <div
                    class="probability"
                    style="color: {away_color};"
                >
                    {prediction["away_win_probability"]:.0%}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with middle_col:

        st.markdown(
            """
            <div class="versus">@</div>
            """,
            unsafe_allow_html=True,
        )

    with home_col:

        st.markdown(
            f"""
            <div class="team-logo-wrap">
                {home_logo_html}
            </div>
            <div
                class="team-badge"
                style="
                    background: {home_color};
                    color: {home_text_color};
                "
            >
                {home}
            </div>
            <div
                class="team-name"
                style="color: {home_color};"
            >
                {home_name}
            </div>
            <div class="team-abbr">
                Home
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prediction is not None:

            st.markdown(
                f"""
                <div
                    class="probability"
                    style="color: {home_color};"
                >
                    {prediction["home_win_probability"]:.0%}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if prediction is not None:

        winner_abbr = prediction[
            "predicted_winner"
        ]

        away_probability = float(
            prediction[
                "away_win_probability"
            ]
        )

        home_probability = float(
            prediction[
                "home_win_probability"
            ]
        )

        winner_probability = max(
            away_probability,
            home_probability,
        )

        winner_color = (
            home_color
            if winner_abbr == home
            else away_color
        )

        winner_text_color = readable_text_color(
            winner_color
        )

        winner_logo_html = logo_html(
            winner_abbr,
            team_name(
                winner_abbr
            ),
            css_class="winner-logo",
        )

        st.markdown(
            f"""
            <div class="prediction-bar">
                <div
                    class="prediction-segment"
                    style="
                        width: {away_probability * 100:.1f}%;
                        background: {away_color};
                    "
                ></div>
                <div
                    class="prediction-segment"
                    style="
                        width: {home_probability * 100:.1f}%;
                        background: {home_color};
                    "
                ></div>
            </div>

            <div class="winner-label">
                Predicted winner
            </div>

            <div
                class="winner-chip"
                style="
                    background: {winner_color};
                    color: {winner_text_color};
                "
            >
                {winner_logo_html}
                {team_name(winner_abbr)} • {winner_probability:.0%}
            </div>

            <div class="prediction-confidence">
                Model win probability
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # ADVANCED ANALYTICS
        # -------------------------------------------------

        with st.expander(
            "View matchup analytics"
        ):

            rows = [
                {
                    "Metric":
                        "Pass EPA / play",
                    away:
                        fmt_epa(
                            prediction[
                                "away_pass_epa"
                            ]
                        ),
                    home:
                        fmt_epa(
                            prediction[
                                "home_pass_epa"
                            ]
                        ),
                    "Advantage":
                        advantage(
                            prediction[
                                "away_pass_epa"
                            ],
                            prediction[
                                "home_pass_epa"
                            ],
                            away,
                            home,
                        ),
                    "Model":
                        "Used",
                },
                {
                    "Metric":
                        "Rush EPA / play",
                    away:
                        fmt_epa(
                            prediction[
                                "away_rush_epa"
                            ]
                        ),
                    home:
                        fmt_epa(
                            prediction[
                                "home_rush_epa"
                            ]
                        ),
                    "Advantage":
                        advantage(
                            prediction[
                                "away_rush_epa"
                            ],
                            prediction[
                                "home_rush_epa"
                            ],
                            away,
                            home,
                        ),
                    "Model":
                        "Used",
                },
                {
                    "Metric":
                        "Defensive EPA / play",
                    away:
                        fmt_epa(
                            prediction[
                                "away_def_epa"
                            ]
                        ),
                    home:
                        fmt_epa(
                            prediction[
                                "home_def_epa"
                            ]
                        ),
                    "Advantage":
                        advantage(
                            prediction[
                                "away_def_epa"
                            ],
                            prediction[
                                "home_def_epa"
                            ],
                            away,
                            home,
                        ),
                    "Model":
                        "Used",
                },
                {
                    "Metric":
                        "Success rate",
                    away:
                        fmt_percent(
                            prediction[
                                "away_success_rate"
                            ]
                        ),
                    home:
                        fmt_percent(
                            prediction[
                                "home_success_rate"
                            ]
                        ),
                    "Advantage":
                        advantage(
                            prediction[
                                "away_success_rate"
                            ],
                            prediction[
                                "home_success_rate"
                            ],
                            away,
                            home,
                        ),
                    "Model":
                        "Used",
                },
                {
                    "Metric":
                        "Rest",
                    away:
                        fmt_days(
                            prediction[
                                "away_rest"
                            ]
                        ),
                    home:
                        fmt_days(
                            prediction[
                                "home_rest"
                            ]
                        ),
                    "Advantage":
                        advantage(
                            prediction[
                                "away_rest"
                            ],
                            prediction[
                                "home_rest"
                            ],
                            away,
                            home,
                        ),
                    "Model":
                        "Used",
                },
                {
                    "Metric":
                        "Overall offensive EPA",
                    away:
                        fmt_epa(
                            prediction[
                                "away_off_epa"
                            ]
                        ),
                    home:
                        fmt_epa(
                            prediction[
                                "home_off_epa"
                            ]
                        ),
                    "Advantage":
                        advantage(
                            prediction[
                                "away_off_epa"
                            ],
                            prediction[
                                "home_off_epa"
                            ],
                            away,
                            home,
                        ),
                    "Model":
                        "Context",
                },
                {
                    "Metric":
                        "Pace",
                    away:
                        fmt_number(
                            prediction[
                                "away_pace"
                            ]
                        ),
                    home:
                        fmt_number(
                            prediction[
                                "home_pace"
                            ]
                        ),
                    "Advantage":
                        "—",
                    "Model":
                        "Context",
                },
            ]

            metrics_df = pd.DataFrame(
                rows
            )

            st.dataframe(
                metrics_df,
                hide_index=True,
                use_container_width=True,
            )

            st.markdown(
                """
                <div class="metric-note">
                    “Used” metrics are inputs to the current
                    production model. “Context” metrics are
                    displayed for deeper football analysis
                    but are not used by Model C.
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.caption(
            "No model probability has been generated "
            "for this week yet."
        )



# ---------------------------------------------------------
# HOW THE MODEL WORKS
# ---------------------------------------------------------

st.divider()

with st.expander(
    "How the prediction works"
):

    st.markdown(
        """
        The production model is a **logistic regression**
        trained on historical NFL games.

        The current version uses five matchup differences:

        - Passing EPA
        - Rushing EPA
        - Defensive EPA
        - Success rate
        - Rest days

        The model also learns a baseline through its
        intercept, which includes the historical tendency
        associated with the home team.

        The percentage shown on each matchup is the
        **final win probability after the model combines
        the baseline and all matchup features**.

        Team strength is calculated using exponentially
        weighted recent performance. Before a new season
        has enough games, the previous season acts as a
        prior and is gradually replaced by current-season
        results.
        """
    )


# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div
        style="
            color: #F5F7FA;
            font-size: 2rem;
            font-weight: 800;
            margin: 0.35rem 0 0.75rem 0;
        "
    >
        Model Performance
    </div>
    """,
    unsafe_allow_html=True,
)

if model_bundle is not None:

    performance_tab, explanation_tab = st.tabs(
        [
            "Performance",
            "What do these numbers mean?",
        ]
    )

    with performance_tab:

        col1, col2, col3, col4 = st.columns(
            4
        )

        col1.metric(
            "Prediction Accuracy",
            f"{model_bundle.get('walk_forward_avg_accuracy', 0):.1%}",
        )

        col2.metric(
            "ROC AUC",
            f"{model_bundle.get('walk_forward_avg_roc_auc', 0):.3f}",
        )

        col3.metric(
            "Log Loss",
            f"{model_bundle.get('walk_forward_avg_log_loss', 0):.3f}",
        )

        col4.metric(
            "Brier Score",
            f"{model_bundle.get('calibration_brier_score', 0):.3f}",
        )

        st.caption(
            "Tested on games the model had not seen during training."
        )

    with explanation_tab:

        st.markdown(
            """
            

            **Prediction Accuracy**

            The percentage of games where the model picked the correct
            winner.

            For example, an accuracy of **68%** means the model
            picked about 68 winners correctly out of every 100 games.

            **ROC AUC**

            This measures how well the model separates stronger
            winning predictions from weaker losing predictions.

            - **0.50** is about the same as random guessing.
            - **1.00** would be perfect.

            **Log Loss**

            This measures the quality of the model's probabilities,
            not just whether it picked the correct winner.

            A confident wrong prediction is penalized more heavily
            than a close prediction that turns out to be wrong.

            Lower is better. A model that predicted every game
            as exactly 50/50 would have a Log Loss of about **0.693**.

            **Brier Score**

            Another way to measure how close the predicted
            probabilities were to what actually happened.

            - **0.00** would be perfect.
            - **0.25** is what you would get by predicting every game
              as 50/50.
            - Lower is better.
            """
        )


if calibration is not None:

    with st.expander(
        "Historical probability calibration"
    ):

        display_calibration = calibration.copy()

        display_calibration[
            "avg_model_confidence"
        ] = (
            display_calibration[
                "avg_model_confidence"
            ]
            * 100
        ).map(
            "{:.1f}%".format
        )

        display_calibration[
            "actual_favorite_win_rate"
        ] = (
            display_calibration[
                "actual_favorite_win_rate"
            ]
            * 100
        ).map(
            "{:.1f}%".format
        )

        display_calibration = (
            display_calibration.rename(
                columns={
                    "confidence_bucket":
                        "Prediction range",
                    "games":
                        "Games",
                    "avg_model_confidence":
                        "Average model confidence",
                    "actual_favorite_win_rate":
                        "Actual win rate",
                }
            )
        )

        st.dataframe(
            display_calibration,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "This compares the model's out-of-sample "
            "confidence with how often the predicted "
            "favorite actually won."
        )