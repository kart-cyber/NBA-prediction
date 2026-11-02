"""
Step 2 of the NBA prop predictor:
Add matchup context (opponent, home/away, opponent defense & pace, rest days)
to every player whose team plays tonight.

Run:  python add_matchups.py
Output: matchup_summary.csv, then a PROP FINDER prompt (see props.py)

Requests made (6 total, 0.6 s apart):
  1. ScoreboardV3           -> tonight's games + team IDs          (NBA.com)
  2. LeagueGameLog          -> every player's game log (reused from pullTonight.py)
  3. LeagueDashTeamStats    -> every team's advanced stats (DEF_RATING, PACE)
  4. ESPN injury report     -> who is Out / Day-To-Day             (ESPN)
  5. LeagueGameLog          -> LAST season's game logs (for head-to-head history)
  6. LeagueDashTeamStats    -> what each team ALLOWS per game (opponent stats)
"""
import sys
import time

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguedashteamstats, scoreboardv3

# Reuse settings and the game-log function from step 1 (we don't modify that file)
from pullTonight import SEASON, LAST_N, get_season_player_logs
# The prop finder lives in its own file to keep this one readable
from props import normalize_name, prepare_history, run_prop_finder

REQUEST_DELAY = 0.6  # seconds to wait between NBA.com requests
TIMEOUT = 60         # seconds before giving up on a single request

# ESPN's public injury feed. It's unofficial/undocumented, so it could change
# without warning - if it fails, the script still runs, just without injuries.
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def get_tonights_games_with_ids() -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Tonight's games from the live scoreboard, including team IDs.

    pullTonight.get_tonights_games() only keeps team abbreviations, so we
    read the scoreboard ourselves to also get the team IDs (more reliable
    for matching than abbreviations). Returns (games, today's date).
    """
    # The live scoreboard keeps showing yesterday's games until late morning
    # ET, so ask for today's date (US Eastern, like the NBA schedule) directly.
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    board = scoreboardv3.ScoreboardV3(
        game_date=today.strftime("%Y-%m-%d"), timeout=TIMEOUT
    ).get_dict()["scoreboard"]

    games = pd.DataFrame([
        {
            "GAME_ID": g["gameId"],
            "AWAY_ID": g["awayTeam"]["teamId"],
            "AWAY": g["awayTeam"]["teamTricode"],
            "HOME_ID": g["homeTeam"]["teamId"],
            "HOME": g["homeTeam"]["teamTricode"],
            "STATUS": g["gameStatusText"].strip(),
        }
        for g in board["games"]
    ])
    return games, today


def build_team_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """
    Turn one row per GAME into one row per TEAM playing tonight.

    Example: the game "POR @ UTA" becomes two rows:
        POR  -> opponent UTA, Away
        UTA  -> opponent POR, Home
    """
    away = pd.DataFrame({
        "GAME_ID": games["GAME_ID"],
        "TEAM_ID": games["AWAY_ID"],
        "OPP_ID": games["HOME_ID"],
        "OPPONENT": games["HOME"],
        "HOME_AWAY": "Away",
        "GAME": games["AWAY"] + " @ " + games["HOME"],
        "STATUS": games["STATUS"],
    })
    home = pd.DataFrame({
        "GAME_ID": games["GAME_ID"],
        "TEAM_ID": games["HOME_ID"],
        "OPP_ID": games["AWAY_ID"],
        "OPPONENT": games["AWAY"],
        "HOME_AWAY": "Home",
        "GAME": games["AWAY"] + " @ " + games["HOME"],
        "STATUS": games["STATUS"],
    })
    return pd.concat([away, home], ignore_index=True)


def get_team_advanced_stats(season: str) -> pd.DataFrame:
    """
    Every team's defensive rating and pace in ONE request, plus ranks out of 30.

    - OPP_DEF_RANK: 1 = best defense (LOWEST defensive rating = fewest points allowed)
    - OPP_PACE_RANK: 1 = fastest pace (MOST possessions per 48 minutes)
    """
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        timeout=TIMEOUT,
    ).get_data_frames()[0]

    teams = stats[["TEAM_ID", "DEF_RATING", "PACE"]].copy()
    teams["DEF_RANK"] = teams["DEF_RATING"].rank(ascending=True, method="min").astype(int)
    teams["PACE_RANK"] = teams["PACE"].rank(ascending=False, method="min").astype(int)

    # Rename so it's clear these describe the OPPONENT once merged in
    return teams.rename(columns={
        "TEAM_ID": "OPP_ID",
        "DEF_RATING": "OPP_DEF_RATING",
        "PACE": "OPP_PACE",
        "DEF_RANK": "OPP_DEF_RANK",
        "PACE_RANK": "OPP_PACE_RANK",
    })


def get_rest_days(logs: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """
    Days since each team last played, using the game logs we already have
    (no extra requests).

    REST_DAYS = today's date minus the team's most recent game date.
        played yesterday  -> 1  (back-to-back)
        played 2 days ago -> 2  (one full day off)
    """
    # Only look at games BEFORE today, in case a game in progress/finished
    # today already shows up in the logs.
    past = logs[logs["GAME_DATE"] < today]

    last_game = past.groupby("TEAM_ID")["GAME_DATE"].max().reset_index()
    last_game = last_game.rename(columns={"GAME_DATE": "LAST_GAME_DATE"})
    last_game["REST_DAYS"] = (today - last_game["LAST_GAME_DATE"]).dt.days
    last_game["BACK_TO_BACK"] = last_game["REST_DAYS"] == 1
    return last_game


def previous_season(season: str) -> str:
    """'2026-27' -> '2025-26'"""
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def get_opponent_allowed(season: str) -> pd.DataFrame:
    """
    What every team ALLOWS per game (opponents' PTS, REB, AST, 3PM, STL, BLK)
    in ONE request. Used by the prop finder to judge each matchup.
    """
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Opponent",
        per_mode_detailed="PerGame",   # the default is season TOTALS
        timeout=TIMEOUT,
    ).get_data_frames()[0]
    return stats[["TEAM_ID", "OPP_PTS", "OPP_REB", "OPP_AST",
                  "OPP_FG3M", "OPP_STL", "OPP_BLK"]]


def get_injury_report() -> pd.DataFrame:
    """
    Today's injury report from ESPN in ONE request.

    Returns one row per injured player with:
      INJURY_STATUS  "Out" or "Day-To-Day" (ESPN's labels)
      INJURY         body part, e.g. "Knee"
      RETURN_DATE    ESPN's estimated return date (if any)
      INJURY_NOTE    ESPN's short news blurb
    """
    response = requests.get(ESPN_INJURIES_URL, timeout=TIMEOUT)
    response.raise_for_status()  # turn HTTP errors (404, 500...) into exceptions

    rows = []
    for team in response.json()["injuries"]:          # one entry per team
        for injury in team["injuries"]:               # one entry per player
            details = injury.get("details", {})
            rows.append({
                "NAME_KEY": normalize_name(injury["athlete"]["displayName"]),
                "INJURY_STATUS": injury["status"],
                "INJURY": details.get("type", ""),
                "RETURN_DATE": details.get("returnDate", ""),
                "INJURY_NOTE": injury.get("shortComment", ""),
            })

    columns = ["NAME_KEY", "INJURY_STATUS", "INJURY", "RETURN_DATE", "INJURY_NOTE"]
    return pd.DataFrame(rows, columns=columns).drop_duplicates("NAME_KEY")


def main():
    # ---- Request 1: tonight's games -------------------------------------
    try:
        games, today = get_tonights_games_with_ids()
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        sys.exit(f"Couldn't load tonight's scoreboard from NBA.com ({e}). "
                 "Check your internet connection and try again.")

    if games.empty:
        print(f"No NBA games scheduled for {today.date()}. Nothing to do.")
        return

    print(f"Tonight ({today.date()}): {len(games)} games\n")
    team_schedule = build_team_schedule(games)

    # ---- Request 2: every player's season game log ----------------------
    time.sleep(REQUEST_DELAY)
    try:
        logs = get_season_player_logs(SEASON)
    except requests.exceptions.Timeout:
        sys.exit("NBA.com timed out loading player game logs. "
                 "It's often slow - wait a minute and run the script again.")
    except requests.exceptions.RequestException as e:
        sys.exit(f"Couldn't load player game logs from NBA.com ({e}).")

    # Drop today's games: a game in progress shows up in the logs with
    # partial stats (e.g. 18 minutes in the 3rd quarter), which would drag
    # averages down. Only finished days count.
    logs = logs[logs["GAME_DATE"] < today]
    if logs.empty:
        sys.exit(f"No player game logs found for {SEASON} yet.")

    # ---- Request 3: team advanced stats (defense + pace) ----------------
    time.sleep(REQUEST_DELAY)
    try:
        team_stats = get_team_advanced_stats(SEASON)
    except requests.exceptions.Timeout:
        sys.exit("NBA.com timed out loading team stats. "
                 "Wait a minute and run the script again.")
    except requests.exceptions.RequestException as e:
        sys.exit(f"Couldn't load team stats from NBA.com ({e}).")

    # ---- Request 4: ESPN injury report ----------------------------------
    # Unlike the requests above, a failure here is NOT fatal: injuries are
    # useful extra info, so we warn and carry on with status "Unknown".
    time.sleep(REQUEST_DELAY)
    try:
        injuries = get_injury_report()
        no_injury_label = "Active"   # not on the report = expected to play
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"WARNING: couldn't load ESPN's injury report ({e}).\n"
              "         Continuing without injuries - check statuses manually!\n")
        injuries = pd.DataFrame(columns=["NAME_KEY", "INJURY_STATUS", "INJURY",
                                         "RETURN_DATE", "INJURY_NOTE"])
        no_injury_label = "Unknown"  # we couldn't check, so don't claim "Active"

    # ---- Which players are playing tonight? -----------------------------
    # Use each player's most recent TEAM_ID (handles trades), then keep only
    # players whose team ID is in tonight's schedule.
    current_team = logs.groupby("PLAYER_ID")["TEAM_ID"].last().rename("TEAM_ID")
    tonight_ids = set(team_schedule["TEAM_ID"])
    active = current_team[current_team.isin(tonight_ids)]
    # Only keep each player's games for their CURRENT team
    logs_tonight = logs.merge(active.reset_index(), on=["PLAYER_ID", "TEAM_ID"])

    # Last-N-game averages (same idea as pullTonight.py)
    recent = logs_tonight.groupby("PLAYER_ID").tail(LAST_N)
    players = (
        recent.groupby(["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"])
        .agg(TEAM=("TEAM_ABBREVIATION", "last"),
             GP=("GAME_ID", "count"),
             MIN=("MIN", "mean"),
             PTS=("PTS", "mean"),
             REB=("REB", "mean"),
             AST=("AST", "mean"))
        .round(1)
        .reset_index()
    )

    # ---- Add matchup context (all joins are on team IDs) ----------------
    matchups = (
        players
        .merge(team_schedule, on="TEAM_ID", how="left")   # opponent, home/away
        .merge(team_stats, on="OPP_ID", how="left")       # opponent def/pace
        .merge(get_rest_days(logs, today), on="TEAM_ID", how="left")  # rest
    )

    # Injuries are the one join done on NAMES (ESPN has different player IDs).
    # Players not on ESPN's report get "Active" (or "Unknown" if ESPN failed).
    matchups["NAME_KEY"] = matchups["PLAYER_NAME"].map(normalize_name)
    matchups = matchups.merge(injuries, on="NAME_KEY", how="left")
    matchups["INJURY_STATUS"] = matchups["INJURY_STATUS"].fillna(no_injury_label)
    matchups[["INJURY", "RETURN_DATE", "INJURY_NOTE"]] = (
        matchups[["INJURY", "RETURN_DATE", "INJURY_NOTE"]].fillna("")
    )

    # Sort by game, then by minutes (most first) within each game
    matchups = matchups.sort_values(["GAME_ID", "MIN"], ascending=[True, False])

    # The CSV keeps EVERY player (including Out) so nothing is silently lost.
    # Filter on INJURY_STATUS != "Out" before making predictions.
    columns = ["GAME", "STATUS", "PLAYER_NAME", "TEAM", "HOME_AWAY", "OPPONENT",
               "INJURY_STATUS", "INJURY", "RETURN_DATE", "INJURY_NOTE",
               "GP", "MIN", "PTS", "REB", "AST",
               "OPP_DEF_RATING", "OPP_DEF_RANK", "OPP_PACE", "OPP_PACE_RANK",
               "REST_DAYS", "BACK_TO_BACK",
               "PLAYER_ID", "TEAM_ID", "OPP_ID", "GAME_ID", "LAST_GAME_DATE"]
    matchups = matchups[columns]
    matchups.to_csv("matchup_summary.csv", index=False)

    # ---- Print a readable table, one block per game ---------------------
    # Players who are Out are listed underneath each table instead of in it.
    show = ["PLAYER_NAME", "TEAM", "HOME_AWAY", "OPPONENT", "INJURY_STATUS",
            "MIN", "PTS", "OPP_DEF_RATING", "OPP_DEF_RANK", "OPP_PACE",
            "OPP_PACE_RANK", "REST_DAYS", "BACK_TO_BACK"]
    for (game, status), block in matchups.groupby(["GAME", "STATUS"], sort=False):
        is_out = block["INJURY_STATUS"] == "Out"
        print(f"=== {game}  ({status}) ===")
        print(block.loc[~is_out, show].to_string(index=False))
        if is_out.any():
            out = block[is_out]
            names = ", ".join(f"{p} ({t}, {i or 'injury'})" for p, t, i
                              in zip(out["PLAYER_NAME"], out["TEAM"], out["INJURY"]))
            print(f"  OUT tonight: {names}")
        print()

    n_out = (matchups["INJURY_STATUS"] == "Out").sum()
    print(f"Saved {len(matchups)} players to matchup_summary.csv "
          f"({n_out} marked Out - filter INJURY_STATUS != 'Out' before predicting)")

    # ---- Extra data for the prop finder (requests 5 and 6) --------------
    # Both are optional: if one fails, the prop finder still works, just
    # with less information (no head-to-head history / no opponent factor).
    time.sleep(REQUEST_DELAY)
    try:
        last_season_logs = get_season_player_logs(previous_season(SEASON))
    except requests.exceptions.RequestException as e:
        print(f"\nWARNING: couldn't load last season's logs ({e}). "
              "Head-to-head history will only use this season.")
        last_season_logs = logs.iloc[0:0]

    time.sleep(REQUEST_DELAY)
    try:
        opp_allowed = get_opponent_allowed(SEASON)
    except requests.exceptions.RequestException as e:
        print(f"\nWARNING: couldn't load opponent stats ({e}). "
              "Projections won't adjust for the opponent's defense.")
        opp_allowed = pd.DataFrame()

    # Both seasons together = up to 2 years of games per player
    history = prepare_history(pd.concat([last_season_logs, logs], ignore_index=True))
    current_season_id = logs["SEASON_ID"].iloc[-1]   # e.g. "22026"

    run_prop_finder(matchups, history, opp_allowed, current_season_id)


if __name__ == "__main__":
    main()
