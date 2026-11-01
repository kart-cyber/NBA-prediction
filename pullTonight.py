"""
Step 1 of the NBA prop predictor:
Pull tonight's games and each player's last 10 games for the teams playing.

Run:  python pull_tonight.py
Output: tonights_games.csv, recent_logs.csv, player_summary.csv
"""
import pandas as pd
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamelog

SEASON = "2026-27"
LAST_N = 10
STATS = ["MIN", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]


def get_tonights_games() -> pd.DataFrame:
    """Today's schedule from NBA.com's live scoreboard."""
    games = scoreboard.ScoreBoard().get_dict()["scoreboard"]["games"]
    return pd.DataFrame([
        {
            "game_id": g["gameId"],
            "away": g["awayTeam"]["teamTricode"],
            "home": g["homeTeam"]["teamTricode"],
            "status": g["gameStatusText"],
        }
        for g in games
    ])


def get_season_player_logs(season: str) -> pd.DataFrame:
    """Every player's game log for the season in ONE request (avoids rate limits)."""
    log = leaguegamelog.LeagueGameLog(
        season=season, player_or_team_abbreviation="P", timeout=60
    )
    df = log.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values("GAME_DATE")


def main():
    games = get_tonights_games()
    if games.empty:
        print("No games today.")
        return
    print("Tonight's games:")
    print(games.to_string(index=False), "\n")

    teams_tonight = set(games["away"]) | set(games["home"])
    logs = get_season_player_logs(SEASON)

    # Use each player's most recent team (handles trades)
    current_team = logs.groupby("PLAYER_ID")["TEAM_ABBREVIATION"].last()
    active_ids = current_team[current_team.isin(teams_tonight)].index
    logs = logs[logs["PLAYER_ID"].isin(active_ids)]

    # Last N games per player
    recent = logs.groupby("PLAYER_ID").tail(LAST_N)

    summary = (
        recent.groupby(["PLAYER_ID", "PLAYER_NAME"])
        .agg(TEAM=("TEAM_ABBREVIATION", "last"),
             GP=("GAME_ID", "count"),
             **{s: (s, "mean") for s in STATS})
        .round(1)
        .reset_index()
        .sort_values(["TEAM", "MIN"], ascending=[True, False])
    )

    games.to_csv("tonights_games.csv", index=False)
    recent.to_csv("recent_logs.csv", index=False)
    summary.to_csv("player_summary.csv", index=False)

    print(f"Last {LAST_N}-game averages (top 15 by minutes):")
    print(summary.sort_values("MIN", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()