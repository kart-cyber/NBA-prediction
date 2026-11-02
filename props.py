"""
Step 3 of the NBA prop predictor: the prop finder.

Used by add_matchups.py - after it builds tonight's matchup table, it calls
run_prop_finder() so you can type a player's name and get:

  - a projection for each prop (PTS, REB, AST, 3PM, PRA, ...)
  - how tonight's injuries (his team AND the opponent's) change it
  - the best 3 "legs" for tonight
  - a check of any sportsbook line you type in, e.g.  pts 27.5

How a projection is built (all from data we already downloaded):

  0. CLEAN HIS GAMES so one weird night can't swing everything:
       - drop games where he played under 60% of his usual minutes
         (early injury exits, blowouts)
       - cap remaining games at 40%-160% of his average for stats he
         averages 3+ in (so a 0-point or 41-point night counts as ~9 / ~35);
         with 5+ games this season, "his average" means this season's
     Hit rates ("went over 6/10") still use his REAL results.

  1. BASE = a weighted average of:
       L10      his last 10 games                                 (weight 0.6)
       SEASON   his 2026-27 average                               (weight 0.4)
       H2H      his games vs tonight's opponent (2 seasons, 2+ games) (0.2)
     Weights are re-scaled to add up to 1, so missing pieces just drop out.
     Once he has 5+ games this season, L10 only uses this season's games -
     last season's role/minutes can be very different (e.g. end-of-season
     rest games). Before that, L10 fills in with last season's games.
     This is done for PTS, REB, AST, 3PM, STL and BLK. Combos (PRA, PA, ...)
     are the sum of their parts.

  2. x OPPONENT FACTOR = how much the opponent allows of that stat per game
     vs the league average, using only HALF the difference (small samples).

  3. x TEAMMATE INJURY FACTOR - for each injured starter on his team:
       ESTIMATE  his share of the team's production x half of what the
                 injured player normally produces (the rest of it is lost -
                 teams score less without stars)
       ACTUAL    how he really did in games without that teammate
     With few "without" games we trust the estimate; with many we trust what
     actually happened (a weighted blend). If his recent averages ALREADY
     include games without that teammate, only the part that isn't already
     counted is applied (no double counting).

  4. x OPPONENT INJURY FACTOR - for each injured starter on the opponent:
     how much more/less that team allowed in games without him. Starts at
     "no effect" and only moves as real games without him pile up.

  5. x 0.97 if his team is on a back-to-back (tired legs).

How the "best legs" are picked (we don't have real sportsbook lines):
  - ESTIMATED LINE = his recent average (L10 + season) rounded to a .5 line,
    since books set lines close to recent averages.
  - The side (OVER/UNDER) is whichever side our projection is on.
  - CONFIDENCE = half the model's estimated chance of hitting, half how often
    he actually hit it in his last 10 games. A leg needs BOTH behind it.

This is a simple, transparent model - not a guarantee. Always compare it
with the real sportsbook line (type e.g. "pts 27.5") before betting.
"""
import difflib
import math
import re
import unicodedata

import pandas as pd

# The six box-score stats we project directly
BASE_STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]

# Each prop and the box-score columns that add up to it
PROPS = {
    "PTS": ["PTS"],
    "REB": ["REB"],
    "AST": ["AST"],
    "3PM": ["FG3M"],
    "STL": ["STL"],
    "BLK": ["BLK"],
    "PRA": ["PTS", "REB", "AST"],
    "PR":  ["PTS", "REB"],
    "PA":  ["PTS", "AST"],
    "RA":  ["REB", "AST"],
}

# --- Base projection ---
WEIGHTS = {"L10": 0.6, "SEASON": 0.4, "H2H": 0.2}
MIN_H2H_GAMES = 2        # need at least this many games vs the opponent
MIN_SEASON_GAMES = 5     # with this many games this season, L10 ignores last season
MATCHUP_WEIGHT = 0.5     # use half of the opponent's above/below-average effect
BACK_TO_BACK_FACTOR = 0.97

# --- Cleaning outlier games ---
MIN_GAME_SHARE = 0.6     # drop games under 60% of his usual minutes
CAP_LOW, CAP_HIGH = 0.4, 1.6   # cap games at 40%-160% of his average...
CAP_MIN_AVG = 3          # ...but only for stats he averages 3+ in

# --- Injuries ---
KEY_PLAYER_MIN = 24      # an injured player counts as a "starter" at 24+ min
# Shots, rebounds and passes get shared out when someone sits; steals and
# blocks don't really "transfer", and they're too small/noisy to adjust.
INJURY_STATS = ["PTS", "REB", "AST", "FG3M"]
REDISTRIBUTE = 0.5       # half of a missing player's production gets shared out
TEAMMATE_TRUST = 5       # the estimate counts like 5 games of real evidence
OPP_TRUST = 8            # "no effect" counts like 8 games for opponent injuries
# Never move a stat more than this. Injured players often sat out the SAME
# games, so multiplying their effects partly double counts - this caps it.
TEAMMATE_LIMITS = (0.80, 1.25)
OPP_LIMITS = (0.90, 1.15)

MIN_AVG_FOR_LEG = 1.5    # don't recommend tiny props (e.g. 0.4 blocks) - too random


def normalize_name(name: str) -> str:
    """
    Turn a player name into a simple key so names from ESPN and NBA.com match.

    ESPN and NBA.com use different player IDs, so we have to match on names.
    Small differences would break that, so we:
      - remove accents          "Vít Krejčí"          -> "vit krejci"
      - drop punctuation        "De'Aaron Fox"        -> "deaaron fox"
      - drop suffixes           "Terrence Shannon Jr" -> "terrence shannon"
    """
    # Split accented letters into letter + accent, then throw the accents away
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r"[^a-z ]", "", name.lower())  # keep only letters and spaces
    words = [w for w in name.split() if w not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(words)


# --------------------------------------------------------------------------
# Data preparation (done once, before the prompt starts)
# --------------------------------------------------------------------------
def add_prop_columns(games: pd.DataFrame) -> pd.DataFrame:
    """Add the combo columns (PRA, PR, ...) and the 3PM alias to a game log."""
    games = games.copy()
    for prop, cols in PROPS.items():
        games[prop] = games[cols].sum(axis=1)
    return games


def prepare_history(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Add prop columns (PRA, PR, ...) and the opponent's TEAM_ID to every game.

    The game log only has the opponent as text ("PHI @ CHA"), so we find it
    by ID instead: every game has two teams, and your opponent is the team
    in the same GAME_ID that isn't you.
    """
    logs = add_prop_columns(logs)

    team_games = logs[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    pairs = team_games.merge(team_games, on="GAME_ID", suffixes=("", "_OPP"))
    pairs = pairs[pairs["TEAM_ID"] != pairs["TEAM_ID_OPP"]]
    pairs = pairs.rename(columns={"TEAM_ID_OPP": "OPP_TEAM_ID"})

    logs = logs.merge(pairs, on=["GAME_ID", "TEAM_ID"], how="left")
    return logs.sort_values("GAME_DATE").reset_index(drop=True)


def build_allowed_by_game(history: pd.DataFrame) -> pd.DataFrame:
    """
    What each team ALLOWED in every game (the other team's totals).

    We add up each team's players per game, then swap the team columns:
    "PHI scored 112 vs CHA" becomes "CHA allowed 112 vs PHI".
    """
    totals = (history.groupby(["GAME_ID", "SEASON_ID", "TEAM_ID", "OPP_TEAM_ID"])
              [BASE_STATS].sum().reset_index())
    return totals.rename(columns={"TEAM_ID": "OPP_TEAM_ID",
                                  "OPP_TEAM_ID": "TEAM_ID"})


def opponent_factors(opp_allowed: pd.DataFrame, opp_id: int) -> dict:
    """
    For each stat: how much tonight's opponent allows vs the league average.

    1.00 = average, 1.10 = allows 10% more (good for overs), 0.90 = 10% less.
    Returns 1.0 for everything if the opponent stats couldn't be downloaded.
    """
    if opp_allowed.empty or opp_id not in set(opp_allowed["TEAM_ID"]):
        return {s: 1.0 for s in BASE_STATS}

    team = opp_allowed[opp_allowed["TEAM_ID"] == opp_id].iloc[0]
    league = opp_allowed.mean(numeric_only=True)
    return {s: 1 + (team["OPP_" + s] / league["OPP_" + s] - 1) * MATCHUP_WEIGHT
            for s in BASE_STATS}


# --------------------------------------------------------------------------
# Step 0: cleaning his games
# --------------------------------------------------------------------------
def clean_games(games: pd.DataFrame, current_season_id) -> pd.DataFrame:
    """Drop short-minute games and cap outliers (see step 0 at the top)."""
    if games.empty:
        return games
    usual_minutes = games.tail(15)["MIN"].median()
    games = games[games["MIN"] >= MIN_GAME_SHARE * usual_minutes].copy()

    # Caps come from this season once he has enough games, so a bigger role
    # this year isn't treated as a string of outliers
    season = games[games["SEASON_ID"] == current_season_id]
    recent = (season if len(season) >= MIN_SEASON_GAMES else games).tail(20)
    for s in BASE_STATS:
        avg = recent[s].mean()
        if avg >= CAP_MIN_AVG:
            games[s] = games[s].clip(lower=CAP_LOW * avg, upper=CAP_HIGH * avg)
    return add_prop_columns(games)   # re-add combos from the capped stats


# --------------------------------------------------------------------------
# Steps 3 and 4: injuries
# --------------------------------------------------------------------------
def starters_out(matchups: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """A team's players ruled Out tonight who normally play starter minutes."""
    return matchups[(matchups["TEAM_ID"] == team_id)
                    & (matchups["INJURY_STATUS"] == "Out")
                    & (matchups["MIN"] >= KEY_PLAYER_MIN)]


def blend(estimate: float, actual, n_games: int, trust: int) -> float:
    """
    Mix an estimate with real results. The estimate counts like `trust` games,
    so with 1 real game we mostly use the estimate; with 35 we mostly use
    what actually happened.
    """
    if actual is None or pd.isna(actual):
        return estimate
    return (trust * estimate + n_games * actual) / (trust + n_games)


def teammate_injury_impact(player, clean, base_games, history, matchups,
                           current_season_id) -> tuple[dict, list]:
    """
    Step 3: how tonight's injured starters on HIS team change his stats.
    Returns ({stat: total factor}, [details per injured teammate]).
    """
    team_id = player["TEAM_ID"]
    total = {s: 1.0 for s in BASE_STATS}
    details = []

    # This season's per-game averages for everyone on his team
    team_season = history[(history["SEASON_ID"] == current_season_id)
                          & (history["TEAM_ID"] == team_id)]
    averages = team_season.groupby("PLAYER_ID")[BASE_STATS].mean()
    out_tonight = set(matchups.loc[(matchups["TEAM_ID"] == team_id)
                                   & (matchups["INJURY_STATUS"] == "Out"),
                                   "PLAYER_ID"])
    healthy = averages[~averages.index.isin(out_tonight)]

    for _, mate in starters_out(matchups, team_id).iterrows():
        mate_games = history[(history["PLAYER_ID"] == mate["PLAYER_ID"])
                             & (history["TEAM_ID"] == team_id)]
        played_ids = set(mate_games["GAME_ID"])

        # His games from seasons when they were teammates, split with/without
        together = clean[(clean["TEAM_ID"] == team_id)
                         & clean["SEASON_ID"].isin(set(mate_games["SEASON_ID"]))]
        mate_played = together["GAME_ID"].isin(played_ids)
        with_mate, without_mate = together[mate_played], together[~mate_played]

        # Share of his base games the teammate ALREADY missed (already counted)
        already = (~base_games["GAME_ID"].isin(played_ids)).mean()

        factors = {s: 1.0 for s in BASE_STATS}
        for s in INJURY_STATS:
            his_avg = healthy[s].get(player["PLAYER_ID"], 0)
            if his_avg <= 0 or healthy[s].sum() <= 0:
                continue
            # ESTIMATE: he picks up his share of half the missing production
            share = his_avg / healthy[s].sum()
            mate_avg = averages[s].get(mate["PLAYER_ID"], 0)
            estimate = (his_avg + REDISTRIBUTE * mate_avg * share) / his_avg

            # ACTUAL: his average without the teammate vs with him
            actual = None
            if len(without_mate) and len(with_mate) and with_mate[s].mean() > 0:
                actual = without_mate[s].mean() / with_mate[s].mean()

            mixed = blend(estimate, actual, len(without_mate), TEAMMATE_TRUST)
            factors[s] = mixed ** (1 - already)   # skip what's already counted
            total[s] *= factors[s]

        details.append({"name": mate["PLAYER_NAME"], "games_without": len(without_mate),
                        "already": already, "factors": factors})

    low, high = TEAMMATE_LIMITS
    return {s: min(max(f, low), high) for s, f in total.items()}, details


def opponent_injury_impact(player, history, allowed, matchups,
                           current_season_id) -> tuple[dict, list]:
    """
    Step 4: how tonight's injured starters on the OPPONENT change what they
    allow. Returns ({stat: total factor}, [details per injured opponent]).
    """
    opp_id = player["OPP_ID"]
    total = {s: 1.0 for s in BASE_STATS}
    details = []
    opp_allowed = allowed[allowed["TEAM_ID"] == opp_id]

    for _, injured in starters_out(matchups, opp_id).iterrows():
        his_games = history[(history["PLAYER_ID"] == injured["PLAYER_ID"])
                            & (history["TEAM_ID"] == opp_id)]
        played_ids = set(his_games["GAME_ID"])

        together = opp_allowed[opp_allowed["SEASON_ID"].isin(set(his_games["SEASON_ID"]))]
        played = together["GAME_ID"].isin(played_ids)
        with_him, without_him = together[played], together[~played]

        # The opponent's season numbers already include games he missed
        this_season = opp_allowed[opp_allowed["SEASON_ID"] == current_season_id]
        already = (~this_season["GAME_ID"].isin(played_ids)).mean()

        factors = {s: 1.0 for s in BASE_STATS}
        for s in INJURY_STATS:
            actual = None
            if len(without_him) and len(with_him) and with_him[s].mean() > 0:
                actual = without_him[s].mean() / with_him[s].mean()
            mixed = blend(1.0, actual, len(without_him), OPP_TRUST)
            factors[s] = mixed ** (1 - already)
            total[s] *= factors[s]

        details.append({"name": injured["PLAYER_NAME"], "games_without": len(without_him),
                        "already": already, "factors": factors})

    low, high = OPP_LIMITS
    return {s: min(max(f, low), high) for s, f in total.items()}, details


# --------------------------------------------------------------------------
# The analysis for one player
# --------------------------------------------------------------------------
def estimated_line(average: float) -> float:
    """Round an average to a sportsbook-style .5 line.  26.1 -> 26.5, 4.6 -> 4.5"""
    return math.floor(average) + 0.5


def chance_over(projection: float, line: float, spread: float) -> float:
    """
    Rough chance of going OVER a line: assume his results follow a bell curve
    centred on our projection, as spread out as his last 10 games.
    """
    if pd.isna(spread):
        spread = 1.0
    spread = max(spread, 1.0)   # avoid silly 99% numbers on tiny samples
    z = (line - projection) / spread
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


def analyze_player(player: pd.Series, history: pd.DataFrame, allowed: pd.DataFrame,
                   opp_allowed: pd.DataFrame, matchups: pd.DataFrame,
                   current_season_id) -> dict:
    """Build every number the report needs for one player."""
    raw = history[history["PLAYER_ID"] == player["PLAYER_ID"]]
    clean = clean_games(raw, current_season_id)

    # Cleaned games feed the projection...
    season = clean[clean["SEASON_ID"] == current_season_id]
    raw_season = raw[raw["SEASON_ID"] == current_season_id]
    # Enough games this season? Then "last 10" stops reaching into last season
    this_season_only = len(season) >= MIN_SEASON_GAMES
    last10 = (season if this_season_only else clean).tail(10)
    h2h = clean[clean["OPP_TEAM_ID"] == player["OPP_ID"]]
    base_games = pd.concat([last10, season]).drop_duplicates("GAME_ID")
    # ...but hit rates use what really happened
    raw_last10 = (raw_season if this_season_only else raw).tail(10)
    raw_h2h = raw[raw["OPP_TEAM_ID"] == player["OPP_ID"]]

    opp = opponent_factors(opp_allowed, player["OPP_ID"])
    mates, mate_details = teammate_injury_impact(
        player, clean, base_games, history, matchups, current_season_id)
    opp_inj, opp_details = opponent_injury_impact(
        player, history, allowed, matchups, current_season_id)
    b2b = BACK_TO_BACK_FACTOR if player["BACK_TO_BACK"] else 1.0

    # Steps 1-5 for each of the six base stats
    steps = {}
    for s in BASE_STATS:
        parts = {"L10": last10[s].mean(), "SEASON": season[s].mean()}
        if len(h2h) >= MIN_H2H_GAMES:
            parts["H2H"] = h2h[s].mean()
        parts = {k: v for k, v in parts.items() if pd.notna(v)}
        base = sum(WEIGHTS[k] * v for k, v in parts.items()) / sum(WEIGHTS[k] for k in parts)
        after_opp = base * opp[s]
        after_inj = after_opp * mates[s] * opp_inj[s]
        steps[s] = (base, after_opp, after_inj, after_inj * b2b)

    rows = []
    for prop, cols in PROPS.items():
        # Combos are just the sum of their parts at each step
        base, after_opp, after_inj, proj = (sum(steps[c][i] for c in cols)
                                            for i in range(4))
        l10_avg = last10[prop].mean()
        edge = (proj - l10_avg) / l10_avg if l10_avg > 0 else 0.0

        # Estimated book line from his REAL averages, then which side we like
        book_avg = (raw_last10[prop].mean() + raw_season[prop].mean()) / 2
        line = estimated_line(book_avg)
        side = "OVER" if proj > line else "UNDER"
        p_over = chance_over(proj, line, last10[prop].std())
        model_chance = p_over if side == "OVER" else 1 - p_over
        hits = (raw_last10[prop] > line) if side == "OVER" else (raw_last10[prop] < line)

        rows.append({
            "PROP": prop,
            "L10": l10_avg,
            "SEASON": season[prop].mean(),
            "H2H": h2h[prop].mean() if len(h2h) else float("nan"),
            "BASE": base,
            "OPP_FACTOR": after_opp / base if base else 1.0,
            "INJ_FACTOR": after_inj / after_opp if after_opp else 1.0,
            "PROJ": proj,
            "EDGE_%": edge * 100,
            "LINE": line,
            "SIDE": side,
            "MODEL_CHANCE": model_chance,
            "L10_HIT": hits.mean(),
            "CONFIDENCE": 0.5 * model_chance + 0.5 * hits.mean(),
        })

    return {
        "table": pd.DataFrame(rows),
        "last10": last10,
        "raw_last10": raw_last10, "raw_season": raw_season, "raw_h2h": raw_h2h,
        "dropped": len(raw) - len(clean),
        "mate_details": mate_details, "mates": mates,
        "opp_details": opp_details, "opp_inj": opp_inj,
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------
def hit_rate(games: pd.DataFrame, prop: str, line: float, over: bool = True) -> str:
    """'7/10' style count of games that went over (or under) a line."""
    if games.empty:
        return "-"
    hits = (games[prop] > line) if over else (games[prop] < line)
    return f"{hits.sum()}/{len(games)}"


def pct(factor: float) -> str:
    """1.06 -> '+6%'"""
    return f"{(factor - 1) * 100:+.0f}%"


def team_h2h_summary(history: pd.DataFrame, team_id: int, opp_id: int) -> str:
    """Team-level record and average score vs this opponent (last 2 seasons)."""
    team_games = (history.groupby(["GAME_ID", "TEAM_ID", "OPP_TEAM_ID", "WL"])
                  ["PTS"].sum().reset_index())
    ours = team_games[(team_games["TEAM_ID"] == team_id)
                      & (team_games["OPP_TEAM_ID"] == opp_id)]
    if ours.empty:
        return "no meetings in the last 2 seasons"
    theirs = team_games[(team_games["TEAM_ID"] == opp_id)
                        & team_games["GAME_ID"].isin(ours["GAME_ID"])]
    wins = (ours["WL"] == "W").sum()
    return (f"{wins}-{len(ours) - wins} in {len(ours)} meetings, "
            f"avg score {ours['PTS'].mean():.1f} - {theirs['PTS'].mean():.1f}")


def print_injury_section(player: pd.Series, result: dict) -> None:
    """Explain how each injured starter changes his PTS / REB / AST."""
    mate_details, opp_details = result["mate_details"], result["opp_details"]
    if not mate_details and not opp_details:
        print("\nInjuries: no starters out on either team")
        return

    print("\nInjury impact on him            PTS   REB   AST")
    for d in mate_details:
        f = d["factors"]
        print(f"  {player['TEAM']} without {d['name']:<17} "
              f"{pct(f['PTS']):>5} {pct(f['REB']):>5} {pct(f['AST']):>5}"
              f"   ({d['games_without']} games without him together, "
              f"{d['already']:.0%} already in his averages)")
    for d in opp_details:
        f = d["factors"]
        print(f"  {player['OPPONENT']} without {d['name']:<17} "
              f"{pct(f['PTS']):>5} {pct(f['REB']):>5} {pct(f['AST']):>5}"
              f"   ({d['games_without']} {player['OPPONENT']} games without him, "
              f"{d['already']:.0%} already in their numbers)")
    total = {s: result["mates"][s] * result["opp_inj"][s] for s in BASE_STATS}
    print(f"  {'Total injury effect':<29}"
          f"{pct(total['PTS']):>5} {pct(total['REB']):>5} {pct(total['AST']):>5}")


def print_report(player: pd.Series, result: dict, history: pd.DataFrame) -> None:
    """Print the full prop report for one player."""
    t = result["table"]
    print()
    print("=" * 78)
    print(f"{player['PLAYER_NAME']} ({player['TEAM']}) - {player['HOME_AWAY']} vs "
          f"{player['OPPONENT']}, {player['STATUS']}")
    print("=" * 78)

    # ---- Context ----
    b2b = "YES (projection x0.97)" if player["BACK_TO_BACK"] else "no"
    print(f"Status: {player['INJURY_STATUS']}"
          + (f" - {player['INJURY_NOTE']}" if player["INJURY_NOTE"] else ""))
    print(f"Rest: {player['REST_DAYS']} day(s), back-to-back: {b2b}")
    print(f"{player['OPPONENT']} defense: rating {player['OPP_DEF_RATING']} "
          f"(rank {player['OPP_DEF_RANK']}/30), pace {player['OPP_PACE']} "
          f"(rank {player['OPP_PACE_RANK']}/30)")
    print(f"{player['TEAM']} vs {player['OPPONENT']}: "
          f"{team_h2h_summary(history, player['TEAM_ID'], player['OPP_ID'])}")
    if result["dropped"]:
        print(f"Ignored {result['dropped']} short-minute game(s) "
              "(early exits / blowouts)")

    print_injury_section(player, result)

    raw_h2h = result["raw_h2h"]
    if not raw_h2h.empty:
        print(f"\nHis games vs {player['OPPONENT']} (last 2 seasons):")
        cols = ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "FG3M"]
        print(raw_h2h[cols].assign(GAME_DATE=raw_h2h["GAME_DATE"].dt.date)
              .to_string(index=False))

    # ---- Projection table ----
    print("\nProjections  (averages are after cleaning outliers; "
          "x > 1 helps overs)")
    show = t[["PROP", "L10", "SEASON", "H2H", "BASE", "OPP_FACTOR",
              "INJ_FACTOR", "PROJ", "EDGE_%"]].rename(
        columns={"OPP_FACTOR": "OPP x", "INJ_FACTOR": "INJ x"})
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"
                         if abs(x) < 3 else f"{x:.1f}", na_rep="-"))

    # ---- Best legs ----
    candidates = (t[t["L10"] >= MIN_AVG_FOR_LEG]
                  .sort_values("CONFIDENCE", ascending=False))
    print("\nBest legs tonight (vs an ESTIMATED line - check your book's real one):")
    for i, (_, r) in enumerate(candidates.head(3).iterrows(), start=1):
        is_over = r["SIDE"] == "OVER"
        print(f"  {i}. {r['PROP']} {r['SIDE']} {r['LINE']}  "
              f"confidence {r['CONFIDENCE']:.0%}  (proj {r['PROJ']:.1f}, "
              f"model {r['MODEL_CHANCE']:.0%})  hit: L10 "
              f"{hit_rate(result['raw_last10'], r['PROP'], r['LINE'], is_over)}, "
              f"vs {player['OPPONENT']} "
              f"{hit_rate(raw_h2h, r['PROP'], r['LINE'], is_over)}")
    if candidates.head(3)["CONFIDENCE"].max() < 0.6:
        print("  (Nothing above 60% confidence - no strong plays for him tonight.)")

    if player["MIN"] < 15:
        print("\n  ! He averages under 15 minutes - props are very unpredictable.")
    if player["INJURY_STATUS"] == "Day-To-Day":
        print("\n  ! Listed Day-To-Day - check he's playing before betting.")
    print("\nCheck a sportsbook line: type e.g.  pts 27.5   (props: "
          + ", ".join(p.lower() for p in PROPS) + ")")


def print_line_check(player: pd.Series, result: dict, prop: str, line: float) -> None:
    """Compare one sportsbook line with our projection and his history."""
    r = result["table"].set_index("PROP").loc[prop]
    p_over = chance_over(r["PROJ"], line, result["last10"][prop].std())

    lean = "OVER" if r["PROJ"] > line else "UNDER"
    print(f"\n{player['PLAYER_NAME']} {prop} {line}:  projection {r['PROJ']:.1f} "
          f"({r['PROJ'] - line:+.1f})  -> lean {lean}")
    print(f"  Estimated chance of OVER: {p_over:.0%}   UNDER: {1 - p_over:.0%}")
    print(f"  Went over: L10 {hit_rate(result['raw_last10'], prop, line)}, "
          f"this season {hit_rate(result['raw_season'], prop, line)}, "
          f"vs {player['OPPONENT']} {hit_rate(result['raw_h2h'], prop, line)}")
    if abs(r["PROJ"] - line) < 0.1 * max(line, 1):
        print("  (Within 10% of the line - close call, not a strong play.)")


# --------------------------------------------------------------------------
# The interactive loop
# --------------------------------------------------------------------------
def find_player(query: str, matchups: pd.DataFrame):
    """
    Find a player playing tonight from what you typed. Returns
    (row or None, message). Accepts partial names like "maxey" or "shai".
    """
    key = normalize_name(query)
    keys = matchups["PLAYER_NAME"].map(normalize_name)
    if not key:
        return None, "Type a player's name."

    exact = matchups[keys == key]
    if len(exact) == 1:
        return exact.iloc[0], ""

    # Every word you typed appears in the name (so "maxey" finds Tyrese Maxey)
    words = key.split()
    partial = matchups[keys.map(lambda k: all(w in k for w in words))]
    if len(partial) == 1:
        return partial.iloc[0], ""
    if len(partial) > 1:
        names = ", ".join(
            n + (" (OUT)" if s == "Out" else "")
            for n, s in zip(partial["PLAYER_NAME"].head(10),
                            partial["INJURY_STATUS"].head(10)))
        return None, f"More than one match - be more specific: {names}"

    # Nothing matched: look for close spellings of any single name word
    # ("maxy" is close to "maxey", so it finds Tyrese Maxey)
    all_words = sorted({w for k in keys for w in k.split()})
    close_words = set()
    for w in words:
        close_words.update(difflib.get_close_matches(w, all_words, n=3, cutoff=0.75))
    suggestions = matchups[keys.map(lambda k: bool(close_words & set(k.split())))]
    if len(suggestions) == 1:
        return suggestions.iloc[0], ""
    hint = (" Did you mean: " + ", ".join(suggestions["PLAYER_NAME"].head(5)) + "?"
            if len(suggestions) else "")
    return None, f"No player named '{query}' is playing tonight.{hint}"


def run_prop_finder(matchups: pd.DataFrame, history: pd.DataFrame,
                    opp_allowed: pd.DataFrame, current_season_id) -> None:
    """Ask for players / lines until you type q."""
    allowed = build_allowed_by_game(history)   # used for opponent injuries

    print("\n" + "#" * 78)
    print("PROP FINDER - type a player playing tonight (e.g. maxey), "
          "'list' for everyone, or 'q' to quit")
    print("#" * 78)

    line_pattern = re.compile(r"^([a-z0-9]+)\s+(\d+(?:\.\d+)?)$")
    player, result = None, None

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):   # Ctrl+C / end of input
            print()
            break

        if text.lower() in {"q", "quit", "exit", ""}:
            break

        if text.lower() == "list":
            playing = matchups[matchups["INJURY_STATUS"] != "Out"]
            for game, block in playing.groupby("GAME", sort=False):
                print(f"{game}: " + ", ".join(block["PLAYER_NAME"]))
            continue

        # "pts 27.5" -> check a line for the player we're looking at
        match = line_pattern.match(text.lower())
        if match and match.group(1).upper() in PROPS:
            if player is None:
                print("Pick a player first, then type the line.")
            else:
                print_line_check(player, result, match.group(1).upper(),
                                 float(match.group(2)))
            continue

        # Anything else is a player name
        found, message = find_player(text, matchups)
        if found is None:
            print(message)
            continue
        if found["INJURY_STATUS"] == "Out":
            print(f"{found['PLAYER_NAME']} is OUT tonight ({found['INJURY']}) "
                  "- no props to recommend.")
            continue

        player = found
        result = analyze_player(player, history, allowed, opp_allowed,
                                matchups, current_season_id)
        print_report(player, result, history)
