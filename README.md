# NBA-prediction
Fun little application that helps with nba players stats and also gives props/legs to casual gamblers

WARNING: THIS TOOL IS NOT 100% ACCURATE THERE ARE TIMES WHERE PLAYERS CAN PERFORM BETTER/WORSE THAN EXPECTED. PLEASE ENSURE THE RISKS YOU ARE TAKING WHEN PROSCEEDING!

Update: Currently has a working output and can review props/legs for players of your own choosing. Can also check the status of players to determine if they are day to day or injured. 

Next Steps: Will need to update specific stats based off of changing environments such as injured teammates/opposing players. Will also create a working UI since the application is currently being used within a terminal which is hard to operate when dealing with multiple players.

Current Problems: Too few games are being played by everyone, will need everyone to play a minimum of 10 games within the season so that there is more accurate data for each player and their specific bets



## First-time setup

No API keys or accounts needed - all data comes from NBA.com and ESPN's free public feeds.

1. Install a recent version of Python from https://www.python.org/downloads/ (this was built on Python 3.15).
   On Windows, tick **"Add python.exe to PATH"** in the installer.
2. Open a terminal **inside the `NBA-prediction` folder** (the one with `add_matchups.py` in it).
3. Create your own venv and install the packages (only needed once):

   Windows (PowerShell):
   ```
   python -m venv venv
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   Mac / Linux:
   ```
   python3 -m venv venv
   ./venv/bin/python -m pip install -r requirements.txt
   ```

Don't copy someone else's `venv` folder - it only works on the computer that made it.

## How to run

1. From the `NBA-prediction` folder, run:
   - Windows: `.\venv\Scripts\python.exe add_matchups.py`
   - Mac / Linux: `./venv/bin/python add_matchups.py`
2. It loads today's games (first run takes a minute while it downloads game logs).
3. Type in the player that you want to bet on (no nicknames, e.g. `maxey`) and review their best props.
4. Type a line like `pts 27.5` to check a sportsbook line, `list` to see everyone playing, or `q` to quit.
