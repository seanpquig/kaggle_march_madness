kaggle_march_madness
====================


####*Purpose:*
This competition aims to estimate winning probablities for every possible matchup of NCAA tournament teams.  Stage one estimates probabilites from the past five tournaments.  Stage two will require estimates be submitted for the 2014 tournament before it starts.  The winning solution will be determined by the lowest LogLoss measure.


####*Data:*
* **teams.csv:** lists 356 teams and their id's.
* **seasons.csv:** lists 19 seasons in dataset (includes current season).  Dayzero variable is the date 154 days before the Championship game.  Lists 4 tournament regions, ordered by semifinal pairings (W vs. X and Y vs. Z)
* **regular_season_results.csv:**  lists regular season results for past 18 seasons:  teams, scores, away vs home, and number of OT periods.
* **tourney_results.csv:**  lists NCAA tournament results for past 18 seasons.  Neutral location always assumed.
* **tourney_seeds.csv:**  lists 64 seeds by region for past 18 seasons.
* **tourney_slots.csv:**  lists all tournament games in an encoded format.  Has same number of records as tourney_results.csv.  Can be used to construct a tree for the bracket.
* **sample_submission.csv:**  submission format for stage one.  Includes every combination of matchups between tourney teams for the past 5 seasons (N-R).
