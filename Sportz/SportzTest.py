import nba_api
import requests
import csv
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import leaguegamelog
season = leaguegamelog.LeagueGameLog("00")
seasonGames = season.get_data_frames()[0]
seasonGames.head()
print(seasonGames.TEAM_NAME)
opDF = seasonGames.sort_values(by=["TEAM_NAME", "GAME_ID"])[["TEAM_NAME", "TEAM_ID", "GAME_ID","MATCHUP","WL","FGM","FGA","FG_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PTS","PLUS_MINUS"]]
opDF.columns = [["TEAM_NAME", "TEAM_ID", "GAME_ID","MATCHUP", "WL","FGM","FGA","FG_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PTS","PLUS_MINUS"]]
opDF.to_csv('NBAStats.csv', index=False)
print("done")
