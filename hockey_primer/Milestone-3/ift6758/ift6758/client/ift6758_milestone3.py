# -*- coding: utf-8 -*-

import requests
import pandas as pd
import os
import json
import numpy as np

"""##1. Update API client (10 %)

#### Updated functions for getting data, cleaning dataframe and adding features
"""

'''
Updated functions for getting data
'''

def get_team_abbreviations():
  team_data_url = "https://api.nhle.com/stats/rest/en/team"
  team_data_response = requests.get(team_data_url)
  teams = team_data_response.json()["data"]
  team_abbreviations = [team["triCode"] for team in teams]

  return team_abbreviations

def get_game_ids_for_season(season, team_abbreviations):

  base_schedule_url = "https://api-web.nhle.com/v1/club-schedule-season/{team_abbr}/{season}"

  all_game_ids = set()

  # Iterate through each team and get the season schedule
  for team_abbr in team_abbreviations:
    schedule_url = base_schedule_url.format(team_abbr=team_abbr, season=season)
    schedule_response = requests.get(schedule_url)
    schedule_data = schedule_response.json()

    # Get game IDs and add them to the set
    game_ids = {game["id"] for game in schedule_data.get("games", [])}
    all_game_ids.update(game_ids)

  game_ids = list(all_game_ids)

  return game_ids


def fetch_game_data(game_id):
    response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for game ID {game_id}")
        return None


def fetch_and_concat_data(season, team_abbreviations):
  game_ids = get_game_ids_for_season(season, team_abbreviations)

  all_data = []
  for game_id in game_ids:
      game_data = fetch_game_data(game_id)
      if game_data:
          all_data.append(game_data)

  # Create a Pandas DataFrame from the list of data
  data_df = pd.DataFrame(all_data)
  return data_df


'''
Helper functions
'''

def get_play_data(ld):

  keep_event_types = {'shot-on-goal', 'goal', 'missed-shot'} #'blocked-shot', which event types? (what are hits?)

  meta = {
      'gamePk': ld['id']
      #'gameDateTime': ld['startTimeUTC'] #ld['gameData']['datetime'].get('dateTime'),
      #'gameEndDateTime': ld['gameData']['datetime'].get('endDateTime')
      }
  period = {'period': ld['period']}

  teams = {
      'awayTeamId': ld['awayTeam']['id'],
      'awayTeamName': ld['awayTeam']['name']['default'],
      'awayTeamAbbrev': ld['awayTeam']['abbrev'],
      'homeTeamId': ld['homeTeam']['id'],
      'homeTeamName': ld['homeTeam']['name']['default'],
      'homeTeamAbbrev': ld['homeTeam']['abbrev']
      }

  for play in ld['plays']: #ld['liveData']['plays']['allPlays']:
    #sys.exit()

    playdata = dict()

    #if not play['result']['eventTypeId'] in keep_event_types:
    if not play['typeDescKey'] in keep_event_types:
      continue

    playdata.update(meta)
    playdata.update(period)
    playdata.update(teams)
    playdata['typeDescKey'] =  play.get('typeDescKey')
    playdata['xCoord'] = play['details'].get('xCoord')
    playdata['yCoord'] = play['details'].get('yCoord')
    playdata['zoneCode'] = play['details'].get('zoneCode') #i think this means offense or defense!
    playdata['situationCode'] = play.get('situationCode')
    playdata['eventOwnerTeamId'] = play['details'].get('eventOwnerTeamId') #find which team number corresponds to'''

    yield playdata


'''
Helper functions for features
'''

# Empty net (1 for empty)
def is_empty_net(row):
    situation_code = row['situationCode']
    if pd.isna(situation_code) or situation_code == "":
        return 0
    shooter = row['eventOwnerTeamId']
    if shooter == row['awayTeamId']:
      if int(situation_code[3]) == 1: # Fourth digit: 1 for goalie on ice for home team
        return 0
      else:
        return 1
    else:
      if int(situation_code[0]) == 1: # First digit: 1 for goalie on ice for away team
        return 0
      else:
        return 1


# if coordinates = left and zoneCode = o then trying to score in left net RIGHT
# if coordinates = right and zoneCode = o then trying to score in right net LEFT

# if coordinates = left and zoneCode = d then trying to score in right net LEFT
# if coordinates = right and zoneCode = d then trying to score in left net RIGHT

# for zoneCode = n
#add period to data
#find an event with the same gameid, period and team : copy the rink side

def decide_rink_side(row):
  if row['xCoord'] < 0 and row['zoneCode'] == 'O':
    return 'right'
  elif row['xCoord'] > 0 and row['zoneCode'] == 'O':
    return 'left'
  elif row['xCoord'] < 0 and row['zoneCode'] == 'D':
    return 'left'
  elif row['xCoord'] > 0 and row['zoneCode'] == 'D':
    return 'right'


'''
Add features:
 -Distance from net
 -Angle from net
 -Is goal (0 or 1)
 -Empty Net
'''
def add_features(tidied_training_set):

  # Add column Is goal (0 or 1)
  print(tidied_training_set)
  tidied_training_set['IsGoal'] = (tidied_training_set['typeDescKey'] == 'goal').astype(int)

  tidied_training_set['emptyNet'] = tidied_training_set.apply(is_empty_net, axis=1)

  tidied_training_set['rinkSide'] = tidied_training_set.apply(decide_rink_side, axis=1)

  def decide_rink_side_for_neutral_zone(row):
    if row['zoneCode'] == 'N':
      game_id = row['gamePk']
      period = row['period']
      team = row['homeTeamId']

      # Find the corresponding row with the same gameid, period, and team
      matching_row = tidied_training_set[
          (tidied_training_set['gamePk'] == game_id) &
          (tidied_training_set['period'] == period) &
          (tidied_training_set['homeTeamId'] == team)
      ]

      if not matching_row.empty:
          # Copy the rinkSide value from the matching row
          return matching_row['rinkSide'].values[0]

    # Return original rinkSide if not in 'N' zoneCode or no matching row found
    return row['rinkSide']

  tidied_training_set['rinkSide'] = tidied_training_set.apply(decide_rink_side_for_neutral_zone, axis=1)

  '''
  if coordinates are for left side
  zoneCode: is this only using coordinates or can i use this as rink side?
  '''
  # Add column Distance from net

  # approximate nets as being (-89, 0) and (89, 0)
  # if rink side is left offense is on the right
  # if rink side is left: trying to score in right net (89, 0)
  # if rink side is right: trying to score in left net (-89, 0)

  tidied_training_set['distanceFromNet'] = np.sqrt(
      np.where(
          tidied_training_set['rinkSide'] == 'left',
          (tidied_training_set['xCoord'] - 89) ** 2 + tidied_training_set['yCoord'] ** 2,
            (tidied_training_set['xCoord'] + 89) ** 2 + tidied_training_set['yCoord'] ** 2))

  # Add column Angle from net

  # If shoots from right in from of net (y=0), then angle is 0 deg
  # if shoots completely from the side (x=89 or -89), then angle is 90 deg

  #add a temp column x_distance_from_net
  tidied_training_set['xDistanceFromNet'] = np.where(
      tidied_training_set['rinkSide'] == 'left',
      abs(89 - tidied_training_set['xCoord'] ),
        abs(-89 - tidied_training_set['xCoord'] ))

  tidied_training_set['angleFromNet'] = np.degrees(np.arccos(tidied_training_set['xDistanceFromNet'] / tidied_training_set['distanceFromNet']))

  #can remove the x_distance_from_net column
  tidied_training_set.drop(['xDistanceFromNet'], axis=1, inplace=True)

  return tidied_training_set


def get_cleaned_data_with_features(season):
  data_df = fetch_and_concat_data(season, team_abbreviations)
  tidied_training_set = pd.DataFrame(t for i, l in data_df.iterrows() for t in get_play_data(l))
  final_dataset = add_features(tidied_training_set)
  return final_dataset

"""#### To run

*   Specify directory and seasons

"""

# directory = "/content/drive/MyDrive/DS-Project/Milestone3"  # Change this to your directory

# team_abbreviations = get_team_abbreviations()

# seasons = ["20162017"]#, "20172018", "20182019", "20192020", "20202021"]

# for season in seasons:
#     data = get_cleaned_data_with_features(season)

#     # save as csv
#     data.to_csv(os.path.join(directory, f"{season}.csv"), index=False)

# tidied_training_set = pd.read_csv("/content/drive/MyDrive/DS-Project/Milestone3/20172018.csv")
# tidied_training_set