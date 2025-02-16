import requests
import pandas as pd
import os
import json
import numpy as np

"""##2. Feature Engineering I (10%)"""

'''
Using the functionality you created for Milestone 1, acquire all of the raw play-by-play data for the 2015/16 season
all the way to the 2019/20 season (inclusive). Note that the tidied data that you created will be useful for the baseline
models, but you will be creating more features that will require the full raw data in Part 4.

Set aside all of the 2019/20 data as your final test set. You should not touch this until you reach the end of this milestone.
You will use the 2015/16 - 2018/19 regular season data to create your training and validation sets. By the end of this milestone,
you will have built up all the framework you need to process new data, so it should be trivial to process your test data once
you have finished all of the necessary feature engineering. Until Part 7, any reference to the “dataset” will exclusively refer
to the 2015/16 - 2018/19 data.
'''

#directory = "/content/drive/MyDrive/DS-Project/milestone2_data" #change this to your directory
directory = "data"  # change this to your directory

"""####Functions from Milestone 1 for getting data"""

'''
Functions from Milestone 1 for getting data
'''

base_url = "https://statsapi.web.nhl.com/api/v1/"


def get_game_ids_for_season(season):

    game_ids = []

    # Fetch the schedule for the given season
    response = requests.get(f"{base_url}schedule?season={season}")
    if response.status_code != 200:
        print(f"Failed to fetch the schedule for season {season}.")
        return []

    data = response.json()
    dates = data.get("dates", [])

    # Extract game IDs
    for date in dates:
        games = date.get("games", [])
        for game in games:
            game_id = game.get("gamePk")
            game_ids.append(game_id)

    return game_ids


# Fetch and save game data for each game ID
def save_game_data_to_local(game_ids, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    for game_id in game_ids:

        fname = os.path.join(directory, str(game_id) + '.json')
        if os.path.exists(fname):
            continue

        response = requests.get(f"{base_url}game/{game_id}/feed/live")

        if response.status_code == 200:
            with open(fname, "w") as file:
                json.dump(response.json(), file)
        else:
            print(f"Failed to fetch data for game ID {game_id}")


def load_data_from_files(directory):

    all_data = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                data = json.load(file)
                all_data.append(data)

    return all_data


def flatten_dict(d, prefix=''):

    def _flatten_dict(d, r, path):
        _prefix = path + '_' if path != '' else ''

        for k in d.keys():
            if isinstance(d[k], dict):
                _flatten_dict(d[k], r, _prefix + k)
            else:
                r[_prefix + k] = d[k]

    r = dict()

    _flatten_dict(d, r, prefix)

    return r


def get_play_data(ld, keep_all_events=False):

    keep_event_types = {'SHOT', 'GOAL'}

    def format_players(pls):
        return {
                pls[i]['playerType'] + '_' + k: pls[i]['player'][k]
                for i in range(len(pls))
                for k in pls[i]['player'].keys()
        }

    meta = {
        'gamePk': ld['gamePk'],
        'gameDateTime': ld['gameData']['datetime'].get('dateTime'),
        'gameEndDateTime': ld['gameData']['datetime'].get('endDateTime')
    }

    for play in ld['liveData']['plays']['allPlays']:
        playdata = dict()

        if (not keep_all_events) and play['result']['eventTypeId'] not in keep_event_types:
            continue

        players = format_players(play['players']) if 'players' in play else dict()
        result = flatten_dict(play['result']) if 'result' in play else dict()
        about = flatten_dict(play['about']) if 'about' in play else dict()

        coordinates = flatten_dict(play['coordinates']) if 'coordinates' in play else dict()
        team = flatten_dict(play['team'], prefix='team') if 'team' in play else dict()
        #team = {'team_' + k: play['team'][k] for k in play['team'].keys()}

        playdata.update(players)
        playdata.update(result)
        playdata.update(about)
        playdata.update(coordinates)
        playdata.update(team)

        playdata.update(meta)

        yield playdata


def create_game_info_list(season):
    '''
    Use linescore to get home and away rinkSide.
    Creates a list of dictionaries:
    [{
    'Game PK': 2017010001,
    'Away Team Name': 'Vancouver Canucks',
    'Home Team Name': 'Los Angeles Kings',
    'Periods Info':
    [{
      'Period': 1,
      'Home Rink Side': 'right',
      'Away Rink Side': 'left'},
      {
      'Period': 2,
      'Home Rink Side': 'left',
      'Away Rink Side': 'right'},
      {
      'Period': 3,
      'Home Rink Side': 'right',
      'Away Rink Side': 'left'},
      etc
    '''
    url = f"https://statsapi.web.nhl.com/api/v1/schedule?season={season}&expand=schedule.linescore"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch the schedule for season {season}.")

    data = response.json()

    # Parse the JSON data
    parsed_data = json.loads(json.dumps(data))

    # Create a list to store the extracted information
    game_info_list = []

    # Extract information for each game
    for date_info in parsed_data['dates']:
        for game_info in date_info['games']:
            gamepk = game_info['gamePk']
            away_team_name = game_info['teams']['away']['team']['name']
            home_team_name = game_info['teams']['home']['team']['name']

            # Extract information for each period
            periods_info = []
            for period in game_info['linescore']['periods']:
                period_num = period['num']

                # Check if 'rinkSide' key exists for home and away
                home_rink_side = period['home'].get('rinkSide', 'N/A')
                away_rink_side = period['away'].get('rinkSide', 'N/A')

                period_info = {
                  'Period': period_num,
                  'Home Rink Side': home_rink_side,
                  'Away Rink Side': away_rink_side,
                }

                periods_info.append(period_info)

            # Create a dictionary to represent the game information
            game_info_dict = {
              'Game PK': gamepk,
              'Away Team Name': away_team_name,
              'Home Team Name': home_team_name,
              'Periods Info': periods_info,
            }

            # Append the game information to the list
            game_info_list.append(game_info_dict)

    return game_info_list


def add_home_away_rink_side_columns(df):
    '''

    '''
    def other_side(s):
        if s is None:
            return None

        if s == 'left':
            return 'right'
        return 'left'

    period_sides = pd.read_csv('resources/period_1_sides.csv')
    dct_sides = dict(zip(period_sides[['gamePk', 'team_name']].apply(tuple, axis=1).tolist(), period_sides['period_1_side'].tolist()))

    # Infer side from where the shots were made
    df['period_1_side'] = df.apply(lambda t: dct_sides.get((t['gamePk'], t['team_name'])), axis=1)
    df['rink_side'] = df.apply(lambda t: t['period_1_side'] if t['period'] % 2 == 1 else other_side(t['period_1_side']), axis=1)
    #df[['gamePk', 'period', 'team_name', 'period_1_side', 'rink_side']].head(60)

    del df['period_1_side']

    return df


def add_home_away_rink_side_columns_api(df, game_info_list):
    '''
    Use game_info_list to add the columns 'home_or_away' and 'rink_side' to the df
    '''

    # Create new columns 'home_or_away' and 'rink_side'
    df['home_or_away'] = ''
    df['rink_side'] = ''

    # Iterate through DataFrame rows
    for index, row in df.iterrows():
        game_pk = row['gamePk']
        team_name = row['team_name']
        period = row['period']

        # Find the corresponding game_info_dict based on gamePk
        for game_info_dict in game_info_list:
            if game_info_dict['Game PK'] == game_pk:

                away_team_name = game_info_dict['Away Team Name']
                home_team_name = game_info_dict['Home Team Name']

                # Determine 'home_or_away' based on team_name
                if team_name == away_team_name:
                    df.at[index, 'home_or_away'] = 'Away'
                elif team_name == home_team_name:
                    df.at[index, 'home_or_away'] = 'Home'

                # Find the corresponding 'rink_side' based on period and team's location (home or away)
                for period_info in game_info_dict['Periods Info']:
                    if period_info['Period'] == period:
                        if df.at[index, 'home_or_away'] == 'Home':
                            df.at[index, 'rink_side'] = period_info.get('Home Rink Side', 'N/A')
                        elif df.at[index, 'home_or_away'] == 'Away':
                            df.at[index, 'rink_side'] = period_info.get('Away Rink Side', 'N/A')





