from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def parse_game_date(t):
    if not pd.isnull(t):
        return datetime.strptime(t[:-4], '%Y-%m-%dT%M:%S')
    else:
        return np.nan


def add_features2(df):
    df = df.copy()

    df['gameDateTime'] = df['gameDateTime'].map(parse_game_date)
    df['gameEndDateTime'] = df['gameEndDateTime'].map(parse_game_date)

    df['game_seconds'] = df.apply(
        lambda t:
            (t['gameEndDateTime'] - t['gameDateTime']).total_seconds()
            if not pd.isnull(t['gameEndDateTime'])
            else np.nan,
        axis=1)
    # BAD DATA: DROP GAMES WITH MORE THAN 200 SECONDS

    df = df.sort_values(['gamePk', 'periodTime'], ascending=[True, True])
    df['last_event_type_id'] = df.groupby('gamePk')['eventTypeId'].shift(1)
    df['last_event_x'] = df.groupby('gamePk')['x'].shift(1)
    df['last_event_y'] = df.groupby('gamePk')['y'].shift(1)
    df['last_event_time'] = df.groupby('gamePk')['periodTime'].shift(1)
    df['last_event_angle'] = df.groupby('gamePk')['angle_from_net'].shift(1)

    norm_period_time = lambda t: datetime(2000, 1, 1, 0, int(t.split(':')[0]), int(t.split(':')[1]))

    def period_time_distance(pt1, pt2):
        if (not isinstance(pt2, str)) or (not isinstance(pt1, str)):
            return np.nan

        dt1 = norm_period_time(pt1)
        dt2 = norm_period_time(pt2)

        return (dt2 - dt1).total_seconds()

    df['time_from_last_event'] = df.apply(
        lambda t: period_time_distance(t['last_event_time'], t['periodTime']),
        axis=1
    )
    df['dist_from_last_event'] = df.apply(
        lambda t: np.linalg.norm(np.array([t['x'], t['y']]) - np.array([t['last_event_x'], t['last_event_y']])),
        axis=1
    )

    df['rebound'] = df.apply(lambda t: t['eventTypeId'] == t['last_event_type_id'], axis=1)
    df['change_in_angle'] = df.apply(lambda t: abs(t['angle_from_net'] - t['last_event_angle']), axis=1)
    df['speed'] = df.apply(lambda t: t['dist_from_last_event'] / (t['time_from_last_event'] + 1), axis=1)

    ####################################
    # BONUS FEATURES ##################

    penalties = df[df.eventTypeId == 'PENALTY'][['gamePk', 'period', 'periodTime', 'team_id', 'penaltyMinutes']]
    penalties['periodTimeEnd'] = penalties.apply(
        lambda t: (norm_period_time(t['periodTime']) + timedelta(minutes=t['penaltyMinutes'])).strftime('%M:%S'),
        axis=1
    )

    game_teams = df.groupby('gamePk')['team_id'].agg(
        lambda t: sorted(map(int, filter(lambda k: not np.isnan(k), np.unique(t.values))))
    ).reset_index()

    opposing_team_dct = dict()
    game_teams.apply(lambda t: opposing_team_dct.update({(t['gamePk'], t['team_id'][0]): t['team_id'][1]}), axis=1)
    game_teams.apply(lambda t: opposing_team_dct.update({(t['gamePk'], t['team_id'][1]): t['team_id'][0]}), axis=1)

    penalties_dct = defaultdict(list)
    penalties[['gamePk', 'period', 'team_id', 'periodTime', 'periodTimeEnd']].apply(
        lambda t: penalties_dct[(t['gamePk'], t['period'], t['team_id'])].append({
            'periodTime': t['periodTime'],
            'periodTimeEnd': t['periodTimeEnd']}
        ), axis=1)

    def get_n_players(gamepk, period, team, time):
        n = 5
        if (gamepk, period, team) in penalties_dct:
            c_penal = penalties_dct[(gamepk, period, team)]
            for times in c_penal:
                if times['periodTime'] <= time < times['periodTimeEnd']:
                    n -= 1
        return n

    df['n_players'] = df.apply(
        lambda t: get_n_players(t['gamePk'], t['period'], t['team_id'], t['periodTime']),
        axis=1
    )
    df['n_opposing_players'] = df.apply(
        lambda t: get_n_players(
            t['gamePk'],
            t['period'],
            opposing_team_dct.get((t['gamePk'], t['team_id']), np.nan),
            t['periodTime']),
        axis=1)

    def powerplay_start(gamepk, period, team, time):
        if np.isnan(team):
            return np.nan  # Doesn't matter as we intend to keep only shots

        _powerplay_start = '99:99'
        for times in penalties_dct.get((gamepk, period, team), []):
            if times['periodTime'] < time < times['periodTimeEnd']:
                _powerplay_start = min(_powerplay_start, times['periodTime'])

        for times in penalties_dct.get((gamepk, period, opposing_team_dct[(gamepk, team)]), []):
            if times['periodTime'] < time < times['periodTimeEnd']:
                _powerplay_start = min(_powerplay_start, times['periodTime'])

        if _powerplay_start == '99:99':
            return np.nan

        return _powerplay_start

    def time_since_powerplay(gamepk, period, team, time):
        _start = powerplay_start(gamepk, period, team, time)
        c_start = _start
        while not pd.isnull(_start):  # Iterate since start of powerplay time windows, until penalty wasn't in powerplay
            c_start = _start
            _start = powerplay_start(gamepk, period, team, _start)

        if pd.isnull(c_start):
            return np.nan

        return period_time_distance(c_start, time)

    df['time_since_powerplay'] = df.apply(
        lambda t: time_since_powerplay(t['gamePk'], t['period'], t['team_id'], t['periodTime']),
        axis=1
    )

    #df[['gamePk', 'eventTypeId', 'periodTime', 'x', 'y', 'last_event_type_id', 'last_event_x', 'last_event_y',
    #    'last_event_time', 'time_from_last_event', 'dist_from_last_event', 'rebound', 'angle_from_net', 'last_event_angle', 'change_in_angle', 'speed']].head(60)

    return df


def infer_side(tidied_training_set, tidied_test_set):

    a = pd.concat((tidied_training_set, tidied_test_set))
    sides = a.groupby(['gamePk', 'period', 'team_name']).agg(
        {'x': 'median'})  # , 'rink_side': lambda t: [k for k in t if not pd.isnull(k)][0]})
    sides = sides.reset_index()

    sides['norm_x'] = sides.apply(lambda t: t['x'] * ((t['period'] % 2) * 2 - 1), axis=1)

    norm_sides = sides.groupby(['gamePk', 'team_name'])['norm_x'].median().reset_index()

    # norm_sides_max = norm_sides.groupby(['gamePk']).norm_x.max().reset_index().rename(columns={'norm_x': 'max_x'})
    norm_sides_min = norm_sides.groupby(['gamePk']).norm_x.min().reset_index().rename(columns={'norm_x': 'min_x'})

    norm_sides = norm_sides.merge(norm_sides_min)

    norm_sides['period_1_side'] = norm_sides.apply(lambda t: ('right' if t['norm_x'] == t['min_x'] else 'left'), axis=1)

    #norm_sides[['gamePk', 'team_name', 'norm_x', 'period_1_side']].to_csv('resources/period_1_sides.csv', index=False)

















