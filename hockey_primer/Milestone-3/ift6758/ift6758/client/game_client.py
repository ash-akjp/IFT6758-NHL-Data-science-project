import json
import requests
import pandas as pd
import logging
import os
import numpy as np
import math
import time

# Assuming ift6758_milestone3 contains the provided helper functions
from ift6758.client.ift6758_milestone3 import get_play_data, add_features, fetch_game_data

class GameClient:
    def __init__(self):
        self.tracker = 0
        self.game = None
        self.home_team = None
        self.away_team = None
        self.dashboard_time = float('inf')
        self.dashboard_period = 0
        
    def get_game(self, game_id):
        self.game_id = game_id
        file_path = f'./{self.game_id}.json'
        game_data = fetch_game_data(game_id)
        if game_data is None:
            return None
        with open(file_path, 'w') as f:
            json.dump(game_data, f)
        return file_path
    
    def update_model_df_length(self):
        self.model_df_length = self.game.shape[0]
   
    def ping_game(self, file_path):
        with open(file_path, 'r') as file:
            game_data = json.load(file)

        # Process the game data to get a DataFrame
        df_game_tidied = pd.DataFrame([x for x in get_play_data(game_data)])
        print(df_game_tidied.columns)
        df_game_features = add_features(df_game_tidied)
        last_event = df_game_features.iloc[-1]
        self.game = df_game_features
        self.update_model_df_length()
        tracker = self.model_df_length

        return df_game_features, last_event, tracker

# Test the GameClient class
game_client = GameClient()
game_id = 2019020001
file_path = game_client.get_game(game_id)
if file_path is not None:
    df_game_features, last_event, tracker = game_client.ping_game(file_path)
    print(df_game_features)
    print(last_event)
else:
    print("Game data not available.")
