import streamlit as st
import pandas as pd
import numpy as np
import joblib
from serving_client import ServingClient
from game_client import GameClient
import os
import requests

IP = os.environ.get('SERVING_IP', "0.0.0.0")
PORT = os.environ.get("SERVING_PORT", "8001")

# Create instances of the serving and game clients
serving_client = ServingClient(ip=IP, port=int(PORT))
game_client = GameClient()
df=None

def get_current_period(game_id):
    try:
        API_URL = "https://api-web.nhle.com/v1/gamecenter/"
        response = requests.get(API_URL + str(game_id) + "/boxscore")
        data = response.json()
        return data['period']
    except:
        return "Issue with api"
    
def get_current_timeleft(game_id):
    try:
        response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
        data = response.json()
        
        # Get the last play in the "plays" list
        last_play = data.get("plays", [])[-1]
        
        # Extract the time remaining from the last play
        time_remaining = last_play.get("timeRemaining", "N/A")

        return time_remaining
    except Exception as e:
        print(f"Error: {e}")
        return "Issue with API"

st.title("NHL Game Dashboard")

with st.sidebar:
    # Text input for CometML model parameters
    workspace = st.text_input("Workspace", "ift6758-milestone2-team07")
    model_name = st.text_input("Model", "simple_both")
    version = st.text_input("Version", "")

    # Button to download the model
    if st.button("Download Model"):
        response, status_code = serving_client.download_registry_model(workspace, model_name, version)
        st.write(response)

with st.container():
    # Text input for Game ID
    game_id = st.text_input("Game ID", "")

    # Button to ping the game
    if st.button("Ping Game"):
        file_path = game_client.get_game(game_id)
        if file_path is not None:
            df_game_features, last_event, tracker = game_client.ping_game(file_path)
            if df_game_features is not None:

                # Perform prediction using the loaded model
                X = df_game_features  # Use the entire dataframe as input for prediction
                predicted_probabilities = serving_client.predict(X)
                #st.write(f"Model Predicted Probabilities: {predicted_probabilities}")

                # Extract the second element from each sublist
                second_elements = [item[1] for item in predicted_probabilities['predicted']]
                second_elements_df = pd.DataFrame(second_elements, columns=['Goal Probabilities'])

                # Display game info and predictions
                st.header(f"Game {game_id} : {last_event['homeTeamName']} vs {last_event['awayTeamName']}")
                #st.write(f"Period: {last_event['period']}")

                #period = get_current_period(game_id)
                time_left = get_current_timeleft(game_id)
                #st.write(time_left)

                st.write(f"Period: {last_event['period']} - Time remaining: {time_left}")

                df = pd.concat([X, second_elements_df], axis=1)

                home_actual = df[df['eventOwnerTeamId'] == df['homeTeamId']]['IsGoal'].cumsum().iloc[-1]
                away_actual = df[df['eventOwnerTeamId'] == df['awayTeamId']]['IsGoal'].cumsum().iloc[-1]

                 # Calculate the cumulative sum of expected goals for home and away teams
                cumulative_xG_home = df[df['eventOwnerTeamId'] == df['homeTeamId']]['Goal Probabilities'].cumsum().iloc[-1]
                cumulative_xG_away = df[df['eventOwnerTeamId'] == df['awayTeamId']]['Goal Probabilities'].cumsum().iloc[-1]
                cumulative_xG_home = round(cumulative_xG_home, 2)
                cumulative_xG_away = round(cumulative_xG_away, 2)

                diff_home = round(home_actual - cumulative_xG_home, 2)
                diff_away = round(away_actual - cumulative_xG_away, 2)

                col1, col2 = st.columns(2)
                col1.metric(label = f"{last_event['homeTeamName']} (Actual)", value = f"{cumulative_xG_home} ({home_actual})", delta = diff_home, delta_color="off" )
                col2.metric(label = f"{last_event['awayTeamName']} (Actual)", value = f"{cumulative_xG_away} ({away_actual})", delta = diff_away, delta_color="off" )

        st.header("Data used for predictions and Predictions:")
        if game_id is not False and not None:
            st.write(df)

# Display logs from the Flask server
logs = serving_client.logs()
#st.write("Server Logs:")
#st.write(logs)
