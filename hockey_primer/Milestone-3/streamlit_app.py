import streamlit as st
import pandas as pd
import numpy as np
import joblib
from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import GameClient
import os
import requests
import plotly.express as px


IP = os.environ.get('SERVING_IP', "serving")
PORT = os.environ.get("SERVING_PORT", "8000")

# Create instances of the serving and game clients
serving_client = ServingClient(ip=IP, port=int(PORT))
game_client = GameClient()

# Initialize session state
if 'old_df' not in st.session_state:
    st.session_state.old_df = None

if 'game_id' not in st.session_state:
    st.session_state.game_id = None


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
    game_id = st.text_input("Game ID", "")

    if st.button("Ping Game"):
        # Explicitly set old_df to None when the game ID changes
        if st.session_state.old_df is not None and st.session_state.game_id != game_id:
            st.session_state.old_df = None

        file_path = game_client.get_game(game_id)
        if file_path is not None:
            df_game_features, last_event, tracker = game_client.ping_game(file_path)
            if df_game_features is not None:
                # Display game info
                st.header(f"Game {game_id} : {last_event['homeTeamName']} vs {last_event['awayTeamName']}")
                time_left = get_current_timeleft(game_id)
                st.write(f"Period: {last_event['period']} - Time remaining: {time_left}")

                X = df_game_features

                # Filter new shots based on index between old_df and X
                if st.session_state.old_df is not None:
                    new_shots = X.loc[~X.index.isin(st.session_state.old_df.index)]
                else:
                    new_shots = X.copy()

                # Check if new_shots is empty (None)
                if new_shots is None or new_shots.empty:
                    #st.warning("No new shots have been added.")
                    # Set full_df equal to old_df
                    full_df = st.session_state.old_df.copy() if st.session_state.old_df is not None else None

                else:
                    #st.write("New shots:")
                    #st.write(new_shots)

                    # Perform prediction for new shots
                    pred_probs_new = serving_client.predict(new_shots)

                    # Extract the second element from each sublist
                    second_elements = [item[1] for item in pred_probs_new['predicted']]
                    second_elements_df = pd.DataFrame(second_elements, columns=['Goal Probabilities'])

                    # Concatenate new shots and their predicted probabilities
                    concat_new_df = pd.concat([new_shots, second_elements_df], axis=1)

                    # Combine old_df and concat_new_df
                    if st.session_state.old_df is not None:
                        full_df = pd.concat([st.session_state.old_df, concat_new_df], ignore_index=True)
                    else:
                        full_df = concat_new_df.copy()

                    # Update old_df
                    st.session_state.old_df = full_df.copy()

                # Perform other calculations using full_df

                home_actual = full_df[full_df['eventOwnerTeamId'] == full_df['homeTeamId']]['IsGoal'].cumsum().iloc[-1]
                away_actual = full_df[full_df['eventOwnerTeamId'] == full_df['awayTeamId']]['IsGoal'].cumsum().iloc[-1]

                # Calculate the cumulative sum of expected goals for home and away teams
                cumulative_xG_home = full_df[full_df['eventOwnerTeamId'] == full_df['homeTeamId']]['Goal Probabilities'].cumsum().iloc[-1]
                cumulative_xG_away = full_df[full_df['eventOwnerTeamId'] == full_df['awayTeamId']]['Goal Probabilities'].cumsum().iloc[-1]
                cumulative_xG_home = round(cumulative_xG_home, 2)
                cumulative_xG_away = round(cumulative_xG_away, 2)

                diff_home = round(home_actual - cumulative_xG_home, 2)
                diff_away = round(away_actual - cumulative_xG_away, 2)

                col1, col2 = st.columns(2)
                col1.metric(label=f"{last_event['homeTeamName']} (Actual)", value=f"{cumulative_xG_home} ({home_actual})", delta=diff_home, delta_color="off")
                col2.metric(label=f"{last_event['awayTeamName']} (Actual)", value=f"{cumulative_xG_away} ({away_actual})", delta=diff_away, delta_color="off")

        st.header("Data used for predictions and Predictions:")
        if game_id is not False and not None:
            st.write(full_df)
        
        # Save the current game_id to session_state
        st.session_state.game_id = game_id

        #Additional functionality
        st.subheader("Some added functionality:")

        # Create a Shot Distribution Heatmap
        fig_heatmap = px.scatter(full_df, x='xCoord', y='yCoord', color='IsGoal', size='Goal Probabilities',
                                title='Shot Distribution Heatmap', labels={'IsGoal': 'Goal'})
        fig_heatmap.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_heatmap)


        # Create a Scatter Plot with 'angleFromNet' and 'distanceFromNet'
        fig_scatter = px.scatter(full_df, x='angleFromNet', y='distanceFromNet', color='IsGoal', size='Goal Probabilities',
                                title='Shot Scatter Plot', labels={'IsGoal': 'Goal'})
        fig_scatter.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_scatter)


        # Create a bar chart
        total_goals_home = full_df[full_df['eventOwnerTeamId'] == full_df['homeTeamId']]['IsGoal'].sum()
        total_goals_away = full_df[full_df['eventOwnerTeamId'] == full_df['awayTeamId']]['IsGoal'].sum()
        fig_bar = px.bar(x=[last_event['homeTeamName'], last_event['awayTeamName']],
                        y=[total_goals_home, total_goals_away],
                        labels={'x': 'Team', 'y': 'Total Goals'},
                        title='Total Goals Comparison',
                        color=['Home', 'Away'])
        fig_bar.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_bar)


        # Create a Line Graph with 'angleFromNet' and 'distanceFromNet' as x and y, and 'Goal Probabilities' as color
        fig_line = px.line(full_df, x='angleFromNet', y='Goal Probabilities', color='distanceFromNet',
                        title='Goal Probabilities vs Angle from Net and Distance from Net',
                        labels={'angleFromNet': 'Angle from Net', 'distanceFromNet': 'Distance from Net'})
        fig_line.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_line)

        # Create a scatter plot
        fig = px.scatter_3d(
            full_df,
            x='distanceFromNet',
            y='angleFromNet',
            z='Goal Probabilities',
            color='IsGoal',
            labels={'distanceFromNet': 'Distance from Net', 'angleFromNet': 'Angle from Net', 'Goal Probabilities': 'Goal Probabilities'},
            title=f'Game {game_id} - Distance, Angle, and Goal Probability 3D Map'
        )
        st.plotly_chart(fig)


# Display logs from the Flask server
logs = serving_client.logs()
# st.write("Server Logs:")
# st.write(logs)