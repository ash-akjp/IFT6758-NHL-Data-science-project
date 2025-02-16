import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from milestone1_func import create_game_info_list, add_home_away_rink_side_columns, get_play_data


# TODO some problems with the data based on this histogram (shouldnt increase as it gets farther)

'''
A histogram of shot counts (goals and no-goals separated), binned by distance
'''

'''
2. Feature Engineering I (10%) 
Using the functionality you created for Milestone 1, acquire all of the raw play-by-play data for the 2015/16 season all
the way to the 2019/20 season (inclusive). Note that the tidied data that you created will be useful for the baseline models, 
but you will be creating more features that will require the full raw data in Part 4. 

Set aside all of the 2019/20 data as your final test set. You should not touch this until you reach the end of this milestone. 
You will use the 2015/16 - 2018/19 regular season data to create your training and validation sets. By the end of this milestone, 
you will have built up all the framework you need to process new data, so it should be trivial to process your test data once 
you have finished all of the necessary feature engineering. Until Part 7, any reference to the “dataset” will exclusively refer 
to the 2015/16 - 2018/19 data. 
'''


"""####Question 1"""

'''
Using your training dataset create a tidied dataset for each SHOT/GOAL event, with the following columns
(you can name them however you want):
 - Distance from net
 - Angle from net
 - Is goal (0 or 1)
 - Empty Net (0 or 1; you can assume NaNs are 0)
You can approximate the net as a single point (i.e. you don’t need to account for the width of the net
when computing the distance or angle). You should be able to create this easily using the functionality
you implemented for tidying data in Milestone 1, as you will only need the (x, y) coordinates for each
shot/goal event.
'''


def tidy_data(df):

    df = df.copy()

    tidied = pd.DataFrame(t for i, l in df.iterrows() for t in get_play_data(l, keep_all_events=True))

    # Add column Is goal (0 or 1)
    tidied['IsGoal'] = (tidied['eventTypeId'] == 'GOAL').astype(int)

    # Replace column emptyNet (0 or 1; you can assume NaNs are 0)
    tidied['emptyNet'] = tidied['emptyNet'].replace([np.nan, False, True], [0, 0, 1])

    # add rink_side column
    # seasons = ["20152016", "20162017", "20172018", "20182019"]  # , "20192020"]
    # for season in seasons:
    #     game_info_list = create_game_info_list(season)
    #     add_home_away_rink_side_columns(tidied, game_info_list)

    tidied = add_home_away_rink_side_columns(tidied)

    # Add column Distance from net

    # approximate nets as being (-89, 0) and (89, 0)
    # if rink side is left offense is on the right
    # if rink side is left: trying to score in right net (89, 0)
    # if rink side is right: trying to score in left net (-89, 0)
    tidied['Distance_from_net'] = np.sqrt(
        np.where(
            tidied['rink_side'] == 'left',
            (tidied['x'] - 89) ** 2 + tidied['y'] ** 2,
            (tidied['x'] + 89) ** 2 + tidied['y'] ** 2
        )
    )

    # Add column Angle from net

    # If shoots from right in from of net (y=0), then angle is 0 deg
    # if shoots completely from the side (x=89 or -89), then angle is 90 deg
    # add a temp column x_distance_from_net

    tidied['x_distance_from_net'] = np.where(
        tidied['rink_side'] == 'left',
        abs(89 - tidied['x']),
        abs(-89 - tidied['x'])
    )

    tidied['angle_from_net'] = np.degrees(
        np.arccos(tidied['x_distance_from_net'] / tidied['Distance_from_net'])
    )

    # can remove the x_distance_from_net column
    tidied.drop(['x_distance_from_net'], axis=1, inplace=True)

    return tidied


"""##Figures"""

'''
Create and include the following figures in your blogpost and briefly discuss your observations (few sentences):
 - A histogram of shot counts (goals and no-goals separated), binned by distance
 - A histogram of shot counts (goals and no-goals separated), binned by angle
 - A 2D histogram where one axis is the distance and the other is the angle. You do not need to separate goals and no-goals.
Hint: check out jointplots.
As always, make sure all of your axes are labeled correctly, and you make the appropriate choice of axis scale.

'''


def q1(tidied_training_set):

    df = tidied_training_set.copy()

    # Separate goals and no-goals
    goals = df[df['IsGoal'] == 1]
    no_goals = df[df['IsGoal'] == 0]

    plt.hist([goals['Distance_from_net'], no_goals['Distance_from_net']], bins=8, alpha=1, label=['Goals', 'No-Goals'], color=['red', 'blue'])

    # Set labels and title
    plt.xlabel('Distance from Net')
    plt.ylabel('Shot Count')
    plt.title('Shot Counts Separated by Goals and No-Goals')

    # Add a legend
    plt.legend()

    # Display the histogram
    plt.show()

    '''
    A histogram of shot counts (goals and no-goals separated), binned by angle
    '''

    df = tidied_training_set.copy()

    # Separate goals and no-goals
    goals = df[df['IsGoal'] == 1]
    no_goals = df[df['IsGoal'] == 0]

    plt.hist([goals['angle_from_net'], no_goals['angle_from_net']], bins=8, alpha=1, label=['Goals', 'No-Goals'], color=['red', 'blue'])

    # Set labels and title
    plt.xlabel('Angle from Net')
    plt.ylabel('Shot Count')
    plt.title('Shot Counts Separated by Goals and No-Goals')

    # Add a legend
    plt.legend()

    # Display the histogram
    plt.show()

    '''
    A 2D histogram where one axis is the distance and the other is the angle. You do not need to separate goals and no-goals.
    '''

    df = tidied_training_set.copy()

    # Create a 2D histogram using jointplot
    sns.jointplot(data=df, x='Distance_from_net', y='angle_from_net', kind='hist', cmap='Blues')

    # Display the plot
    plt.show()


"""####Question 2"""

'''
Now, create two more figures relating the goal rate, i.e. #goals / (#no_goals + #goals), to the distance,
and goal rate to the angle of the shot.
Include these figures in your blogpost and briefly discuss your observations.
'''


"""####Question 3"""

'''
Finally, let’s do some quick checks to see if our data makes sense.
Unfortunately we don’t have time to do automated anomaly detection, but we can use our “domain knowledge”
for some quick sanity checks! The domain knowledge is that “it is incredibly rare to score a non-empty net
goal on the opposing team from within your defensive zone”. Knowing this, create another histogram, this
time of goals only, binned by distance, and separate empty net and non-empty net events. Include this figure
in your blogpost and discuss your observations. Can you find any events that have incorrect features
(e.g. wrong x/y coordinates)? If yes, prove that one event has incorrect features.
Hint: the NHL gamecenter usually has video clips of goals for every game.
'''


def q3(tidied_training_set):

    df = tidied_training_set.copy()

    # Filter rows for goals only
    goals = df[df['IsGoal'] == 1]

    # Separate empty net and non-empty net goals
    empty_net_goals = goals[goals['emptyNet'] == 1]
    non_empty_net_goals = goals[goals['emptyNet'] == 0]

    plt.hist([
        empty_net_goals['Distance_from_net'],
        non_empty_net_goals['Distance_from_net']
        ], bins=8, alpha=1, label=['Empty Net Goals', 'Non-Empty Net Goals'], color=['red', 'blue'])

    # Set labels and title
    plt.xlabel('Distance from Net')
    plt.ylabel('Goal Count')
    plt.title('Goals Separated by Empty Net and Non-Empty Net (Binned by Distance)')

    # Add a legend
    plt.legend()

    # Display the histogram
    plt.show()

    non_empty_net_goals = goals[goals['emptyNet'] == 0]
    non_empty_net_goals.iloc[1]

    # There is a mistake in the coordinates: I checked https://www.nhl.com/gamecenter/mtl-vs-tor/2015/10/07/2015020001/playbyplay
    # and https://www.youtube.com/watch?v=KNsWliHJnCE
    # the same mistake appears on the nhl Play By Play map

    non_empty_net_goals = goals[goals['emptyNet'] == 0]

    non_empty_net_goals_sorted = non_empty_net_goals.sort_values(by='Distance_from_net', ascending=False)

    non_empty_net_goals_sorted.iloc[0]