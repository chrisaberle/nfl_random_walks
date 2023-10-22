import pandas as pd
import numpy as np
import random
import polars as pl
import math
from extract import extract_data
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Polars
cfg = pl.Config()
cfg.set_tbl_rows(2000)
cfg.set_tbl_cols(1000)

# If scraping is being used, fetch new data. Otherwise, use the example file or one prepared outside of the script
if os.getenv('SCRAPE_URL'):
    df = extract_data()
else:
    df = pd.read_csv('./data/example.csv')

# Determine the first week in the data, this will change as the season progresses
first_week_column = df.columns[1]

# Teams to drop because they've already been chosen this season
teams_to_drop = ['BAL', 'DAL', 'KC', 'PHI', 'MIA', 'LV']

# Drop rows where 'TEA' column matches any value in teams_to_drop
df = df[~df['TEA'].isin(teams_to_drop)]

# Remove team names from columns 2-19, leaving only the spreads
spread_columns = df.columns[1:]
for col in spread_columns:
    df[col] = df[col].str.extract('([+-]?\d+\.?\d*)', expand=False)

# Convert the spreads to integer values and BYEs to NaNs
df[spread_columns] = df[spread_columns].apply(pd.to_numeric, errors='coerce', downcast='integer')

# Calculate the sum of negative spreads for each week
sum_neg_spreads = df[df.applymap(np.isreal)].apply(lambda x: x[x < 0].sum())

# Initialize variables to keep track of the best (lowest sum) walk
best_walk = None
best_sum = float('inf')

# Define the weights for each week
week_weights = {
    # Old probabilities based on standard pool washout rates
    '1': 0.7570,
    '2': 0.5885,
    '3': 0.4500,
    '4': 0.3335,
    '5': 0.2475,
    '6': 0.1930,
    '7': 0.1610,
    '8': 0.1020,
    '9': 0.0835,
    '10': 0.0400,
    '11': 0.0260,
    '12': 0.0210,
    '13': 0.0155,
    '14': 0.0110,
    '15': 0.0080,
    '16': 0.0050,
    '17': 0.0040,
    '18': 0.0010
    # New probabilities based on binomial distribution and washout rates beginning week 10
    # '1': 1,
    # '2': 1,
    # '3': 0.98438,
    # '4': 0.94922,
    # '5': 0.89648,
    # '6': 0.83057,
    # '7': 0.75641,
    # '8': 0.67854,
    # '9': 0.60068,
    # '10': 0.52559,
    # '11': 0.39787,
    # '12': 0.30930,
    # '13': 0.23651,
    # '14': 0.17528,
    # '15': 0.13008,
    # '16': 0.10143,
    # '17': 0.84619,
    # '18': 0.00052
}

# Calculate combined scores using the inverse of the weights, skipping the 'TEA' column
combined_scores = {week: math.log(1 / week_weights[week]) * sum_neg_spreads[week] for week in sum_neg_spreads.index if week != 'TEA' and week in week_weights}

# Sort the weeks by the combined scores
sorted_weeks = sorted(combined_scores, key=combined_scores.get, reverse=True)
print(f'The week order is {sorted_weeks}')

# Columns corresponding to weeks (adjusted to start from index 1)
week_columns = [str(week) for week in sorted_weeks]

# Number of valid random walks generated
valid_walks = 0

# Target number of random walks to generate
num_walks = 530000  # Adjust as needed

# Spinning pipe characters for loading indicator
spinning_pipe = ['|', '/', '-', '\\']

# Initialize a dictionary to keep track of the team selections for each week
walk_dict = {}

# Initialize a variable to keep track of the largest spread encountered during each walk
largest_spread = float('inf')

# Weighting factor for scalarization
alpha_str = os.getenv('ALPHA')
beta_str = os.getenv('BETA')
gamma_str = os.getenv('GAMMA')

alpha = float(alpha_str)
beta = float(beta_str)
gamma = float(gamma_str)

# Initialize a variable to keep track of the best scalarized objective value
best_scalarized_obj = float('-inf')

while valid_walks < num_walks:
    # Initialize variables for this walk
    available_rows = set(range(df.shape[0]))  # Rows that can still be used
    walk_sum = 0
    walk_largest_spread = float('inf')  # Reset for this walk
    walk_smallest_spread = float('-inf') # Reset for this walk

    # Flag to indicate if this walk should be abandoned
    abandon_walk = False

    # Handle the weeks in order of their difficulty
    for week in sorted_weeks:

        if week not in week_columns:
            continue  # Skip weeks not in 2-18 range

        # Identify the lowest three spreads for this week among available rows
        low_spread_rows = df.loc[
            (df.index.isin(available_rows)) & (df[week].notna()) & (df[week] < 0),
            week
        ].nsmallest(3).index.tolist()

        # Abandon this walk if no suitable rows are available
        if not low_spread_rows:
            abandon_walk = True
            break

        # Randomly pick one of these rows
        chosen_row = random.choice(low_spread_rows)
        chosen_team = df.loc[chosen_row, 'TEA']
        chosen_spread = df.loc[chosen_row, week]

        # Update the dictionaries with the selected team for this week
        walk_dict[week] = chosen_team

        # Update the sum of spreads, largest spread, and average spread for this walk
        walk_sum += chosen_spread
        walk_largest_spread = min(walk_largest_spread, chosen_spread)
        walk_smallest_spread = max(walk_smallest_spread, chosen_spread)
        walk_avg_spread = walk_sum / len(sorted_weeks)  # Assuming you are considering all weeks

        # Remove the chosen row from the available rows
        available_rows.remove(chosen_row)

    # Skip this walk if it was flagged to be abandoned
    if abandon_walk:
        continue

    # Convert the dictionary back to a list in the original chronological order
    walk_teams = [walk_dict[week] for week in week_columns if week in walk_dict]

    # Increment the counter for valid walks
    valid_walks += 1

    # Scalarize the objectives for this walk
    walk_scalarized_obj = beta * abs(walk_avg_spread) - alpha * abs(walk_largest_spread) + gamma * abs(walk_smallest_spread)

    # Check if this walk has a better scalarized objective than the current best
    if walk_scalarized_obj > best_scalarized_obj:
        best_scalarized_obj = walk_scalarized_obj
        best_sum = walk_sum
        best_avg = walk_avg_spread
        best_walk = walk_teams.copy()
        largest_spread = walk_largest_spread  # Update the largest spread for the best walk
        smallest_spread = walk_smallest_spread

    # Update the console with the current status and a loading indicator
    print(f"\rIteration: {valid_walks} | Best score: {best_scalarized_obj} | Best spread: {best_sum} | Largest spread: {largest_spread} | Smallest spread: {smallest_spread} | Average spread: {best_avg} | Best walk: {best_walk} {spinning_pipe[valid_walks % 4]}", end='')

# Initialize a DataFrame to store team names for each week
team_df = df.copy()
for week in week_columns:
    team_df[week] = ''

# Populate the DataFrame with team names
for week, team in zip(week_columns, best_walk):
    row_idx = df[df['TEA'] == team].index[0]
    spread = df.at[row_idx, week]
    team_with_spread = f"{team} ({spread})"
    team_df.at[row_idx, week] = team_with_spread

# Sort the DataFrame alphabetically by the team names
team_df.sort_values(by=['TEA'], inplace=True)

# Print final results
print(f"\nFinal best sum: {best_sum}")
print(f"Final best walk: {best_walk}")

# Convert the sorted DataFrame to a Polars DataFrame for pretty-printing
team_df_pl = pl.DataFrame(team_df)

# Display the entire Polars DataFrame
print(team_df_pl)