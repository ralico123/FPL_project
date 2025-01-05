import pandas as pd
import os
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import requests
from pulp import *

app = Flask(__name__)

# Ensure the folder for saving CSV files exists
UPLOAD_FOLDER = 'scraped_data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
prediction_data_file = r"C:\Users\User\Downloads\FPL_project\FPL_project\PointPredictorPrice.csv"
# Function to load prediction data from CSV
def load_prediction_data():
    try:
        # Load the PlayerPredictedPrice.csv into a pandas DataFrame
        
        prediction_data = pd.read_csv(prediction_data_file)

        # Create a dictionary for quick lookups (assuming 'player_id' is the column for matching)
        prediction_dict = {}
        for _, row in prediction_data.iterrows():
            prediction_dict[row['player_id']] = {  # Mapping by player_id
                'player_name': row['Player'].strip(),  # Strip to remove leading/trailing spaces
                'PredP1': row['PredP1'],
                'PredP3': row['PredP3'],
                'PredP5': row['PredP5'],
                'price': row['price'],
                'Squad' : row['Squad']
            }

        return prediction_dict
    except Exception as e:
        print(f"Error loading prediction data: {e}")
        return {}

# Function to scrape FPL data
def scrape_fpl_data(team_id):
    picks_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/19/picks/"
    entry_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"

    try:
        # Fetch picks data
        picks_response = requests.get(picks_url)
        picks_response.raise_for_status()
        picks_data = picks_response.json()

        # Load prediction data from CSV (this will now include player names, predictions, and prices)
        prediction_dict = load_prediction_data()

        # Map element IDs to player names and include prediction data from CSV
        picks_info = []
        for pick in picks_data.get("picks", []):
            element_id = pick.get("element")

            # Get the prediction data from the prediction_dict using the player_id (element_id)
            prediction_data = prediction_dict.get(element_id, {  # Match by player_id (element)
                'player_name': "Unknown Player",
                'PredP1': None,
                'PredP3': None,
                'PredP5': None,
                'price': None,
                'Squad': None
            })

            picks_info.append({
                "player_id": element_id,
                "player_name": prediction_data['player_name'],
                "multiplier": pick.get("multiplier"),
                "Pos": pick.get("element_type"),  # Assuming 'element_type' is the position
                "PredP1": prediction_data['PredP1'],
                "PredP3": prediction_data['PredP3'],
                "PredP5": prediction_data['PredP5'],
                "price": prediction_data['price'],
                "Squad": prediction_data['Squad']
                
            })

        # Fetch player entry details (team information including bank balance)
        entry_response = requests.get(entry_url)
        entry_response.raise_for_status()
        entry_data = entry_response.json()
        picks_response = requests.get(picks_url)
        picks_response.raise_for_status()
        picks_data = picks_response.json()
        player_info = {
            "first_name": entry_data.get("player_first_name"),
            "last_name": entry_data.get("player_last_name"),
            "overall_points": entry_data.get("summary_overall_points"),
            "overall_rank": entry_data.get("summary_overall_rank"),
            "entry_name": entry_data.get("name"),
            "bank": entry_data.get("last_deadline_bank"), 
        }
        print("HERE IS THE PLAYER INFO")
        print(player_info)

        return picks_info, player_info

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    return None, None


# Flask route for the home page (to enter team_id)
@app.route('/')
def index():
    return render_template('index.html')  # Ensure this form in index.html is correctly sending 'team_id'

# Flask route to handle the team_id input and scrape FPL data
@app.route('/scrape', methods=['POST'])
def scrape_route():
    team_id = request.form.get('team_id')

    if not team_id:
        return "Error: team_id is required"

    try:
        team_id = int(team_id)  # Convert to integer if possible
    except ValueError:
        return "Error: team_id must be an integer"

    picks, player_data = scrape_fpl_data(team_id)

    if picks and player_data:
        # Save data to CSV files
        picks_file = os.path.join(app.config['UPLOAD_FOLDER'], f'Picks_Data_{team_id}.csv')
        player_file = os.path.join(app.config['UPLOAD_FOLDER'], f'Player_Data_{team_id}.csv')

        # Save the CSV files
        pd.DataFrame(picks).to_csv(picks_file, index=False)
        pd.DataFrame([player_data]).to_csv(player_file, index=False)

        # Redirect to the "My Team" page
        return redirect(url_for('my_team', team_id=team_id))
    else:
        return "Error: Could not fetch the data for the provided team_id"

# Flask route to show the "My Team" page with scraped data
@app.route('/my_team/<int:team_id>')
def my_team(team_id):
    try:
        picks_file = os.path.join(app.config['UPLOAD_FOLDER'], f'Picks_Data_{team_id}.csv')
        player_file = os.path.join(app.config['UPLOAD_FOLDER'], f'Player_Data_{team_id}.csv')

        picks_df = pd.read_csv(picks_file)
        player_data_df = pd.read_csv(player_file)

        return render_template(
            'my_team.html', 
            picks=picks_df.to_dict(orient='records'), 
            player_data=player_data_df.to_dict(orient='records'),
            team_id=team_id
        )
    except Exception as e:
        return f"Error loading your team data: {e}"


# Flask route to create and show the optimized team
@app.route('/optimized_team/<team_id>')
def optimized_team(team_id):
    try:
        # Load prediction data for optimization
        df = pd.read_csv(prediction_data_file)  # Ensure this file is available

        # Define the optimization problem
        prob = LpProblem("MaximizePoints", LpMaximize)

        # Decision variables: 1 if the player is selected, 0 otherwise
        player_vars = LpVariable.dicts("Player", df.index, cat='Binary')

        # Objective function: maximize total points
        prob += lpSum([df['PredP5'][i] * player_vars[i] for i in df.index])

        # Constraints
        prob += lpSum([df['price'][i] * player_vars[i] for i in df.index]) <= 1000  # Budget constraint
        prob += lpSum([player_vars[i] for i in df.index if df['Pos'][i] == 'GK']) == 2  # GK positions
        prob += lpSum([player_vars[i] for i in df.index if df['Pos'][i] == 'DF']) == 5  # DF positions
        prob += lpSum([player_vars[i] for i in df.index if df['Pos'][i] == 'MF']) == 5  # MF positions
        prob += lpSum([player_vars[i] for i in df.index if df['Pos'][i] == 'FW']) == 3  # FW positions

        # Squad constraint: At most 3 players from any squad
        for squad in df['Squad'].unique():
            prob += lpSum([player_vars[i] for i in df.index if df['Squad'][i] == squad]) <= 3, f"MaxPlayersFrom_{squad}"

        # Solve the problem
        prob.solve()

        # Collect the selected players for the Wildcard Team
        wildcard_players = []
        for v in prob.variables():
            if v.varValue == 1:
                player_index = int(v.name.split("_")[1])
                wildcard_players.append({
                    "Player": df['Player'][player_index],
                    "PredP1": df['PredP1'][player_index],
                    "PredP3": df['PredP3'][player_index],
                    "PredP5": df['PredP5'][player_index],
                    "Price": df['price'][player_index],
                    "Position": df['Pos'][player_index],
                    "player_id": df['player_id'][player_index],
                    "Squad": df['Squad'][player_index]
                })

        # Sort the selected players by position (GK, DF, MF, FW)
        sorted_wildcard_players = sorted(wildcard_players, key=lambda x: x["Position"])

        # Load the previously scraped "Current Team" data
        picks_file = os.path.join(app.config['UPLOAD_FOLDER'], f'Picks_Data_{team_id}.csv')
        current_team_df = pd.read_csv(picks_file)

        # Function to compute the sum of PredP1, PredP3, and PredP5 for a given DataFrame
        def compute_sums(df):
            predp1_sum = df['PredP1'].sum()
            predp3_sum = df['PredP3'].sum()
            predp5_sum = df['PredP5'].sum()
            return predp1_sum, predp3_sum, predp5_sum

        # Compute sums for Wildcard team
        wildcard_p1_sum, wildcard_p3_sum, wildcard_p5_sum = compute_sums(pd.DataFrame(wildcard_players))

        # Compute sums for Current Team
        current_p1_sum, current_p3_sum, current_p5_sum = compute_sums(current_team_df)

        # Safely compute the percentage comparison between Current Team and Wildcard
        result1 = 0 if wildcard_p1_sum == 0 else 100 * current_p1_sum / wildcard_p1_sum
        result3 = 0 if wildcard_p3_sum == 0 else 100 * current_p3_sum / wildcard_p3_sum
        result5 = 0 if wildcard_p5_sum == 0 else 100 * current_p5_sum / wildcard_p5_sum

        # Ensure that results are treated as floats
        result1 = float(result1)
        result3 = float(result3)
        result5 = float(result5)

        df['PredP5'] = pd.to_numeric(df['PredP5'], errors='coerce').fillna(0)
        
        # Create a DataFrame from the results
        results_df = pd.DataFrame({
            'Metric': ['PredP1', 'PredP3', 'PredP5'],
            'Percentage': [result1, result3, result5]
        })

        # Return optimized team results to the template along with the comparison data
        return render_template('optimized_team.html', 
                       optimized_team=wildcard_players,  # Pass the players as a list of dictionaries
                       team_id=team_id,
                       wildcard_p1_sum=wildcard_p1_sum, 
                       wildcard_p3_sum=wildcard_p3_sum, 
                       wildcard_p5_sum=wildcard_p5_sum,
                       current_p1_sum=current_p1_sum, 
                       current_p3_sum=current_p3_sum, 
                       current_p5_sum=current_p5_sum, 
                       results_df=results_df.to_html(index=False),
                       result1=result1,
                       result3=result3,
                       result5=result5)
    except Exception as e:
        return f"Error loading optimized team: {e}"



def optimize_team_with_bank(existing_players, num_transfers, bank, total_budget, range=5):
    """
    Optimizes the team by maximizing points considering bank and transfers.

    Args:
        existing_players: List of player IDs in the current team.
        num_transfers: Maximum number of transfers allowed.
        bank: The remaining budget.
        total_budget: The total budget (player value + bank).
        range: Number of gameweeks to consider.

    Returns:
         A list of player IDs representing the optimized team.
    """

    # Load player data (replace with your actual data loading)
    try:
        df = pd.read_csv('PointPredictorPrice.csv')
    except FileNotFoundError:
        print("Error: 'PointPredictorPrice.csv' not found.")
        return None

    # Define the optimization problem
    prob = LpProblem("OptimalTeam", LpMaximize)

    # Decision variables (1 if player is in the optimized team, 0 otherwise)
    player_vars = LpVariable.dicts("Player", df.index, cat='Binary')
    transfer_vars = LpVariable.dicts("Transfer", df.index, cat='Binary')  # Track transfers

    # Objective Function: Maximize total predicted points over specified gameweeks
    objective_function = lpSum([df[f'PredP{range}'][i] * player_vars[i] for i in df.index])
    prob += objective_function

    # Constraints
    # 1. Budget Constraint: Total cost of players must be within total budget
    prob += lpSum([df['price'][i] * player_vars[i] for i in df.index]) <= total_budget

    # 2. Team Size Constraint: Must have 15 players
    prob += lpSum([player_vars[i] for i in df.index]) == 15

    # 3. Positional constraints
    prob += lpSum([player_vars[i] for i in df.index if df['Pos'][i] == 'GK']) == 2
    prob += lpSum([player_vars[i] for i in df.index if df['Pos'][i] == 'DF']) == 5
    prob += lpSum([player_vars[i] for i in df.index if df['Pos'][i] == 'MF']) == 5
    prob += lpSum([player_vars[i] for i in df.index if df['Pos'][i] == 'FW']) == 3

    # 4. Transfer Constraint
    prob += lpSum([transfer_vars[i] for i in df.index]) <= num_transfers

    # 5. Existing players constraint
    for player_id in existing_players:
        player_index = df.index[df['player_id'] == player_id].tolist()
        if player_index:  # Check if player id exists in the dataframe
            prob += player_vars[player_index[0]] == 1

    # Solve the problem
    prob.solve()

    optimized_team = []
    for v in prob.variables():
        if v.name.startswith("Player_") and v.varValue == 1:
            player_index = int(v.name.split("_")[1])
            optimized_team.append(df['player_id'][player_index])

    return optimized_team

@app.route('/transfers', methods=['GET', 'POST'])
def transfers():
    from itertools import combinations

    team_id = request.args.get('team_id') or request.form.get('team_id')

    if not team_id:
        return "Error: team_id is required"

    picks_info, player_info = scrape_fpl_data(team_id)

    if not picks_info or not player_info:
        return "Error: Could not fetch the team data. Please try again later."

    try:
        # Map positions and process input data
        position_mapping = {1: "GK", 2: "DF", 3: "MF", 4: "FW"}
        for pick in picks_info:
            pick['Pos'] = position_mapping.get(int(pick['Pos']), "Unknown")
            
        df = pd.read_csv(prediction_data_file)
        if df.empty:
            print("ERROR: DataFrame from CSV is EMPTY!")
            return "Error: Prediction data is empty."

        df['player_id'] = df['player_id'].astype(int)
        df['price'] = df['price'].astype(float)
        df['PredP5'] = pd.to_numeric(df['PredP5'], errors='coerce').fillna(0)
        df['Pos'] = df['Pos'].astype(str)
        df['Squad'] = df['Squad'].astype(str)
        
        for pick in picks_info:
            pick['player_id'] = int(pick['player_id'])
            pick['price'] = float(pick['price'])
            pick['Pos'] = str(pick['Pos'])
            pick['Squad'] = str(pick['Squad'])

        bank = player_info.get("bank", 0) or 0
        existing_players = [pick['player_id'] for pick in picks_info]
        total_budget = sum([pick['price'] for pick in picks_info]) + bank

        squad_counts = df[df['player_id'].isin(existing_players)]['Squad'].value_counts().to_dict()

        if request.method in ['GET', 'POST']:
            num_transfers = int(request.args.get('num_transfers', request.form.get('num_transfers', 1)))

            # Global optimization for transfers
            best_combination = None
            best_total_point_diff = float('-inf')

            # Generate all possible transfer combinations
            for transfer_combination in combinations(picks_info, num_transfers):
                # Prepare variables for this combination
                remaining_budget = bank + sum(pick['price'] for pick in transfer_combination)
                temp_squad_counts = squad_counts.copy()
                temp_existing_players = existing_players.copy()
                total_point_diff = 0
                valid_combination = True
                transfers = []

                # Process each transfer in the combination
                for player_to_sell in transfer_combination:
                    pos_condition = df['Pos'] == player_to_sell['Pos']
                    price_condition = df['price'] <= remaining_budget
                    not_in_team_condition = ~df['player_id'].isin(temp_existing_players)
                    squad_condition = df['Squad'].map(lambda s: temp_squad_counts.get(s, 0) < 3)

                    potential_buys = df[pos_condition & price_condition & not_in_team_condition & squad_condition]
                    
                    if potential_buys.empty:
                        valid_combination = False
                        break

                    # Choose the best buy for this sale
                    best_buy = potential_buys.loc[potential_buys['PredP5'].idxmax()]
                    point_diff = best_buy['PredP5'] - player_to_sell['PredP5']
                    total_point_diff += point_diff

                    # Update remaining budget, squad, and existing players
                    remaining_budget -= (best_buy['price'] - player_to_sell['price'])
                    temp_existing_players.remove(player_to_sell['player_id'])
                    temp_existing_players.append(best_buy['player_id'])
                    temp_squad_counts[player_to_sell['Squad']] -= 1
                    temp_squad_counts[best_buy['Squad']] = temp_squad_counts.get(best_buy['Squad'], 0) + 1

                    # Record the transfer
                    transfers.append({
                        'sell': player_to_sell,
                        'buy': best_buy.to_dict(),
                        'point_diff': point_diff
                    })

                # Update the best combination if this one is better
                if valid_combination and total_point_diff > best_total_point_diff:
                    best_total_point_diff = total_point_diff
                    best_combination = transfers

            return render_template(
                'transfers.html',
                team_id=team_id,
                picks_info=picks_info,
                player_info=player_info,
                suggested_transfers=best_combination or [],
                bank=bank
            )

        return render_template('transfers.html', team_id=team_id, picks_info=picks_info, player_info=player_info, bank=bank)

    except Exception as e:
        print(f"ERROR in transfers function: {e}")
        import traceback
        traceback.print_exc()
        return f"An error occurred during transfer processing: {e}"


@app.route('/player_suggestions/<int:team_id>', methods=['GET', 'POST'])
def player_suggestions(team_id):
    try:
        # Load team and prediction data
        picks_info, player_info = scrape_fpl_data(team_id)
        prediction_df = pd.read_csv(prediction_data_file)
        
        if not picks_info or prediction_df.empty:
            return "Error: Could not fetch data for the team or prediction file is empty."

        # Format the prediction data
        prediction_df['player_id'] = prediction_df['player_id'].astype(int)
        prediction_df['price'] = prediction_df['price'].astype(float)
        prediction_df['PredP5'] = pd.to_numeric(prediction_df['PredP5'], errors='coerce').fillna(0)

        # POST: Handle user selection and provide suggestions
        if request.method == 'GET':
            selected_player_id = int(request.form.get('selected_player_id'))
            bank = player_info.get("bank", 0) or 0
            remaining_budget = bank

            # Get details of the selected player
            selected_player = next((p for p in picks_info if p['player_id'] == selected_player_id), None)
            if not selected_player:
                return "Error: Selected player not found in the team."

            # Apply filters to find valid replacements
            pos_condition = prediction_df['Pos'] == selected_player['Pos']
            price_condition = prediction_df['price'] <= (remaining_budget + selected_player['price'])
            not_in_team_condition = ~prediction_df['player_id'].isin([p['player_id'] for p in picks_info])
            squad_counts = {p['Squad']: sum(1 for pick in picks_info if pick['Squad'] == p['Squad']) for p in picks_info}
            squad_condition = prediction_df['Squad'].map(lambda s: squad_counts.get(s, 0) < 3)

            # Filter potential replacements
            potential_replacements = prediction_df[pos_condition & price_condition & not_in_team_condition & squad_condition]

            # Sort by predicted points and select the top 3
            top_replacements = potential_replacements.sort_values('PredP5', ascending=False).head(3)

            # Prepare data for rendering
            top_replacements_list = top_replacements.to_dict(orient='records')
            return render_template(
                'player_suggestions.html',
                team_id=team_id,
                picks_info=picks_info,
                player_info=player_info,
                selected_player=selected_player,
                suggestions=top_replacements_list
            )

        # GET: Show the team and allow selection
        return render_template(
            'player_suggestions.html',
            team_id=team_id,
            picks_info=picks_info,
            player_info=player_info
        )
    except Exception as e:
        print(f"Error in player_suggestions route: {e}")
        return f"An error occurred: {e}"
    
import random  # Import the random module

@app.route('/players', methods=['GET', 'POST'])
def players():
    team_id = request.args.get('team_id') or request.form.get('team_id')
    player_id = request.form.get('player_id')  # Get the selected player's ID

    if not team_id:
        return "Error: team_id is required"

    picks_info, player_info = scrape_fpl_data(team_id)

    if not picks_info or not player_info:
        return "Error: Could not fetch the team data. Please try again later."

    # Position mapping: map numeric positions to string names
    position_mapping = {1: "GK", 2: "DF", 3: "MF", 4: "FW"}
    for pick in picks_info:
        pick['Pos'] = position_mapping.get(int(pick['Pos']), "Unknown")  # Map numeric position to string

    # Assuming the prediction data is correctly loaded
    df = pd.read_csv(prediction_data_file)
    df['player_id'] = df['player_id'].astype(int)

    # If no player_id is selected, select a random player
    if not player_id:
        # Randomly select a player from picks_info
        selected_player = random.choice(picks_info)  # Randomly choose a player
        player_id = selected_player['player_id']  # Set player_id to the selected player's ID
        print(f"Randomly selected player: {selected_player['player_name']}")
    else:
        # Find the selected player based on player_id
        selected_player = next((pick for pick in picks_info if pick['player_id'] == int(player_id)), None)

    # Get the bank and remaining budget
    bank = player_info.get("bank", 0) or 0
    total_budget = sum([pick['price'] for pick in picks_info]) + bank

    # Find the best replacements
    recommended_replacements = []
    if selected_player:
        # Find the best 3 replacements based on predicted points and affordability
        pos_condition = df['Pos'] == selected_player['Pos']  # Match position
        price_condition = df['price'] <= (bank + selected_player['price'])  # Match affordability
        potential_replacements = df[pos_condition & price_condition]

        # Exclude the selected player from the potential replacements
        potential_replacements = potential_replacements[potential_replacements['player_id'] != selected_player['player_id']]

        # Sort by predicted points and select top 3
        recommended_replacements = potential_replacements.nlargest(3, 'PredP5').to_dict(orient='records')

    return render_template(
        'players.html', 
        team_id=team_id,
        picks_info=picks_info,
        selected_player=selected_player,
        recommended_replacements=recommended_replacements
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
