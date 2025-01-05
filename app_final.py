from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import random
import requests
import json
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
from itertools import combinations

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a secure secret key

# Global variables
team_data_store = {}
prediction_df = None

def init_prediction_data():
    global prediction_df
    try:
        prediction_df = pd.read_csv('PointPredictorPrice.csv')
        print(f"Successfully loaded prediction data with {len(prediction_df)} rows")
    except Exception as e:
        print(f"Error loading prediction data: {e}")
        prediction_df = pd.DataFrame()

def load_prediction_data():
    global prediction_df
    prediction_dict = {}
    try:
        for _, row in prediction_df.iterrows():
            prediction_dict[row['player_id']] = {
                'player_name': row['Player'].strip(),
                'PredP1': row['PredP1'],
                'PredP3': row['PredP3'],
                'PredP5': row['PredP5'],
                'price': row['price'],
                'Squad': row['Squad']
            }
        return prediction_dict
    except Exception as e:
        print(f"Error creating prediction dictionary: {e}")
        return {}

def scrape_fpl_data(team_id):
    picks_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/19/picks/"
    entry_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        print(f"Attempting to fetch picks data from: {picks_url}")
        picks_response = requests.get(picks_url, headers=headers, timeout=10)
        print(f"Picks response status code: {picks_response.status_code}")
        print(f"Picks response content: {picks_response.text[:200]}")
        
        picks_response.raise_for_status()
        picks_data = picks_response.json()
        
        print(f"Attempting to fetch entry data from: {entry_url}")
        entry_response = requests.get(entry_url, headers=headers, timeout=10)
        print(f"Entry response status code: {entry_response.status_code}")
        
        entry_response.raise_for_status()
        entry_data = entry_response.json()

        prediction_dict = load_prediction_data()
        picks_info = []
        for pick in picks_data.get("picks", []):
            element_id = pick.get("element")
            prediction_data = prediction_dict.get(element_id, {
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
                "Pos": pick.get("element_type"),
                "PredP1": prediction_data['PredP1'],
                "PredP3": prediction_data['PredP3'],
                "PredP5": prediction_data['PredP5'],
                "price": prediction_data['price'],
                "Squad": prediction_data['Squad']
            })

        player_info = {
            "first_name": entry_data.get("player_first_name"),
            "last_name": entry_data.get("player_last_name"),
            "overall_points": entry_data.get("summary_overall_points"),
            "overall_rank": entry_data.get("summary_overall_rank"),
            "entry_name": entry_data.get("name"),
            "bank": entry_data.get("last_deadline_bank"),
        }
        
        print("Successfully processed team data")
        return picks_info, player_info

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text[:200]}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape_route():
    team_id = request.form.get('team_id')
    print(f"Received request for team_id: {team_id}")

    if not team_id:
        return "Error: team_id is required"

    try:
        team_id = int(team_id)
        picks, player_data = scrape_fpl_data(team_id)
        
        if picks and player_data:
            # Store in memory instead of writing to files
            team_data_store[team_id] = {
                'picks': picks,
                'player_data': player_data
            }
            return redirect(url_for('my_team', team_id=team_id))
        else:
            return "Error: Could not fetch the data for the provided team_id"
    except Exception as e:
        print(f"Error in scrape_route: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/my_team/<int:team_id>')
def my_team(team_id):
    try:
        team_data = team_data_store.get(team_id)
        if not team_data:
            return "Error: Team data not found. Please try scraping again."

        return render_template(
            'my_team.html', 
            picks=team_data['picks'],
            player_data=[team_data['player_data']],
            team_id=team_id
        )
    except Exception as e:
        return f"Error loading your team data: {e}"

@app.route('/optimized_team/<team_id>')
def optimized_team(team_id):
    try:
        global prediction_df
        prob = LpProblem("MaximizePoints", LpMaximize)
        player_vars = LpVariable.dicts("Player", prediction_df.index, cat='Binary')
        
        prob += lpSum([prediction_df['PredP5'][i] * player_vars[i] for i in prediction_df.index])
        prob += lpSum([prediction_df['price'][i] * player_vars[i] for i in prediction_df.index]) <= 1000
        prob += lpSum([player_vars[i] for i in prediction_df.index if prediction_df['Pos'][i] == 'GK']) == 2
        prob += lpSum([player_vars[i] for i in prediction_df.index if prediction_df['Pos'][i] == 'DF']) == 5
        prob += lpSum([player_vars[i] for i in prediction_df.index if prediction_df['Pos'][i] == 'MF']) == 5
        prob += lpSum([player_vars[i] for i in prediction_df.index if prediction_df['Pos'][i] == 'FW']) == 3

        for squad in prediction_df['Squad'].unique():
            prob += lpSum([player_vars[i] for i in prediction_df.index if prediction_df['Squad'][i] == squad]) <= 3

        prob.solve()

        wildcard_players = []
        for v in prob.variables():
            if v.varValue == 1:
                player_index = int(v.name.split("_")[1])
                wildcard_players.append({
                    "Player": prediction_df['Player'][player_index],
                    "PredP1": prediction_df['PredP1'][player_index],
                    "PredP3": prediction_df['PredP3'][player_index],
                    "PredP5": prediction_df['PredP5'][player_index],
                    "Price": prediction_df['price'][player_index],
                    "Position": prediction_df['Pos'][player_index],
                    "player_id": prediction_df['player_id'][player_index],
                    "Squad": prediction_df['Squad'][player_index]
                })

        team_data = team_data_store.get(int(team_id))
        if not team_data:
            return "Error: Team data not found. Please try scraping again."

        current_team_df = pd.DataFrame(team_data['picks'])

        def compute_sums(df):
            predp1_sum = df['PredP1'].sum() if 'PredP1' in df else 0
            predp3_sum = df['PredP3'].sum() if 'PredP3' in df else 0
            predp5_sum = df['PredP5'].sum() if 'PredP5' in df else 0
            return predp1_sum, predp3_sum, predp5_sum

        wildcard_df = pd.DataFrame(wildcard_players)
        wildcard_p1_sum, wildcard_p3_sum, wildcard_p5_sum = compute_sums(wildcard_df)
        current_p1_sum, current_p3_sum, current_p5_sum = compute_sums(current_team_df)

        result1 = 0 if wildcard_p1_sum == 0 else 100 * current_p1_sum / wildcard_p1_sum
        result3 = 0 if wildcard_p3_sum == 0 else 100 * current_p3_sum / wildcard_p3_sum
        result5 = 0 if wildcard_p5_sum == 0 else 100 * current_p5_sum / wildcard_p5_sum

        results_df = pd.DataFrame({
            'Metric': ['PredP1', 'PredP3', 'PredP5'],
            'Percentage': [result1, result3, result5]
        })

        return render_template('optimized_team.html', 
                       optimized_team=wildcard_players, 
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
        print(f"Error in optimized_team: {str(e)}")
        return f"Error loading optimized team: {e}"

@app.route('/transfers', methods=['GET', 'POST'])
def transfers():
    team_id = request.args.get('team_id') or request.form.get('team_id')
    if not team_id:
        return "Error: team_id is required"

    try:
        picks_info, player_info = scrape_fpl_data(team_id)
        if not picks_info or not player_info:
            return "Error: Could not fetch the team data. Please try again later."

        position_mapping = {1: "GK", 2: "DF", 3: "MF", 4: "FW"}
        for pick in picks_info:
            pick['Pos'] = position_mapping.get(int(pick['Pos']), "Unknown")

        global prediction_df
        bank = player_info.get("bank", 0) or 0
        existing_players = [pick['player_id'] for pick in picks_info]
        squad_counts = prediction_df[prediction_df['player_id'].isin(existing_players)]['Squad'].value_counts().to_dict()

        if request.method in ['GET', 'POST']:
            num_transfers = int(request.args.get('num_transfers', request.form.get('num_transfers', 1)))
            best_combination = None
            best_total_point_diff = float('-inf')

            for transfer_combination in combinations(picks_info, num_transfers):
                remaining_budget = bank + sum(pick['price'] for pick in transfer_combination)
                temp_squad_counts = squad_counts.copy()
                temp_existing_players = existing_players.copy()
                total_point_diff = 0
                valid_combination = True
                transfers = []

                for player_to_sell in transfer_combination:
                    pos_condition = prediction_df['Pos'] == player_to_sell['Pos']
                    price_condition = prediction_df['price'] <= remaining_budget
                    not_in_team_condition = ~prediction_df['player_id'].isin(temp_existing_players)
                    squad_condition = prediction_df['Squad'].map(lambda s: temp_squad_counts.get(s, 0) < 3)
                    potential_buys = prediction_df[pos_condition & price_condition & not_in_team_condition & squad_condition]

                    if potential_buys.empty:
                        valid_combination = False
                        break

                    best_buy = potential_buys.loc[potential_buys['PredP5'].idxmax()]
                    point_diff = best_buy['PredP5'] - player_to_sell['PredP5']
                    total_point_diff += point_diff
                    remaining_budget -= (best_buy['price'] - player_to_sell['price'])
                    temp_existing_players.remove(player_to_sell['player_id'])
                    temp_existing_players.append(best_buy['player_id'])
                    temp_squad_counts[player_to_sell['Squad']] -= 1
                    temp_squad_counts[best_buy['Squad']] = temp_squad_counts.get(best_buy['Squad'], 0) + 1

                    transfers.append({
                        'sell': player_to_sell,
                        'buy': best_buy.to_dict(),
                        'point_diff': point_diff
                    })

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
        print(f"Error in transfers function: {e}")
        import traceback
        traceback.print_exc()
        return f"An error occurred during transfer processing: {e}"

@app.route('/player_suggestions/<int:team_id>', methods=['GET', 'POST'])
def player_suggestions(team_id):
    try:
        team_data = team_data_store.get(team_id)
        if not team_data:
            picks_info, player_info = scrape_fpl_data(team_id)
            if not picks_info or not player_info:
                return "Error: Could not fetch data for the team."
            team_data = {'picks': picks_info, 'player_data': player_info}
            team_data_store[team_id] = team_data

        picks_info = team_data['picks']
        player_info = team_data['player_data']
        
        global prediction_df
        if prediction_df is None or prediction_df.empty:
            return "Error: Prediction data not available."

        if request.method == 'GET':
            selected_player_id = request.args.get('selected_player_id')
            if selected_player_id:
                selected_player_id = int(selected_player_id)
            else:
                return "Error: No player selected"

            bank = player_info.get("bank", 0) or 0
            selected_player = next((p for p in picks_info if p['player_id'] == selected_player_id), None)
            if not selected_player:
                return "Error: Selected player not found in the team."

            pos_condition = prediction_df['Pos'] == selected_player['Pos']
            price_condition = prediction_df['price'] <= (bank + selected_player['price'])
            not_in_team_condition = ~prediction_df['player_id'].isin([p['player_id'] for p in picks_info])
            squad_counts = {p['Squad']: sum(1 for pick in picks_info if pick['Squad'] == p['Squad']) for p in picks_info}
            squad_condition = prediction_df['Squad'].map(lambda s: squad_counts.get(s, 0) < 3)
            
            potential_replacements = prediction_df[pos_condition & price_condition & not_in_team_condition & squad_condition]
            top_replacements = potential_replacements.sort_values('PredP5', ascending=False).head(3)
            top_replacements_list = top_replacements.to_dict(orient='records')
            
            return render_template(
                'player_suggestions.html',
                team_id=team_id,
                picks_info=picks_info,
                player_info=player_info,
                selected_player=selected_player,
                suggestions=top_replacements_list
            )

        return render_template(
            'player_suggestions.html',
            team_id=team_id,
            picks_info=picks_info,
            player_info=player_info
        )
    except Exception as e:
        print(f"Error in player_suggestions route: {e}")
        return f"An error occurred: {e}"

@app.route('/players', methods=['GET', 'POST'])
def players():
    team_id = request.args.get('team_id') or request.form.get('team_id')
    player_id = request.form.get('player_id')
    
    if not team_id:
        return "Error: team_id is required"

    try:
        team_data = team_data_store.get(int(team_id))
        if not team_data:
            picks_info, player_info = scrape_fpl_data(team_id)
            if not picks_info or not player_info:
                return "Error: Could not fetch the team data. Please try again later."
            team_data = {'picks': picks_info, 'player_data': player_info}
            team_data_store[int(team_id)] = team_data

        picks_info = team_data['picks']
        player_info = team_data['player_data']

        position_mapping = {1: "GK", 2: "DF", 3: "MF", 4: "FW"}
        for pick in picks_info:
            pick['Pos'] = position_mapping.get(int(pick['Pos']), "Unknown")

        if not player_id:
            selected_player = random.choice(picks_info)
            player_id = selected_player['player_id']
        else:
            player_id = int(player_id)
            selected_player = next((pick for pick in picks_info if pick['player_id'] == player_id), None)

        bank = player_info.get("bank", 0) or 0
        
        global prediction_df
        recommended_replacements = []
        if selected_player:
            pos_condition = prediction_df['Pos'] == selected_player['Pos']
            price_condition = prediction_df['price'] <= (bank + selected_player['price'])
            potential_replacements = prediction_df[pos_condition & price_condition]
            potential_replacements = potential_replacements[potential_replacements['player_id'] != selected_player['player_id']]
            recommended_replacements = potential_replacements.nlargest(3, 'PredP5').to_dict(orient='records')

        return render_template(
            'players.html',
            team_id=team_id,
            picks_info=picks_info,
            selected_player=selected_player,
            recommended_replacements=recommended_replacements
        )
    except Exception as e:
        print(f"Error in players route: {e}")
        return f"An error occurred: {e}"

@app.route('/test_api/<int:team_id>')
def test_api(team_id):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    picks_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/19/picks/"
    
    try:
        response = requests.get(picks_url, headers=headers, verify=True)
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': response.text[:500]
        }
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    # Initialize prediction data when the app starts
    init_prediction_data()
    app.run(debug=True, host='0.0.0.0', port=5000)
