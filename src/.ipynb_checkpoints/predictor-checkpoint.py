import pickle
import numpy as np
import pandas as pd
import sqlite3
import os

from database import create_database, save_prediction, get_all_predictions

def load_model():
    src_folder = os.path.dirname(os.path.abspath(__file__))
    
    model_path    = os.path.join(src_folder, 'model.pkl')
    encoders_path = os.path.join(src_folder, 'encoders.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "model.pkl not found! Please run 04_ml_model.ipynb first!")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    print("Model loaded successfully!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Teams known to encoder: {len(encoders['team'].classes_)}")
    
    return model, encoders


def predict_winner(team1, team2, venue, target, 
                   toss_winner, toss_decision,
                   model, encoders, save_to_db=True):
    
    le_team  = encoders['team']
    le_venue = encoders['venue']
    
    try:
        team1_enc       = le_team.transform([team1])[0]
        team2_enc       = le_team.transform([team2])[0]
        toss_winner_enc = le_team.transform([toss_winner])[0]
        venue_enc       = le_venue.transform([venue])[0]
        toss_dec_enc    = 1 if toss_decision == 'bat' else 0
        
    except ValueError as e:
        return {
            'error': f"Unknown team or venue: {e}. "
                     f"Please use teams/venues from the training data!"
        }
    
    input_features = np.array([[
        team1_enc,
        team2_enc,
        venue_enc,
        toss_winner_enc,
        toss_dec_enc,
        target
    ]])
    
    print(f"\nEncoded input: {input_features}")
    
    prediction   = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]
    
    team1_win_prob = probabilities[0]
    team2_win_prob = probabilities[1]
    
    predicted_winner = team2 if prediction == 1 else team1
    winner_probability = team2_win_prob if prediction == 1 else team1_win_prob
    
    if save_to_db:
        conn = create_database()
        save_prediction(
            conn,
            team1            = team1,
            team2            = team2,
            venue            = venue,
            target           = target,
            predicted_winner = predicted_winner,
            win_probability  = winner_probability
        )
        conn.close()
    
    
    result = {
        'team1'            : team1,
        'team2'            : team2,
        'venue'            : venue,
        'target'           : target,
        'predicted_winner' : predicted_winner,
        'team1_win_prob'   : round(team1_win_prob * 100, 1),
        'team2_win_prob'   : round(team2_win_prob * 100, 1),
        'confidence'       : round(winner_probability * 100, 1)
    }
    
    return result


def display_prediction(result):
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n{'='*50}")
    print(f"  🏏 IPL MATCH PREDICTION")
    print(f"{'='*50}")
    print(f"  Match  : {result['team1']} vs {result['team2']}")
    print(f"  Venue  : {result['venue']}")
    print(f"  Target : {result['target']} runs")
    print(f"{'='*50}")
    print(f"  🏆 Predicted Winner: {result['predicted_winner']}")
    print(f"{'='*50}")
    print(f"  Win Probabilities:")
    print(f"  {result['team1']:<30} {result['team1_win_prob']}%")
    print(f"  {result['team2']:<30} {result['team2_win_prob']}%")
    print(f"{'='*50}")
    
    team1_bar = '█' * int(result['team1_win_prob'] / 5)
    team2_bar = '█' * int(result['team2_win_prob'] / 5)
    print(f"\n  {result['team1'][:15]:<15} {team1_bar}")
    print(f"  {result['team2'][:15]:<15} {team2_bar}")


if __name__ == '__main__':
    
    model, encoders = load_model()
    
        
    result = predict_winner(
        team1         = 'Chennai Super Kings',
        team2         = 'Mumbai Indians',
        venue         = 'Wankhede Stadium',
        target        = 178,
        toss_winner   = 'Mumbai Indians',
        toss_decision = 'field',
        model         = model,
        encoders      = encoders,
        save_to_db    = True
    )
    display_prediction(result)
    
    print("\n📋 Checking database...")
    conn = create_database()
    df = get_all_predictions(conn)
    print(f"Total predictions in database: {len(df)}")
    print(df[['team1', 'team2', 
              'predicted_winner', 
              'win_probability']].tail(3))
    conn.close()
    
    print("\n\n📋 Second Test Match:")
    print("RCB scored 156 at Chinnaswamy")
    print("KKR needs to chase 157")
    
    result2 = predict_winner(
        team1         = 'Royal Challengers Bengaluru',
        team2         = 'Kolkata Knight Riders',
        venue         = 'M Chinnaswamy Stadium',
        target        = 156,
        toss_winner   = 'Royal Challengers Bengaluru',
        toss_decision = 'bat',
        model         = model,
        encoders      = encoders,
        save_to_db    = True
    )
    
    display_prediction(result2)