import sqlite3
import pandas as pd
from datetime import datetime

def create_database():
    
    conn = sqlite3.connect('ipl_predictions.db')
    
    cursor = conn.cursor()
        
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            team1            TEXT,
            team2            TEXT,
            venue            TEXT,
            target           INTEGER,
            predicted_winner TEXT,
            win_probability  REAL,
            prediction_date  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    
    
    return conn


def save_prediction(conn, team1, team2, venue, 
                    target, predicted_winner, win_probability):
    
    cursor = conn.cursor()
        
    cursor.execute('''
        INSERT INTO match_predictions 
        (team1, team2, venue, target, predicted_winner, win_probability)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (team1, team2, venue, target, predicted_winner, win_probability))
    
    conn.commit()
    

def get_all_predictions(conn):
    
    df = pd.read_sql('''
        SELECT * FROM match_predictions 
        ORDER BY prediction_date DESC
    ''', conn)
    
    return df


def get_prediction_stats(conn):
    
    
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM match_predictions")
    total = cursor.fetchone()[0]  # fetchone() gets the single result
    cursor.execute('''
        SELECT predicted_winner, COUNT(*) as times_predicted
        FROM match_predictions
        GROUP BY predicted_winner
        ORDER BY times_predicted DESC
    ''')
    team_predictions = cursor.fetchall()
    cursor.execute('''
        SELECT AVG(win_probability) as avg_confidence
        FROM match_predictions
    ''')
    avg_conf = cursor.fetchone()[0]
    
    print(f"\n📊 Prediction Statistics:")
    print(f"Total predictions made : {total}")
    print(f"Average confidence     : {avg_conf*100:.1f}%" if avg_conf else "No predictions yet")
    print(f"\nMost predicted winners:")
    for team, count in team_predictions:
        print(f"  {team:<30} → {count} times")
    
    return total, team_predictions


if __name__ == '__main__':
    
    conn = create_database()
    

    df = get_all_predictions(conn)
    print(df.to_string())
    
    get_prediction_stats(conn)
    
    conn.close()
    print("\n✅ Database connection closed!")