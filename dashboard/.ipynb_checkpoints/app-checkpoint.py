import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from database import create_database, save_prediction, get_all_predictions
from predictor import load_model, predict_winner

st.set_page_config(
    page_title="🏏 IPL Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .winner-text {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data', 'processed')
    
    matches    = pd.read_csv(os.path.join(data_path, 'matches_clean.csv'))
    deliveries = pd.read_csv(os.path.join(data_path, 'deliveries_clean.csv'))
    matches['date'] = pd.to_datetime(matches['date'])
    matches['year'] = matches['date'].dt.year
    return matches, deliveries

@st.cache_resource
def load_ml_model():
    """Load ML model - cached so only loads once!
    cache_resource is used for ML models (heavy objects)
    cache_data is used for dataframes
    """
    return load_model()

matches, deliveries = load_data()
model, encoders     = load_ml_model()

all_teams  = sorted(matches['team1'].unique().tolist())
all_venues = sorted(matches['venue'].unique().tolist())


st.sidebar.markdown("# 🏏 IPL Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["📊 Analytics",
     "🤖 Win Predictor",
     "📋 Prediction History",
     "📈 Team Stats"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    "End-to-end IPL Data Science Project\n\n"
    "Built with Python, Pandas, Scikit-learn, SQLite & Streamlit"
)

# ══════════════════════════════════════════════════════
# PAGE 1: ANALYTICS
# ══════════════════════════════════════════════════════
if page == "📊 Analytics":
    
    # Main title
    st.markdown('<p class="main-header">🏏 IPL Analytics Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏟️ Total Matches", len(matches))
    with col2:
        st.metric("📅 Seasons", matches['year'].nunique())
    with col3:
        st.metric("🏏 Teams", matches['team1'].nunique())
    with col4:
        st.metric("⚡ Avg 1st Innings", "165.5 runs")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Most IPL Wins by Team")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        win_counts = matches['winner'].value_counts().head(10)
        sns.barplot(x=win_counts.values,
                   y=win_counts.index,
                   palette='viridis', ax=ax)
        ax.set_xlabel('Number of Wins')
        
        # Add value labels
        for i, v in enumerate(win_counts.values):
            ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🎯 Toss Decision Preference")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        toss_pref = matches['toss_decision'].value_counts()
        ax.pie(toss_pref.values,
               labels=toss_pref.index,
               autopct='%1.1f%%',
               colors=['#2196F3', '#FF9800'],
               startangle=90,
               explode=(0.05, 0.05))
        ax.set_title('Bat vs Field after winning toss')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏏 Top 10 Run Scorers (All Time)")
        
        top_bat = (deliveries.groupby('batsman')['batsman_runs']
                             .sum()
                             .sort_values(ascending=False)
                             .head(10)
                             .reset_index())
        top_bat.columns = ['Batsman', 'Total Runs']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=top_bat, x='Total Runs', 
                   y='Batsman', palette='rocket', ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🎳 Top 10 Wicket Takers (All Time)")
        
        wickets = deliveries[deliveries['is_wicket'] == 1]
        top_bowl = (wickets.groupby('bowler')['is_wicket']
                           .sum()
                           .sort_values(ascending=False)
                           .head(10)
                           .reset_index())
        top_bowl.columns = ['Bowler', 'Wickets']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=top_bowl, x='Wickets',
                   y='Bowler', palette='mako', ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("📈 Total Runs Per IPL Season")
    
    merged = deliveries.merge(
        matches[['id', 'year']], 
        left_on='match_id', right_on='id')
    runs_per_year = (merged.groupby('year')['total_runs']
                           .sum().reset_index())
    
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.lineplot(data=runs_per_year, x='year', y='total_runs',
                marker='o', color='#E91E63', linewidth=2.5, ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Runs')
    
    for _, row in runs_per_year.iterrows():
        ax.text(row['year'], row['total_runs'] + 500,
               str(int(row['total_runs'])), ha='center', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # ── Over Heatmap ──────────────────────────────────
    st.markdown("---")
    st.subheader("🔥 Average Runs Per Over — 1st Innings")
    
    over_runs = (deliveries[deliveries['inning'] == 1]
                 .groupby('over')['total_runs']
                 .mean().reset_index())
    over_runs.columns = ['Over', 'Avg Runs']
    heatmap_data = over_runs.set_index('Over')['Avg Runs'].values.reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(16, 2))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f',
               cmap='YlOrRd', xticklabels=range(1, 21),
               yticklabels=['Avg Runs'], ax=ax)
    ax.set_xlabel('Over Number')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════
# PAGE 2: WIN PREDICTOR
# ══════════════════════════════════════════════════════
elif page == "🤖 Win Predictor":
    
    st.markdown('<p class="main-header">🤖 IPL Match Win Predictor</p>',
                unsafe_allow_html=True)
    st.markdown("Enter match details below to predict the winner!")
    st.markdown("---")
    
    # ── Input Form ────────────────────────────────────
    # st.columns() → side by side inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏏 Match Details")
        
        team1 = st.selectbox(
            "Team Batting First (Team 1)",
            all_teams,
            index=all_teams.index('Chennai Super Kings') 
                  if 'Chennai Super Kings' in all_teams else 0
        )
        
        team2_options = [t for t in all_teams if t != team1]
        team2 = st.selectbox(
            "Team Chasing (Team 2)",
            team2_options,
            index=team2_options.index('Mumbai Indians')
                  if 'Mumbai Indians' in team2_options else 0
        )
        
        target = st.number_input(
            "Target Score (1st Innings Total)",
            min_value=50,
            max_value=300,
            value=178,
            step=1
        )
    
    with col2:
        st.subheader("🎯 Toss Details")
        
        toss_winner = st.selectbox(
            "Toss Winner",
            [team1, team2]
        )
        
        toss_decision = st.selectbox(
            "Toss Decision",
            ["field", "bat"]
        )
        
        venue = st.selectbox(
            "Venue",
            all_venues,
            index=all_venues.index('Wankhede Stadium')
                  if 'Wankhede Stadium' in all_venues else 0
        )
    
    st.markdown("---")
    
    if st.button("🔮 Predict Winner!", 
                 type="primary",
                 use_container_width=True):
        
        with st.spinner("Analyzing match situation..."):
            
            result = predict_winner(
                team1         = team1,
                team2         = team2,
                venue         = venue,
                target        = target,
                toss_winner   = toss_winner,
                toss_decision = toss_decision,
                model         = model,
                encoders      = encoders,
                save_to_db    = True
            )
        
        if 'error' in result:
            # st.error() → shows red error box
            st.error(f"❌ {result['error']}")
        
        else:
            # ── Show Results ──────────────────────────
            st.markdown("---")
            
            # st.success() → shows green success box
            st.markdown(
                f'<p class="winner-text">🏆 Predicted Winner: '
                f'{result["predicted_winner"]}</p>',
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            
            # Win probability metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"🏏 {team1}",
                    f"{result['team1_win_prob']}%"
                )
            with col2:
                st.metric(
                    "🎯 Confidence",
                    f"{result['confidence']}%"
                )
            with col3:
                st.metric(
                    f"🏃 {team2}",
                    f"{result['team2_win_prob']}%"
                )
            
            st.markdown("---")
            st.subheader("Win Probability Breakdown")
            
            fig, ax = plt.subplots(figsize=(10, 3))
            
            teams  = [team1, team2]
            probs  = [result['team1_win_prob'], 
                     result['team2_win_prob']]
            colors = ['#FF6B6B' if t != result['predicted_winner'] 
                     else '#2E8B57' for t in teams]
            
            bars = ax.barh(teams, probs, color=colors, height=0.4)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Win Probability (%)')
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, probs):
                ax.text(prob + 1, bar.get_y() + bar.get_height()/2,
                       f'{prob}%', va='center', fontweight='bold',
                       fontsize=12)
            
            # Green = predicted winner, Red = predicted loser
            ax.set_title('Green = Predicted Winner', 
                        fontsize=10, color='gray')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Confirmation message
            st.info("✅ Prediction saved to database!")


# ══════════════════════════════════════════════════════
# PAGE 3: PREDICTION HISTORY
# ══════════════════════════════════════════════════════
elif page == "📋 Prediction History":
    
    st.markdown('<p class="main-header">📋 Prediction History</p>',
                unsafe_allow_html=True)
    st.markdown("All predictions made so far, stored in SQLite database")
    st.markdown("---")
    
    # Load predictions from database
    conn = create_database()
    df   = get_all_predictions(conn)
    conn.close()
    
    if len(df) == 0:
        # st.info() → blue information box
        st.info("No predictions yet! Go to Win Predictor and make some predictions!")
    
    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(df))
        with col2:
            avg_conf = df['win_probability'].mean() * 100
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        with col3:
            most_predicted = df['predicted_winner'].value_counts().index[0]
            st.metric("Most Predicted Winner", most_predicted)
        
        st.markdown("---")
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['win_probability'] = (
            display_df['win_probability'] * 100
        ).round(1).astype(str) + '%'
        
        # Rename columns for readability
        display_df = display_df.rename(columns={
            'team1'            : 'Batting First',
            'team2'            : 'Chasing',
            'venue'            : 'Venue',
            'target'           : 'Target',
            'predicted_winner' : 'Predicted Winner',
            'win_probability'  : 'Confidence',
            'prediction_date'  : 'Date'
        })
        
        # Drop id column (not meaningful to display)
        display_df = display_df.drop(columns=['id'])
        
        st.dataframe(display_df, use_container_width=True)
        
        st.markdown("---")
        
        # Most predicted teams chart
        st.subheader("🏆 Most Predicted Winners")
        
        team_pred = df['predicted_winner'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=team_pred.values,
                   y=team_pred.index,
                   palette='viridis', ax=ax)
        ax.set_xlabel('Times Predicted as Winner')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════
# PAGE 4: TEAM STATS
# ══════════════════════════════════════════════════════
elif page == "📈 Team Stats":
    
    st.markdown('<p class="main-header">📈 Team Statistics</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Team selector
    selected_team = st.selectbox(
        "Select a Team to Analyze:",
        all_teams
    )
    
    team_matches = matches[
        (matches['team1'] == selected_team) | 
        (matches['team2'] == selected_team)
    ]
    
    team_wins = matches[matches['winner'] == selected_team]
    
    # ── Team Metrics ──────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    total_matches = len(team_matches)
    total_wins    = len(team_wins)
    win_rate      = (total_wins / total_matches * 100) if total_matches > 0 else 0
    
    with col1:
        st.metric("Matches Played", total_matches)
    with col2:
        st.metric("Matches Won", total_wins)
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        losses = total_matches - total_wins
        st.metric("Matches Lost", losses)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wins per season for selected team
        st.subheader(f"📅 {selected_team} Wins Per Season")
        
        wins_per_year = (team_wins.groupby('year')
                                  .size()
                                  .reset_index(name='wins'))
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=wins_per_year, x='year', 
                   y='wins', palette='Blues_d', ax=ax)
        ax.set_xlabel('Year')
        ax.set_ylabel('Wins')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Favorite venues
        st.subheader(f"🏟️ {selected_team} Favorite Winning Venues")
        
        venue_wins = (team_wins['venue']
                     .value_counts()
                     .head(5)
                     .reset_index())
        venue_wins.columns = ['Venue', 'Wins']
        
        # Clean venue names (some are very long)
        venue_wins['Venue'] = venue_wins['Venue'].str[:25]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=venue_wins, x='Wins',
                   y='Venue', palette='Greens_d', ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # ── Head to Head ──────────────────────────────────
    st.markdown("---")
    st.subheader(f"⚔️ {selected_team} vs Other Teams (Head to Head Wins)")
    
    # Count wins against each opponent
    h2h_data = []
    
    for opponent in all_teams:
        if opponent == selected_team:
            continue
        
        # Matches between these two teams
        h2h_matches = matches[
            ((matches['team1'] == selected_team) & 
             (matches['team2'] == opponent)) |
            ((matches['team1'] == opponent) & 
             (matches['team2'] == selected_team))
        ]
        
        # Wins for selected team against this opponent
        h2h_wins = h2h_matches[
            h2h_matches['winner'] == selected_team
        ]
        
        if len(h2h_matches) > 0:
            h2h_data.append({
                'Opponent'     : opponent,
                'Matches'      : len(h2h_matches),
                'Wins'         : len(h2h_wins),
                'Win Rate'     : f"{len(h2h_wins)/len(h2h_matches)*100:.0f}%"
            })
    
    h2h_df = pd.DataFrame(h2h_data).sort_values('Wins', ascending=False)
    st.dataframe(h2h_df, use_container_width=True)