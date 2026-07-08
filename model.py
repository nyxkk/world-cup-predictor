"""
model.py — train a logistic regression model to predict World Cup knockout stage outcomes.
 
Workflow:
    1. Load historical knockout matches (2010, 2014, 2018, 2022)
    2. Build feature rows using build_features() from features.py
    3. Train logistic regression with 80/20 train/test split
    4. Evaluate and print metrics
    5. Save trained model to model.pkl for use in the Streamlit app
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split

from features import build_features

# group stage match counts per World Cup year — used to identify knockout matches
GROUP_STAGE_COUNTS = {2010: 48, 2014: 48, 2018: 48, 2022: 48}

FEATURE_COLS = [
    'curr_wc_goal_diff_diff',
    'curr_wc_goals_scored_per_game_diff',
    'curr_wc_goals_conceded_per_game_diff',
    'curr_wc_win_pct_diff',
    'curr_wc_quali_goal_diff_diff',
    'curr_wc_quali_win_pct_diff',
    'recent_4_years_goal_diff_diff',
    'recent_4_years_win_pct_diff',
    'head_to_head_win_rate'
]


if __name__ == "__main__":

    # ── load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv('clean.csv', parse_dates=['date'])
    df['goal_diff'] = df['home_score'] - df['away_score']

    # shootout results for knockout matches that ended in a draw
    shootout_df = pd.read_csv('shootouts.csv', parse_dates=['date'])

    # ── build knockout data ───────────────────────────────────────────────────
    # skip 2026 — no knockout matches yet, only group stage in dataset
    world_cup_knockout_data_temp = []
    for year, num_of_matches in GROUP_STAGE_COUNTS.items():
        wc_by_year = (
            df[(df['tournament'] == 'FIFA World Cup') & (df['date'].dt.year == year)]
            .sort_values('date')
        )
        world_cup_knockout_data_temp.append(wc_by_year.iloc[num_of_matches:])

    world_cup_knockout_data = pd.concat(world_cup_knockout_data_temp, ignore_index=True)

    # ── build training data ───────────────────────────────────────────────────
    # for each knockout match, use matches before that to build features using build_features 
    # use shootout_df to handle penalty shootout results for matches that end in a draw
    training_data = []
    for _, match in world_cup_knockout_data.iterrows():
        try:
            features_for_match = build_features(df, match['home_team'], match['away_team'], match['date'])

            # resolve penalty shootout winner
            if match['results'] == 'draw':
                pen_shootout_match = (
                    shootout_df[
                        (shootout_df['date'] == match['date']) &
                        (shootout_df['home_team'] == match['home_team']) &
                        (shootout_df['away_team'] == match['away_team'])
                    ]
                )

                winning_country = pen_shootout_match['winner'].iloc[0]
                features_for_match['results'] = 'team_a_wins' if winning_country == match['home_team'] else 'team_b_wins'
            else:
                features_for_match['results'] = match['results']

            training_data.append(features_for_match)

        except Exception as e:
            print(f"Skipping match between {match['home_team']} and {match['away_team']} on {match['date']} due to error: {e}")

    training_df = pd.DataFrame(training_data)
    print(f"Training rows: {len(training_df)}")

    # ── split features and label ───────────────────────────────────────────────
    X = training_df[FEATURE_COLS]
    y = training_df['results']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ── train ──────────────────────────────────────────────────────────────────
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print(f"Classes: {model.classes_}")

    # ── feature coefficients ───────────────────────────────────────────────────
    # with 2 classes, model stores 1 row of coefficients
    # negate to get team_a_wins perspective, stack for readability
    coef_df = pd.DataFrame(
        np.vstack([-model.coef_[0], model.coef_[0]]),
        columns=FEATURE_COLS,
        index=model.classes_
    )
    print("\nFeature coefficients:")
    print(coef_df.round(3))

    # ── evaluate ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.1%}")
    print(f"log-loss: {log_loss(y_test, y_proba):.3f}")
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # ── save model ─────────────────────────────────────────────────────────────
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # ── example usage — predict one known 2022 matchup ─────────────────────────
    team_a = 'France'
    team_b = 'Poland'
    cut_off_date = pd.Timestamp('2022-12-02')

    features = build_features(df, team_a, team_b, cut_off_date)
    X_new = pd.DataFrame([features])[FEATURE_COLS]
    probs = model.predict_proba(X_new)[0]

    print(f"\nTest prediction: (team_a) {team_a} vs {team_b} (team_b) - 2022 Round of 16")
    for cls, prob in zip(model.classes_, probs):
        print(f"  {cls}: {prob:.1%}")
    print("Actual result: France 3 - 1 Poland")
