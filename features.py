import pandas as pd
import numpy as np

MIN_H2H_MATCHES = 3


def build_features(df, team_a, team_b, cut_off_date):
    """
    Build a feature dictionary for a matchup between team_a and team_b in the World Cup knockout stage happening after the cut_off_date.

    Parameters
    ----------
    df           : cleaned DataFrame from clean.csv
    team_a       : first team name exactly as in dataset
    team_b       : second team name exactly as in dataset
    cut_off_date : the end of the group stage of the World Cup to analyse. pd.Timestamp — only use data before this date (prevents leakage)

    Returns
    -------
    dict of 9 features, all differentials (team_a minus team_b)
    """
    valid_data = df[df['date'] <= cut_off_date].copy()

    # catch input errors for invalid team name or cut_off_date type
    all_teams = set(valid_data['home_team']).union(set(valid_data['away_team']))
    if team_a not in all_teams:
        raise ValueError(f"team_a '{team_a}' not found in dataset. Check spelling.")
    if team_b not in all_teams:
        raise ValueError(f"team_b '{team_b}' not found in dataset. Check spelling.")
    if not isinstance(cut_off_date, pd.Timestamp):
        raise TypeError(f"cut_off_date must be a pd.Timestamp, got {type(cut_off_date)}")

    # helper function - get the team's matches
    def team_matches(data, team):
        return data[(data['home_team'] == team) | (data['away_team'] == team)]

    # current world cup group stage data
    latest_wc_year = valid_data[valid_data['tournament'] == 'FIFA World Cup']['date'].dt.year.max()
    curr_wc_data = valid_data[(valid_data['tournament'] == 'FIFA World Cup') & (valid_data['date'].dt.year == latest_wc_year)].copy()

    # feature 1 - current wc goal difference
    def curr_wc_goal_diff(team):
        team_curr_wc_matches = team_matches(curr_wc_data, team)
        if len(team_curr_wc_matches) == 0:
            raise ValueError(f"No current WC group stage matches found for '{team}'. Check team name and cut_off_date.")
        pos_neg_corrected_goal_diff = team_curr_wc_matches.apply(lambda row: row['goal_diff'] if row['home_team'] == team else -row['goal_diff'], axis=1)
        return pos_neg_corrected_goal_diff.mean()

    # feature 2 - current wc goals scored
    def curr_wc_goals_scored_per_game(team):
        team_curr_wc_matches = team_matches(curr_wc_data, team)
        if len(team_curr_wc_matches) == 0:
            raise ValueError(f"No current WC group stage matches found for '{team}'. Check team name and cut_off_date.")
        goals_scored = team_curr_wc_matches[(team_curr_wc_matches['home_team'] == team)]['home_score'].sum() + team_curr_wc_matches[(team_curr_wc_matches['away_team'] == team)]['away_score'].sum()
        return goals_scored / len(team_curr_wc_matches)
    
    # feature 3 - current wc goals conceded
    def curr_wc_goals_conceded_per_game(team):
        team_curr_wc_matches = team_matches(curr_wc_data, team)
        if len(team_curr_wc_matches) == 0:
            raise ValueError(f"No current WC group stage matches found for '{team}'. Check team name and cut_off_date.")
        goals_conceded = team_curr_wc_matches[(team_curr_wc_matches['home_team'] == team)]['away_score'].sum() + team_curr_wc_matches[(team_curr_wc_matches['away_team'] == team)]['home_score'].sum()
        return goals_conceded / len(team_curr_wc_matches)

    # feature 4 - current wc win %
    def curr_wc_win_pct(team):
        team_curr_wc_matches = team_matches(curr_wc_data, team)
        if len(team_curr_wc_matches) == 0:
            raise ValueError(f"No current WC group stage matches found for '{team}'. Check team name and cut_off_date.")
        number_of_wins = len(team_curr_wc_matches[(team_curr_wc_matches['home_team'] == team) & (team_curr_wc_matches['results'] == 'team_a_wins')]) + len(team_curr_wc_matches[(team_curr_wc_matches['away_team'] == team) & (team_curr_wc_matches['results'] == 'team_b_wins')])
        total_matches = len(team_curr_wc_matches)
        return number_of_wins / total_matches

    # current world cup qualification data
    curr_wc_quali_years = latest_wc_year - 4
    curr_wc_quali_data = valid_data[(valid_data['tournament'] == 'FIFA World Cup qualification') & (valid_data['date'].dt.year >= curr_wc_quali_years)]
    
    # feature 5 - current wc quali goal diff
    def curr_wc_quali_goal_diff(team):
        team_wc_quali_matches = team_matches(curr_wc_quali_data, team)
        if len(team_wc_quali_matches) == 0:
            return 0.0
        pos_neg_corrected_goal_diff = team_wc_quali_matches.apply(lambda row: row['goal_diff'] if row['home_team'] == team else -row['goal_diff'], axis=1)
        return pos_neg_corrected_goal_diff.mean()
    
    # feature 6 - current wc quali win %
    def curr_wc_quali_win_pct(team):
        team_wc_quali_matches = team_matches(curr_wc_quali_data, team)
        number_of_wins = len(team_wc_quali_matches[(team_wc_quali_matches['home_team'] == team) & (team_wc_quali_matches['results'] == 'team_a_wins')]) + len(team_wc_quali_matches[(team_wc_quali_matches['away_team'] == team) & (team_wc_quali_matches['results'] == 'team_b_wins')])
        total_matches = len(team_wc_quali_matches)
        return number_of_wins / total_matches if total_matches > 0 else 0.5
    
    # recent 4 years data excluding the current world cup group stage
    recent_4_years_data = valid_data[(valid_data['date'] >= cut_off_date - pd.DateOffset(years=4)) & ~((valid_data['tournament'] == 'FIFA World Cup') & (valid_data['date'].dt.year == latest_wc_year))]

    # feature 7 - recent 4 years goal diff
    def recent_4_years_goal_diff(team):
        team_recent_4_years_data = team_matches(recent_4_years_data, team)
        if len(team_recent_4_years_data) == 0:
            return 0.0
        pos_neg_corrected_goal_diff = team_recent_4_years_data.apply(lambda row: row['goal_diff'] if row['home_team'] == team else -row['goal_diff'], axis=1)
        return pos_neg_corrected_goal_diff.mean()

    # feature 8 - recent 4 years win %
    def recent_4_years_win_pct(team):
        team_recent_4_years_data = team_matches(recent_4_years_data, team)
        number_of_wins = len(team_recent_4_years_data[(team_recent_4_years_data['home_team'] == team) & (team_recent_4_years_data['results'] == 'team_a_wins')]) + len(team_recent_4_years_data[(team_recent_4_years_data['away_team'] == team) & (team_recent_4_years_data['results'] == 'team_b_wins')])
        total_matches = len(team_recent_4_years_data)
        return number_of_wins / total_matches if total_matches > 0 else 0.5

    # feature 9 - head to head win % in data
    def head_to_head_win_rate(team_a, team_b):
        head_to_head_matches = valid_data[((valid_data['home_team'] == team_a) & (valid_data['away_team'] == team_b)) | ((valid_data['home_team'] == team_b) & (valid_data['away_team'] == team_a))]
        number_of_wins = len(head_to_head_matches[(head_to_head_matches['home_team'] == team_a) & (head_to_head_matches['results'] == 'team_a_wins')]) + len(head_to_head_matches[(head_to_head_matches['away_team'] == team_a) & (head_to_head_matches['results'] == 'team_b_wins')])
        total_matches = len(head_to_head_matches)

        if total_matches < MIN_H2H_MATCHES:
            return 0.5
        else:
            return number_of_wins / total_matches if total_matches > 0 else 0.5

    features = {
        'team_a': team_a,
        'team_b': team_b,
        'curr_wc_goal_diff_diff': curr_wc_goal_diff(team_a) - curr_wc_goal_diff(team_b),
        'curr_wc_goals_scored_per_game_diff': curr_wc_goals_scored_per_game(team_a) - curr_wc_goals_scored_per_game(team_b),
        'curr_wc_goals_conceded_per_game_diff': curr_wc_goals_conceded_per_game(team_a) - curr_wc_goals_conceded_per_game(team_b),
        'curr_wc_win_pct_diff': curr_wc_win_pct(team_a) - curr_wc_win_pct(team_b),
        'curr_wc_quali_goal_diff_diff': curr_wc_quali_goal_diff(team_a) - curr_wc_quali_goal_diff(team_b),
        'curr_wc_quali_win_pct_diff': curr_wc_quali_win_pct(team_a) - curr_wc_quali_win_pct(team_b),
        'recent_4_years_goal_diff_diff': recent_4_years_goal_diff(team_a) - recent_4_years_goal_diff(team_b),
        'recent_4_years_win_pct_diff': recent_4_years_win_pct(team_a) - recent_4_years_win_pct(team_b),
        'head_to_head_win_rate': head_to_head_win_rate(team_a, team_b)
    }

    return features


if __name__ == "__main__":
    df = pd.read_csv('clean.csv', parse_dates=['date'])
    df['goal_diff'] = df['home_score'] - df['away_score']

    team_a = 'Argentina'
    team_b = 'Australia'
    cut_off_date = pd.Timestamp('2022-12-02')  # Example cut-off date for the 2022 World Cup
    features = build_features(df, team_a, team_b, cut_off_date)
    print(features)