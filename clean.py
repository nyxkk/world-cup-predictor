import pandas as pd

df = pd.read_csv('results.csv', parse_dates=['date'])
print(f"raw dataset: {len(df)} matches")

TOURNEYS = [
    'FIFA World Cup',
    'UEFA Euro',
    'Copa América',
    'African Cup of Nations',
    'AFC Asian Cup',
    'FIFA World Cup qualification',
]

df = df[df['tournament'].isin(TOURNEYS)] # filter to only important tournaments
df = df[df['date'].between('2010-01-01', '2026-06-27')] # filter to only matches after 2010 and before world cup 2026 knockout

df['results'] = df.apply(
    lambda row: 'team_a_wins' if row['home_score'] > row['away_score'] 
        else ('team_b_wins' if row['home_score'] < row['away_score'] else 'draw'),
    axis=1)

print(df.tail(100))

df.to_csv('clean.csv', index=False)
