from statsbombpy import sb
import pandas as pd

# 1. Load a match — Spain vs England, Euro 2024 final
matches = sb.matches(competition_id=55, season_id=282)
final = matches[matches['home_team'] == 'Spain'].iloc[-1]
print(f"Match: {final['home_team']} vs {final['away_team']}")

# 2. Get events
events = sb.events(match_id=final['match_id'])
print(f"Total events: {len(events)}")
print(f"Event types: {events['type'].value_counts().head(10)}")

# 3. Look at a pass
passes = events[events['type'] == 'Pass']
print(f"Total passes: {len(passes)}")
print(passes[['player', 'location', 'pass_end_location', 'pass_outcome']].head())