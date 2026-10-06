import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

NUM_ROWS = 5000

time_of_day_opts = ['early_morning', 'morning_peak', 'mid_morning', 'afternoon', 'evening_peak', 'night', 'late_night']
day_of_week_opts = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
transit_line_opts = ['western', 'central', 'harbour']
weather_opts = ['clear', 'light_rain', 'heavy_monsoon']
crowding_opts = ['Minimal', 'Low', 'Moderate', 'High', 'Extreme']

data = {
    'time_of_day': np.random.choice(time_of_day_opts, NUM_ROWS),
    'day_of_week': np.random.choice(day_of_week_opts, NUM_ROWS),
    'transit_line': np.random.choice(transit_line_opts, NUM_ROWS),
    'weather': np.random.choice(weather_opts, NUM_ROWS)
}

df = pd.DataFrame(data)

# Generate targets based on some logic to make models learn
def calculate_delay_and_crowding(row):
    base_delay = {'western': 4, 'central': 5, 'harbour': 3}[row['transit_line']]
    time_mult = {'early_morning': 0.6, 'morning_peak': 2.2, 'mid_morning': 1.1,
                 'afternoon': 0.9, 'evening_peak': 2.4, 'night': 0.7, 'late_night': 0.4}[row['time_of_day']]
    day_mult = 1.3 if row['day_of_week'] not in ['saturday', 'sunday'] else 0.7
    weather_add = {'clear': 0, 'light_rain': 3, 'heavy_monsoon': 12}[row['weather']]
    
    raw_delay = base_delay * time_mult * day_mult + weather_add
    
    # Add some noise
    delay = max(0, int(round(raw_delay + np.random.normal(0, 2))))
    
    # Crowding
    crowd_score = time_mult * day_mult * 10
    if row['weather'] == 'heavy_monsoon':
        crowd_score += 5
    elif row['weather'] == 'light_rain':
        crowd_score += 2
        
    crowd_score += np.random.normal(0, 2)
    
    if crowd_score < 8:
        crowd_level = 'Minimal'
        crowd_pct = min(15, max(0, int(crowd_score * 2)))
    elif crowd_score < 14:
        crowd_level = 'Low'
        crowd_pct = min(35, max(16, int(crowd_score * 2.5)))
    elif crowd_score < 20:
        crowd_level = 'Moderate'
        crowd_pct = min(60, max(36, int(crowd_score * 3)))
    elif crowd_score < 28:
        crowd_level = 'High'
        crowd_pct = min(85, max(61, int(crowd_score * 3)))
    else:
        crowd_level = 'Extreme'
        crowd_pct = min(100, max(86, int(crowd_score * 3.5)))
        
    return delay, crowd_level, crowd_pct

targets = df.apply(calculate_delay_and_crowding, axis=1)
df['Delay_Minutes'] = [t[0] for t in targets]
df['Crowding_Level'] = [t[1] for t in targets]
df['Crowding_Pct'] = [t[2] for t in targets]

df.to_csv('transit_data.csv', index=False)
print("Data generated and saved to transit_data.csv")
