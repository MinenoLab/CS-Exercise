import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_cafe_customers_data(start_date='2020-04-01', end_date='2024-04-02', output_dir='../data/raw'):
    """
    Generate complex dummy cafe customer data for advanced time series analysis.

    This function creates a challenging time series dataset with irregular patterns,
    seasonal effects, random events, weather impacts, and long-term trends.
    Designed to test advanced forecasting models and feature engineering skills.

    Args:
        start_date (str): The start date for the data generation (YYYY-MM-DD).
        end_date (str): The end date for the data generation (YYYY-MM-DD).
        output_dir (str): The directory to save the generated CSV file.

    Returns:
        None. The function saves the data as 'cafe_customers.csv'.
    """
    # 1. Create a timestamp range for every 30 minutes
    timestamps = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='30T'))
    df = pd.DataFrame(timestamps, columns=['Timestamp'])
    
    # 2. Initialize random seed for reproducible randomness
    np.random.seed(42)
    
    # 3. Generate special events (random dates with high impact)
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    special_events = np.random.choice(total_days, size=int(total_days * 0.05), replace=False)
    special_event_dates = [pd.to_datetime(start_date) + timedelta(days=int(d)) for d in special_events]
    
    # 4. Generate weather effects (0: bad weather, 1: normal, 2: great weather)
    weather_effects = np.random.choice([0, 1, 2], size=total_days, p=[0.2, 0.6, 0.2])

    # 5. Define complex customer patterns with multiple influencing factors
    customers = []
    for i, ts in enumerate(df['Timestamp']):
        hour = ts.hour
        minute = ts.minute
        day_of_week = ts.dayofweek  # Monday=0, Sunday=6
        day_of_year = ts.dayofyear
        month = ts.month
        date_only = ts.date()
        
        # Calculate days since start for trend analysis
        days_since_start = (ts - pd.to_datetime(start_date)).days
        
        # Base customer count with higher randomness
        base_customers = np.random.randint(0, 5)
        
        # 1. SEASONAL EFFECTS (stronger variations)
        seasonal_multiplier = 1.0
        if month in [12, 1, 2]:  # Winter - less customers
            seasonal_multiplier *= 0.7 + 0.3 * np.random.random()
        elif month in [6, 7, 8]:  # Summer - more customers
            seasonal_multiplier *= 1.3 + 0.4 * np.random.random()
        elif month in [3, 4, 5]:  # Spring - variable
            seasonal_multiplier *= 0.9 + 0.6 * np.random.random()
        else:  # Fall - stable but declining
            seasonal_multiplier *= 0.8 + 0.4 * np.random.random()
        
        # 2. LONG-TERM TREND (business growth/decline with noise)
        trend_factor = 1.0 + (days_since_start / total_days) * 0.5  # 50% growth over period
        # Add trend noise
        trend_factor *= (0.8 + 0.4 * np.random.random())
        
        # 3. WEATHER EFFECTS
        weather_today = weather_effects[min(days_since_start, len(weather_effects)-1)]
        weather_multiplier = {0: 0.5 + 0.3 * np.random.random(),  # Bad weather
                             1: 0.9 + 0.2 * np.random.random(),  # Normal
                             2: 1.2 + 0.5 * np.random.random()}[weather_today]  # Great weather
        
        # 4. SPECIAL EVENTS EFFECT
        event_multiplier = 1.0
        if date_only in [event.date() for event in special_event_dates]:
            event_multiplier = 1.5 + 1.0 * np.random.random()  # 50-150% boost
        
        # 5. COMPLEX DAILY PATTERNS WITH IRREGULARITIES
        if day_of_week < 5:  # Monday to Friday
            # Monday blues
            if day_of_week == 0:
                base_customers *= (0.7 + 0.4 * np.random.random())
            # Friday excitement
            elif day_of_week == 4:
                base_customers *= (1.1 + 0.3 * np.random.random())
            
            # Time-based patterns with more variation
            if 6 <= hour <= 8:  # Early morning
                base_customers += np.random.poisson(3) + np.random.randint(0, 8)
            elif hour == 9:  # Morning rush with irregularity
                rush_intensity = np.random.choice([0.5, 1.0, 1.5, 2.0], p=[0.2, 0.4, 0.3, 0.1])
                base_customers += int(np.random.poisson(12) * rush_intensity)
            elif 10 <= hour <= 11:  # Mid morning
                base_customers += np.random.poisson(5) + np.random.randint(-2, 5)
            elif hour == 12:  # Lunch peak with high variation
                lunch_factor = np.random.choice([0.3, 0.7, 1.0, 1.5, 2.2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                base_customers += int(np.random.poisson(20) * lunch_factor)
            elif hour == 13:  # Post-lunch
                base_customers += np.random.poisson(15) + np.random.randint(-5, 8)
            elif 14 <= hour <= 16:  # Afternoon with randomness
                base_customers += np.random.poisson(8) + np.random.randint(-3, 6)
            elif 17 <= hour <= 19:  # Evening varying patterns
                evening_pattern = np.random.choice([0.4, 0.8, 1.2], p=[0.3, 0.5, 0.2])
                base_customers += int(np.random.poisson(10) * evening_pattern)
            elif 20 <= hour <= 21:  # Late evening
                base_customers += np.random.poisson(4) + np.random.randint(-2, 4)
            else:  # Other times with occasional spikes
                if np.random.random() < 0.05:  # 5% chance of random spike
                    base_customers += np.random.randint(10, 25)
                else:
                    base_customers += np.random.poisson(2)
        
        else:  # Weekend patterns (more irregular)
            # Saturday vs Sunday differences
            weekend_multiplier = 1.3 if day_of_week == 5 else 0.9
            
            if 8 <= hour <= 10:  # Weekend brunch rush
                brunch_intensity = np.random.choice([0.5, 1.0, 1.8, 2.5], p=[0.2, 0.3, 0.3, 0.2])
                base_customers += int(np.random.poisson(12) * brunch_intensity * weekend_multiplier)
            elif 11 <= hour <= 15:  # Weekend peak with high variation
                peak_variation = np.random.choice([0.6, 1.0, 1.4, 2.0], p=[0.15, 0.35, 0.35, 0.15])
                base_customers += int(np.random.poisson(18) * peak_variation * weekend_multiplier)
            elif 16 <= hour <= 19:  # Weekend evening
                base_customers += int(np.random.poisson(12) * weekend_multiplier)
            elif 20 <= hour <= 21:  # Weekend night
                base_customers += int(np.random.poisson(8) * weekend_multiplier)
            else:
                base_customers += np.random.poisson(3)
        
        # 6. MINUTE-LEVEL VARIATIONS (within the hour)
        minute_factor = 1.0
        if minute < 15:  # Beginning of hour
            minute_factor = 0.8 + 0.4 * np.random.random()
        elif minute >= 45:  # End of hour
            minute_factor = 0.9 + 0.3 * np.random.random()
        else:  # Middle of hour
            minute_factor = 1.0 + 0.2 * np.random.random()
        
        # 7. CLOSE TIME EFFECTS
        if hour < 6 or hour >= 23:
            base_customers = max(0, np.random.poisson(0.5))
        elif hour == 6 or hour == 22:  # Opening/closing hours
            base_customers = max(0, int(base_customers * 0.3 * np.random.random()))
        
        # 8. APPLY ALL MULTIPLIERS WITH ADDITIONAL NOISE
        final_customers = base_customers
        final_customers *= seasonal_multiplier
        final_customers *= trend_factor
        final_customers *= weather_multiplier
        final_customers *= event_multiplier
        final_customers *= minute_factor
        
        # 9. ADD FINAL LAYER OF UNPREDICTABILITY
        # Random shock events (rare but impactful)
        if np.random.random() < 0.002:  # 0.2% chance
            shock_multiplier = np.random.choice([0.1, 0.2, 3.0, 5.0], p=[0.4, 0.3, 0.2, 0.1])
            final_customers *= shock_multiplier
        
        # Poisson noise for final realism
        final_customers = max(0, np.random.poisson(max(0.1, final_customers)))
        
        # Occasional zero customers (closed unexpectedly, etc.)
        if np.random.random() < 0.01:  # 1% chance
            final_customers = 0
        
        customers.append(int(final_customers))

    df['Customers'] = customers

    # 3. Save the dataframe to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cafe_customers.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully generated '{output_path}' with {len(df)} records.")

if __name__ == '__main__':
    # You need to adjust the output directory based on your script location.
    # Assuming this script is in 01_tutorial/src/
    generate_cafe_customers_data(output_dir='./raw')

