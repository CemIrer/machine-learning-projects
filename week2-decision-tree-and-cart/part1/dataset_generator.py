"""
Housing Dataset Generator for Bornova District
Creates realistic synthetic data for Decision Tree analysis
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Bornova neighborhoods (real neighborhoods in Bornova, Izmir)
neighborhoods = [
    'Erzene',
    'Kazimdirik',
    'Evka-3',
    'Yesilova',
    'Altas',
    'Kizilcikli'
]

# Generate 100 houses to ensure we have 80+ samples
n_houses = 100

# Base prices for different neighborhoods (in Turkish Lira)
neighborhood_base_prices = {
    'Erzene': 3_500_000,
    'Kazimdirik': 3_200_000,
    'Evka-3': 2_800_000,
    'Yesilova': 3_000_000,
    'Altas': 2_600_000,
    'Kizilcikli': 2_400_000
}

data = []

for i in range(n_houses):
    neighborhood = np.random.choice(neighborhoods)

    # Age: 0-30 years (realistic for completed buildings)
    age = np.random.randint(0, 31)

    # Net square meters: 90-160 m² (realistic for 3+1 apartments)
    net_sqm = np.random.randint(90, 161)

    # Calculate price based on neighborhood, age, and size
    base_price = neighborhood_base_prices[neighborhood]

    # Price adjustments
    # Newer buildings are more expensive
    age_factor = 1 - (age * 0.015)
    # Larger apartments are more expensive (per sqm)
    size_factor = net_sqm / 120  # 120 is average
    # Add some randomness
    random_factor = np.random.uniform(0.85, 1.15)

    price = int(base_price * age_factor * size_factor * random_factor)
    # Round to nearest 50,000
    price = round(price / 50_000) * 50_000

    data.append({
        'Price': price,
        'Neighborhood': neighborhood,
        'Age': age,
        'NetSquareMeters': net_sqm
    })

df = pd.DataFrame(data)

# Sort by price for better readability
df = df.sort_values('Price').reset_index(drop=True)

# Save to Excel
df.to_excel('bornova_housing_dataset.xlsx', index=False)

print("Dataset created successfully!")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset statistics:")
print(df.describe())
print(f"\nNeighborhood distribution:")
print(df['Neighborhood'].value_counts())
print(f"\nDataset saved to: bornova_housing_dataset.xlsx")
