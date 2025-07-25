#!/usr/bin/env python3
import pandas as pd
import sys

try:
    # Try to read the parquet file
    df = pd.read_parquet('data/acs_metadata.parquet')
    
    # Filter for race-related columns to see what's available
    race_columns = df[df['description'].str.contains('race|black|white|asian|hispanic|latino', case=False, na=False)]
    print(f'Found {len(race_columns)} race-related columns out of {len(df)} total items')
    
    print('\n=== GENERAL POPULATION COLUMNS (ending with _pop) ===')
    general_pop = race_columns[race_columns['column_name'].str.contains('_pop$', regex=True)]
    for idx, row in general_pop[['column_name', 'description']].head(15).iterrows():
        print(f'- {row["column_name"]}: {row["description"]}')
    
    print(f'\n=== SPECIFIC DEMOGRAPHIC COLUMNS (age/gender breakdowns) ===')
    specific = race_columns[~race_columns['column_name'].str.contains('_pop$', regex=True)]
    for idx, row in specific[['column_name', 'description']].head(15).iterrows():
        print(f'- {row["column_name"]}: {row["description"]}')
        
    print(f'\n=== SAMPLE OF ALL COLUMN TYPES ===')
    all_columns = df[df['type'] == 'column']
    print(f'Total columns: {len(all_columns)}')
    
    # Show column name patterns
    pop_columns = all_columns[all_columns['column_name'].str.contains('_pop$', regex=True)]
    print(f'Columns ending with _pop: {len(pop_columns)}')
    
    # Show some examples of different column types
    print('\nSample _pop columns:')
    for idx, row in pop_columns[['column_name', 'description']].head(10).iterrows():
        print(f'- {row["column_name"]}: {row["description"]}')
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)