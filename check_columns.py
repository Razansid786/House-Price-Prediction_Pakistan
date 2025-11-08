import pandas as pd
import joblib

print("="*60)
print("ACTUAL DATASET COLUMNS:")
print("="*60)
df = pd.read_csv('data/zameen_updated.csv')
print(df.columns.tolist())
print()

print("="*60)
print("SAMPLE DATA (first row):")
print("="*60)
print(df.head(1).T)
print()

print("="*60)
print("COLUMNS EXPECTED BY MODEL:")
print("="*60)
features_df = pd.read_csv('data/processed/features.csv')
print(features_df.columns.tolist())
print()

print("="*60)
print("DATA TYPES:")
print("="*60)
print(features_df.dtypes)
