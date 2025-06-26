import pandas as pd
import numpy as np
from src.configurations.enums import DatasetName, Frequency
from src.dataset.dataset_factory import DatasetFactory
from src.configurations.input_column import InputColumnConfig
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.forecasting import ForecastConfig
from src.configurations.cross_validation import CrossValidationConfig
from src.preprocessing.nixtla_preprocessor import NixtlaPreprocessor
from tabpfn import TabPFNRegressor

print("=== Debugging TabPFN Negative Predictions ===")

# Load a single problematic series to debug
dataset = DatasetFactory.create_dataset(DatasetName.BAKERY)

input_columns = InputColumnConfig(
    sku_index="bdID",
    date="dateID", 
    target="target",
    frequency="frequency"
)

forecast_columns = ForecastColumnConfig(
    sku_index="skuID",
    date="date",
    target="demand",
    cutoff="cutoff",
    base_exogenous=["companyID", "storeID", "productID", "not_for_sale"],
    static=["companyID", "storeID", "productID"]
)

# Simple preprocessing without fill_gaps to avoid errors
print("Loading raw data...")
df_raw = dataset.get_merged_data().to_pandas()
print(f"Raw data shape: {df_raw.shape}")

# Filter for weekly frequency and select one problematic series
df_weekly = df_raw[df_raw["frequency"] == "weekly"].copy()
series_id = 207420  # One of the problematic series

series_data = df_weekly[df_weekly["bdID"] == series_id].copy()
print(f"\n=== Series {series_id} Analysis ===")
print(f"Number of observations: {len(series_data)}")

if len(series_data) > 0:
    # Sort by date
    series_data = series_data.sort_values("dateID")
    target_values = series_data["target"].values
    
    print(f"Target stats:")
    print(f"  Min: {target_values.min():.4f}")
    print(f"  Max: {target_values.max():.4f}")
    print(f"  Mean: {target_values.mean():.4f}")
    print(f"  Std: {target_values.std():.4f}")
    print(f"  Zeros: {(target_values == 0).sum()} / {len(target_values)}")
    print(f"  Negatives: {(target_values < 0).sum()}")
    
    print(f"\nFirst 10 target values: {target_values[:10]}")
    print(f"Last 10 target values: {target_values[-10:]}")
    
    # Manually create lag features like TabPFN would
    n_lags = 20
    if len(target_values) > n_lags:
        print(f"\n=== Lag Feature Analysis (n_lags={n_lags}) ===")
        
        X_features = []
        y_target = []
        
        for i in range(n_lags, len(target_values)):
            lag_features = target_values[i - n_lags : i]
            target = target_values[i]
            
            X_features.append(lag_features)
            y_target.append(target)
        
        X_features = np.array(X_features)
        y_target = np.array(y_target)
        
        print(f"Training data shape: X={X_features.shape}, y={y_target.shape}")
        print(f"Y training stats:")
        print(f"  Min: {y_target.min():.4f}")
        print(f"  Max: {y_target.max():.4f}")
        print(f"  Mean: {y_target.mean():.4f}")
        print(f"  Std: {y_target.std():.4f}")
        
        print(f"\nX features stats (lag features):")
        print(f"  Min: {X_features.min():.4f}")
        print(f"  Max: {X_features.max():.4f}")
        print(f"  Mean: {X_features.mean():.4f}")
        print(f"  Std: {X_features.std():.4f}")
        
        # Check if any lag features are negative (this could cause issues)
        print(f"  Negative lag features: {(X_features < 0).sum()}")
        
        # Try fitting a simple TabPFN model with just lag features
        print(f"\n=== Testing TabPFN with Lag Features Only ===")
        try:
            model = TabPFNRegressor(device="cpu", n_estimators=4, random_state=42)
            
            # Use only a subset if too much data
            if len(X_features) > 1000:
                indices = np.random.choice(len(X_features), size=1000, replace=False)
                X_train = X_features[indices]
                y_train = y_target[indices]
            else:
                X_train = X_features
                y_train = y_target
                
            print(f"Training TabPFN with {len(X_train)} samples...")
            model.fit(X_train, y_train)
            
            # Test prediction with last lag features
            test_features = target_values[-n_lags:].reshape(1, -1)
            print(f"Test features (last {n_lags} values): {test_features[0]}")
            
            prediction = model.predict(test_features)[0]
            print(f"TabPFN prediction: {prediction:.6f}")
            
            # Check what the expected prediction should be
            recent_mean = target_values[-10:].mean()
            print(f"Recent mean (last 10): {recent_mean:.6f}")
            print(f"Overall mean: {target_values.mean():.6f}")
            
            if prediction < 0:
                print(f"*** NEGATIVE PREDICTION DETECTED! ***")
                print("This suggests TabPFN is learning incorrect patterns")
                
                # Try multiple predictions to see pattern
                print("\nTesting multiple predictions:")
                for i in range(3):
                    test_idx = -(n_lags + i + 1)
                    if abs(test_idx) <= len(target_values):
                        test_feat = target_values[test_idx:test_idx+n_lags].reshape(1, -1)
                        pred = model.predict(test_feat)[0]
                        actual = target_values[test_idx + n_lags] if test_idx + n_lags < len(target_values) else "N/A"
                        print(f"  Features from index {test_idx}: pred={pred:.4f}, actual={actual}")
            
        except Exception as e:
            print(f"TabPFN fitting failed: {e}")
    
    # Check if scaling might be involved
    print(f"\n=== Checking for Scaling Issues ===")
    # Try z-score scaling like the preprocessing might do
    scaled_target = (target_values - target_values.mean()) / (target_values.std() + 1e-8)
    print(f"If z-score scaled - Min: {scaled_target.min():.4f}, Max: {scaled_target.max():.4f}")
    
    if scaled_target.min() < -3 or scaled_target.max() > 3:
        print("*** EXTREME SCALING DETECTED - this could cause negative predictions ***")

else:
    print("No data found for this series!") 