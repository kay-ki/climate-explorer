# %%
import shutil
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os
import warnings
import pickle
import shutil
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    print("Prophet not available. Skipping Prophet model.")

# =====================================================
# CONFIG
# =====================================================
BASELINE_START_YEAR = 1991
BASELINE_END_YEAR = 2020
TRAIN_TEST_SPLIT_YEAR = 2010
NUM_COUNTRIES = None  # Fewer countries = faster execution
# SELECTED_COUNTRIES = ['USA', 'RUS', 'DEU', 'CHN', 'IND',
#                       'BRA', 'AUS', 'CAN', 'JPN', 'ZAF', 'GBR', 'FRA', 'MEX']
SELECTED_COUNTRIES = None
SAVE_INTERVAL = 5
OUTPUT_DIR = 'output'
USE_ENSEMBLE = False  # Use ensemble (True) or best model (False)
ADD_CLIMATE_TREND = False  # Add realistic warming trend
GLOBAL_WARMING_RATE = 0.25  # °C per decade
RANDOM_SEED = 2

# Regional climate zones for realistic variations
CLIMATE_REGIONS = {
    'north_america': ['CAN', 'USA', 'UMI', 'GRL', 'SPM', 'BMU'],
    'caribbean_central_america': [
        'ABW', 'AIA', 'ATG', 'BHS', 'BLM', 'BLZ', 'BRB', 'CUB', 'CUW', 'CYM',
        'DMA', 'DOM', 'GRD', 'GTM', 'HND', 'HTI', 'JAM', 'KNA', 'LCA', 'MAF',
        'MEX', 'MSR', 'MTQ', 'NIC', 'PAN', 'PRI', 'SLV', 'SXM', 'TCA', 'TTO',
        'VCT', 'VGB', 'VIR', 'BES'
    ],
    'south_america': [
        'ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'GUF', 'GUY', 'PER', 'PRY',
        'SUR', 'URY', 'VEN'
    ],
    'western_europe': [
        'ALA', 'AND', 'AUT', 'BEL', 'CHE', 'DEU', 'DNK', 'ESP', 'FIN', 'FRA',
        'GBR', 'GGY', 'GIB', 'IRL', 'IMN', 'ISL', 'ITA', 'JEY', 'LIE', 'LUX',
        'MCO', 'NLD', 'NOR', 'PRT', 'SMR', 'SWE', 'VAT'
    ],
    'eastern_europe': [
        'ALB', 'BLR', 'BGR', 'BIH', 'CZE', 'EST', 'GEO', 'HRV', 'HUN', 'KSV',
        'LTU', 'LVA', 'MDA', 'MKD', 'MNE', 'POL', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'UKR'
    ],
    'middle_east_north_africa': [
        'ARE', 'BHR', 'DZA', 'EGY', 'IRN', 'IRQ', 'ISR', 'JOR', 'KWT', 'LBN',
        'LBY', 'MAR', 'OMN', 'PSE', 'QAT', 'SAU', 'SDN', 'SYR', 'TUN', 'TUR', 'YEM'
    ],
    'sub_saharan_africa': [
        'AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR', 'COD', 'COG',
        'CPV', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ',
        'KEN', 'LBR', 'LSO', 'MLI', 'MOZ', 'MRT', 'MUS', 'MWI', 'NAM', 'NER',
        'NGA', 'REU', 'RWA', 'SEN', 'SHN', 'SLE', 'SOM', 'SSD', 'STP', 'SWZ',
        'TCD', 'TGO', 'TZA', 'UGA', 'ZAF', 'ZMB', 'ZWE'
    ],
    'south_asia': [
        'AFG', 'BGD', 'BTN', 'IND', 'LKA', 'MDV', 'NPL', 'PAK'
    ],
    'east_asia': [
        'CHN', 'HKG', 'JPN', 'KOR', 'MAC', 'MNG', 'PRK', 'TWN'
    ],
    'southeast_asia': [
        'BRN', 'IDN', 'KHM', 'LAO', 'MMR', 'MYS', 'PHL', 'SGP', 'THA', 'TLS', 'VNM'
    ],
    'central_asia': [
        'KAZ', 'KGZ', 'TJK', 'TKM', 'UZB'
    ],
    'australia_oceania': [
        'ASM', 'AUS', 'CCK', 'COK', 'CXR', 'FJI', 'FSM', 'GUM', 'KIR', 'MHL',
        'MNP', 'NCL', 'NFK', 'NIU', 'NRU', 'NZL', 'PCN', 'PLW', 'PNG', 'PYF',
        'SLB', 'TKL', 'TON', 'TUV', 'VUT', 'WLF', 'WSM', 'MYT'
    ],
    'polar_regions': [
        'ATA', 'ATF', 'BVT', 'HMD', 'IOT', 'SJM'
    ]
}

# Create country to region mapping
COUNTRY_TO_REGION = {}
for region, countries in CLIMATE_REGIONS.items():
    for country in countries:
        COUNTRY_TO_REGION[country] = region

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# %%


def save_progress(data, filename, is_df=True):
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        if is_df:
            data.to_csv(filepath, index=False)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        print(f"Saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving: {str(e)}")
        return False


def load_progress(filename, is_df=True):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return None

    try:
        if is_df:
            return pd.read_csv(filepath)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading: {str(e)}")
        return None

# %%


def calculate_statistical_confidence(predictions_df, temp_long, performance_df):
    """
    Pass through function. Obsolete
    """
    print("Using direct confidence intervals for visualization...")
    return predictions_df


def check_stationarity(series):
    """Check if a time series is stationary and transform if needed"""
    # Make a copy
    data = series.copy()
    transform_info = {'differenced': False, 'log_transform': False}

    # Check if series has enough data
    if len(data) < 30:
        return data, transform_info

    # Check stationarity with ADF test
    try:
        adf_result = adfuller(data.dropna())
        p_value = adf_result[1]

        # Try log transform for skewed data
        if data.min() > 0 and np.abs(data.skew()) > 1:
            transform_info['log_transform'] = True
            data = np.log1p(data)

            # Check if log transform helped
            adf_result = adfuller(data.dropna())
            p_value = adf_result[1]

        # If still not stationary, try differencing
        if p_value > 0.05:
            transform_info['differenced'] = True
            data = data.diff().dropna()
    except:
        pass

    return data, transform_info


def reverse_transforms(predictions, transform_info, last_value=None):
    """Reverse transformations applied to make data stationary"""
    result = predictions.copy()

    # Undo differencing
    if transform_info['differenced'] and last_value is not None:
        result = np.insert(result, 0, last_value)
        result = np.cumsum(result)[1:]

    # Undo log transform
    if transform_info['log_transform']:
        result = np.expm1(result)

    return result


def find_best_lags(series, max_lag=36):
    """Find significant lag values based on autocorrelation"""
    from statsmodels.tsa.stattools import acf

    if len(series) <= max_lag:
        return [1, 12]  # Default to monthly and yearly

    try:
        # Calculate autocorrelation
        acf_values = acf(series.dropna(), nlags=max_lag)

        # Find significant lags
        significant_lags = [i for i in range(1, len(acf_values))
                            if abs(acf_values[i]) > 0.2]

        # Always include key climate lags
        key_lags = [1, 12]  # Monthly and yearly seasonality
        optimal_lags = sorted(list(set(significant_lags + key_lags)))

        # Limit number of lags
        if len(optimal_lags) > 6:
            optimal_lags = sorted(
                list(set([1, 2, 3, 6, 12, 24, 36]) & set(optimal_lags)))

        return optimal_lags
    except:
        return [1, 12]  # Default if anything fails

# %%


def add_features(df, target_col):
    """Add time series features with smarter lag selection"""
    df_result = df.sort_values(['year', 'month']).copy()

    # Add seasonal indicators
    df_result['sin_month'] = np.sin(2 * np.pi * df_result['month'] / 12)
    df_result['cos_month'] = np.cos(2 * np.pi * df_result['month'] / 12)

    for code in df_result['code'].unique():
        country_data = df_result[df_result['code'] == code].copy()

        # Find best lags for this country
        optimal_lags = find_best_lags(country_data[target_col])

        # Add lagged features
        for lag in optimal_lags:
            lag_name = f'lag{lag}'
            country_data[lag_name] = country_data[target_col].shift(lag)

        # Add rolling averages
        country_data['roll12_mean'] = country_data[target_col].rolling(
            12).mean()

        # Add trend feature
        if len(country_data) >= 60:
            country_data['roll60_mean'] = country_data[target_col].rolling(
                60).mean()
            if len(country_data) >= 120:
                country_data['roll120_mean'] = country_data[target_col].rolling(
                    120).mean()
                country_data['trend'] = country_data['roll60_mean'] - \
                    country_data['roll120_mean']
            else:
                country_data['trend'] = country_data['roll60_mean'] - \
                    country_data['roll60_mean'].mean()
        else:
            # Simple trend for shorter series
            country_data['trend'] = country_data[target_col] - \
                country_data[target_col].mean()

        years_since_start = country_data['year'] - \
            country_data['year'].min() + 1
        country_data['explicit_trend'] = years_since_start

        # Update main dataframe
        df_result.loc[df_result['code'] == code] = country_data

    # Fill NAs with country means
    feature_cols = [col for col in df_result.columns
                    if col.startswith('lag') or col.startswith('roll') or col == 'trend']

    for col in feature_cols:
        group_means = df_result.groupby('code')[col].transform('mean')
        df_result[col] = df_result[col].fillna(group_means)

    return df_result

# %%


def reverse_transforms(predictions, transform_info, last_value=None):
    """Reverse transformations applied to make data stationary"""
    result = predictions.copy()

    # Undo differencing
    if transform_info['differenced'] and last_value is not None:
        result = np.insert(result, 0, last_value)
        result = np.cumsum(result)[1:]

    # Undo log transform
    if transform_info['log_transform']:
        # Use a different approach for confidence bounds
        if len(result.shape) > 1 or isinstance(result, tuple):
            # Handle confidence bounds separately
            result = np.expm1(result)
        else:
            result = np.expm1(result)

    return result


def build_sarima_model(train_data, test_data, target_col, country_code, forecast_steps=360):
    """Build SARIMA model with auto-parameter selection"""
    try:
        # Filter to one country
        train_y = train_data[train_data['code'] == country_code][target_col]
        test_y = test_data[test_data['code'] == country_code][target_col]

        # Check stationarity
        transformed_train, transform_info = check_stationarity(train_y)
        last_train_value = train_y.iloc[-1] if len(train_y) > 0 else 0

        # Try different parameter combinations
        param_options = [
            ((1, 0, 0), (1, 0, 0, 12)),  # Simple AR with seasonal AR
            ((1, 1, 0), (1, 0, 0, 12)),  # AR with differencing
            ((0, 1, 1), (0, 1, 1, 12)),  # Simple MA with seasonal differencing
            ((1, 1, 1), (1, 0, 0, 12)),  # ARIMA with seasonal AR
            ((1, 1, 1), (0, 1, 1, 12)),  # ARIMA with seasonal MA
            ((1, 1, 1), (1, 1, 1, 12)),  # Full seasonal ARIMA
        ]

        # If already differenced in stationarity check, adjust models
        if transform_info['differenced']:
            param_options = [
                ((1, 0, 0), (1, 0, 0, 12)),  # AR with seasonal AR
                ((0, 0, 1), (0, 0, 1, 12)),  # MA with seasonal MA
                ((1, 0, 1), (1, 0, 1, 12)),  # ARMA with seasonal ARMA
            ]

        best_aic = float("inf")
        best_params = None

        for order, seasonal_order in param_options:
            try:
                model = SARIMAX(
                    transformed_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit = model.fit(disp=False, maxiter=50)

                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_params = (order, seasonal_order)
            except:
                continue

        # Use default parameters if none worked
        if best_params is None:
            best_params = ((1, 1, 1), (1, 1, 1, 12))

        # Fit model with best parameters
        best_model = SARIMAX(
            transformed_train,
            order=best_params[0],
            seasonal_order=best_params[1],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = best_model.fit(disp=False)

        # Forecast test period
        raw_test_forecast = model_fit.forecast(steps=len(test_y))
        test_forecast = reverse_transforms(
            raw_test_forecast, transform_info, last_train_value)
        test_mse = mean_squared_error(test_y, test_forecast)

        # Forecast future
        raw_future_forecast = model_fit.forecast(steps=forecast_steps)
        future_forecast = reverse_transforms(
            raw_future_forecast, transform_info, last_train_value)

        # Get confidence intervals - need special handling for log transform!
        pred_interval = model_fit.get_forecast(steps=forecast_steps).conf_int()

        if transform_info['log_transform']:
            # For log transform, we need to be extra careful with intervals
            # Transform the point forecast first
            point_forecast = reverse_transforms(
                raw_future_forecast, transform_info, last_train_value)

            # For log-transformed data, use percentage-based intervals
            lower_pct = pred_interval.iloc[:, 0] - raw_future_forecast
            upper_pct = pred_interval.iloc[:, 1] - raw_future_forecast

            # Apply percentage adjustments to the untransformed forecast
            lower_bounds = point_forecast * \
                (1 + lower_pct.apply(lambda x: np.tanh(x)))
            upper_bounds = point_forecast * \
                (1 + upper_pct.apply(lambda x: np.tanh(x)))
        else:
            # Non-log-transformed: use standard reversal
            lower_bounds = reverse_transforms(
                pred_interval.iloc[:, 0], transform_info, last_train_value)
            upper_bounds = reverse_transforms(
                pred_interval.iloc[:, 1], transform_info, last_train_value)

        return {
            'model': model_fit,
            'test_mse': test_mse,
            'forecast': future_forecast,
            'lower': np.array(lower_bounds),
            'upper': np.array(upper_bounds)
        }
    except Exception as e:
        print(f"  SARIMA error for {country_code}")
        return None


def build_prophet_model(train_data, test_data, target_col, country_code, forecast_steps=360):
    """Build and evaluate a Prophet model with parameter tuning"""
    if not prophet_available:
        return None

    try:
        # Filter to one country
        country_train = train_data[train_data['code'] == country_code].copy()
        country_test = test_data[test_data['code'] == country_code].copy()

        # Prepare data for Prophet
        prophet_train = pd.DataFrame({
            'ds': pd.to_datetime(country_train['year'].astype(str) + '-' +
                                 country_train['month'].astype(str).str.zfill(2) + '-01'),
            'y': country_train[target_col]
        })

        prophet_test = pd.DataFrame({
            'ds': pd.to_datetime(country_test['year'].astype(str) + '-' +
                                 country_test['month'].astype(str).str.zfill(2) + '-01'),
            'y': country_test[target_col]
        })

        # Different parameter options
        param_options = [
            {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0},
            {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0},
            {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 1.0}
        ]

        best_mse = float('inf')
        best_model = None

        # Find best parameters
        for params in param_options:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                **params
            )

            model.fit(prophet_train)
            test_forecast = model.predict(prophet_test[['ds']])
            current_mse = mean_squared_error(
                prophet_test['y'], test_forecast['yhat'])

            if current_mse < best_mse:
                best_mse = current_mse
                best_model = model

        # Forecast with best model
        last_date = prophet_test['ds'].max() if len(
            prophet_test) > 0 else prophet_train['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_steps,
            freq='MS'  # Month start
        )
        future = pd.DataFrame({'ds': future_dates})

        future_forecast = best_model.predict(future)

        return {
            'model': best_model,
            'test_mse': best_mse,
            'forecast': future_forecast['yhat'].values,
            'lower': future_forecast['yhat_lower'].values,
            'upper': future_forecast['yhat_upper'].values
        }
    except Exception as e:
        print(f"  Prophet error for {country_code}")
        return None


def build_gbr_model(train_data, test_data, target_col, country_code, forecast_steps=360):
    """Build Gradient Boosting model"""
    try:
        # Filter to one country
        country_train = train_data[train_data['code'] == country_code].copy()
        country_test = test_data[test_data['code'] == country_code].copy()

        # Get all available feature columns
        feature_cols = [col for col in country_train.columns if col.startswith('lag') or
                        col.startswith('roll') or col == 'trend' or col == 'explicit_trend' or
                        col in ['sin_month', 'cos_month', 'month']]

        # Prepare data
        X_train = country_train[feature_cols]
        y_train = country_train[target_col]
        X_test = country_test[feature_cols]
        y_test = country_test[target_col]

        # Try different parameters
        param_options = [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 4},
        ]

        best_mse = float('inf')
        best_model = None

        for params in param_options:
            model = GradientBoostingRegressor(
                random_state=RANDOM_SEED, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            if mse < best_mse:
                best_mse = mse
                best_model = model

        # Generate future features for prediction
        future_data = generate_future_features(
            pd.concat([country_train, country_test]),
            forecast_steps,
            feature_cols
        )

        # Predict future
        future_forecast = best_model.predict(future_data)

        # Calculate CORRECT confidence intervals
        # Use standard error from test data, not training data
        test_errors = y_test - best_model.predict(X_test)
        std_error = np.std(test_errors)

        # Increase uncertainty with prediction distance
        uncertainty = std_error * np.sqrt(np.arange(1, forecast_steps + 1))

        lower_bounds = future_forecast - 1.96 * uncertainty
        upper_bounds = future_forecast + 1.96 * uncertainty

        return {
            'model': best_model,
            'test_mse': best_mse,
            'forecast': future_forecast,
            'lower': lower_bounds,
            'upper': upper_bounds
        }
    except Exception as e:
        print(f"  GBR error for {country_code}")
        return None


def generate_future_features(historical_data, steps, feature_cols=None):
    """Generate feature matrix for future predictions"""
    # Get last date
    last_data = historical_data.sort_values(['year', 'month']).iloc[-1]
    last_year = last_data['year']
    last_month = last_data['month']

    # Generate future dates
    future_dates = []
    year, month = last_year, last_month

    for _ in range(steps):
        month += 1
        if month > 12:
            month = 1
            year += 1
        future_dates.append((year, month))

    # Create DataFrame with dates
    future_df = pd.DataFrame(future_dates, columns=['year', 'month'])

    # Add seasonal features
    future_df['sin_month'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['cos_month'] = np.cos(2 * np.pi * future_df['month'] / 12)

    # Use last values for other features
    if feature_cols is None:
        feature_cols = [col for col in historical_data.columns if col.startswith('lag') or
                        col.startswith('roll') or col == 'trend']

    for col in feature_cols:
        if col in historical_data.columns and col not in future_df.columns:
            future_df[col] = historical_data[col].iloc[-1]

    # Return only the required columns
    return future_df[feature_cols]


def create_weighted_ensemble(forecasts, test_mses=None):
    """Create weighted ensemble based on model performance"""
    if not forecasts or len(forecasts) == 0:
        return np.array([])

    # Calculate weights from MSE values
    if test_mses and len(test_mses) == len(forecasts):
        # Weights inversely proportional to MSE
        inverse_mses = [1/max(0.0001, mse)
                        for mse in test_mses]  # Prevent division by zero
        sum_weights = sum(inverse_mses)

        # Prevent division by zero
        if sum_weights > 0:
            weights = [inv_mse/sum_weights for inv_mse in inverse_mses]
        else:
            weights = [1/len(forecasts)] * len(forecasts)
    else:
        # Equal weights if no MSE values
        weights = [1/len(forecasts)] * len(forecasts)

    # Make sure all forecasts have same length
    min_len = min(len(f) for f in forecasts)
    forecasts = [f[:min_len] for f in forecasts]

    # Create weighted average
    ensemble = np.zeros(min_len)
    for i, forecast in enumerate(forecasts):
        if not isinstance(forecast, np.ndarray):
            forecast = np.array(forecast)
        ensemble += forecast * weights[i]

    return ensemble


def analyze_climate_patterns(temp_long, prec_long):
    """Analyze regional climate variability patterns"""
    # Calculate monthly standard deviations by country
    country_temp_variability = {}
    region_temp_variability = {}

    # Loop through each country
    for country in temp_long['code'].unique():
        country_temp = temp_long[temp_long['code'] == country].copy()

        # Skip if insufficient data
        if len(country_temp) < 24:
            continue

        # Get region for this country
        region = COUNTRY_TO_REGION.get(country, 'other')

        # Calculate monthly standard deviations
        monthly_temp_std = country_temp.groupby(
            'month')['temp_anomaly'].std().to_dict()

        # Store by country
        country_temp_variability[country] = monthly_temp_std

        # Add to region statistics
        if region not in region_temp_variability:
            region_temp_variability[region] = {
                month: [] for month in range(1, 13)}

        for month in range(1, 13):
            if month in monthly_temp_std:
                region_temp_variability[region][month].append(
                    monthly_temp_std[month])

    # Calculate average standard deviations by region
    for region in region_temp_variability:
        for month in range(1, 13):
            if region_temp_variability[region][month]:
                region_temp_variability[region][month] = np.mean(
                    region_temp_variability[region][month])
            else:
                # Global average if no data
                all_stds = [std for country_std in country_temp_variability.values()
                            for m, std in country_std.items() if m == month]
                region_temp_variability[region][month] = np.mean(
                    all_stds) if all_stds else 0.5

    # Calculate global average monthly std
    global_temp_std = {
        month: np.mean([std for country_std in country_temp_variability.values()
                       for m, std in country_std.items() if m == month])
        for month in range(1, 13)
    }

    return {
        'country_temp_variability': country_temp_variability,
        'region_temp_variability': region_temp_variability,
        'global_temp_std': global_temp_std
    }


def generate_regional_variations(variability_data, forecast_steps=360):
    """Generate realistic regional climate variations"""
    region_temp_variability = variability_data['region_temp_variability']
    global_temp_std = variability_data['global_temp_std']

    # Generate base variations by region
    region_variations = {}
    np.random.seed(RANDOM_SEED)

    for region in region_temp_variability.keys():
        # Generate realistic time series with autocorrelation
        white_noise = np.random.normal(0, 1, forecast_steps)

        # Apply autocorrelation (AR1 process)
        phi = 0.7  # Temporal persistence parameter
        red_noise = np.zeros(forecast_steps)
        red_noise[0] = white_noise[0]

        for t in range(1, forecast_steps):
            red_noise[t] = phi * red_noise[t-1] + \
                white_noise[t] * np.sqrt(1 - phi**2)

        # Scale by monthly standard deviation
        months = [(t % 12) + 1 for t in range(forecast_steps)]
        monthly_stds = [region_temp_variability[region].get(month, global_temp_std.get(month, 0.5))
                        for month in months]

        scaled_variations = red_noise * np.array(monthly_stds)

        # Store for this region
        region_variations[region] = {'temperature': scaled_variations}

    return region_variations


def apply_regional_variations(country_code, temp_ensemble, region_variations, variability_data):
    """Apply realistic regional variations to model forecasts"""
    # Get region for this country
    region = COUNTRY_TO_REGION.get(country_code, 'other')

    # Use default region if not found
    if region not in region_variations:
        available_regions = list(region_variations.keys())
        region = available_regions[0] if available_regions else None

    if region is None:
        return temp_ensemble

    # Get regional variations
    temp_variations = region_variations[region]['temperature']
    temp_variations = temp_variations[:len(temp_ensemble)]

    # If variations are shorter than ensemble, repeat
    if len(temp_variations) < len(temp_ensemble):
        multiplier = int(np.ceil(len(temp_ensemble) / len(temp_variations)))
        temp_variations = np.tile(temp_variations, multiplier)[
            :len(temp_ensemble)]

    # Apply country-specific scaling factor
    country_temp_factor = 1.0
    if country_code in variability_data.get('country_temp_variability', {}):
        # Use average of monthly standard deviations
        country_std = np.mean(
            list(variability_data['country_temp_variability'][country_code].values()))
        region_std = np.mean(
            [std for month, std in variability_data['region_temp_variability'][region].items()])
        if region_std > 0:
            country_temp_factor = country_std / region_std

    # Apply variations to forecast
    temp_ensemble_with_variations = temp_ensemble + \
        temp_variations * country_temp_factor

    return temp_ensemble_with_variations


def add_climate_trend(forecast, start_year, country_code):
    """Add realistic climate trend to forecasts"""
    if not ADD_CLIMATE_TREND:
        return forecast

    # Create year array
    years = np.array([start_year + (i // 12) for i in range(len(forecast))])

    # Calculate year fractions since start
    year_fractions = (years - start_year) / 10  # per decade

    # Vary the trend slightly by country
    np.random.seed(
        int(hashlib.md5(country_code.encode()).hexdigest(), 16) % 2**32)
    country_factor = 0.8 + 0.4 * np.random.random()  # 0.8-1.2 range

    # More warming in later years (accelerating trend)
    trend_adjustment = GLOBAL_WARMING_RATE * country_factor * \
        (year_fractions + 0.1 * year_fractions**2)

    return forecast + trend_adjustment

# %%


def load_and_process_data(temp_file, prec_file):
    """Load climate data and process it"""
    print("Loading climate data...")

    # Check if processed data already exists
    processed_temp = load_progress('processed_temp.csv')
    processed_prec = load_progress('processed_prec.csv')

    if processed_temp is not None and processed_prec is not None:
        print("Loaded preprocessed data")
        return processed_temp, processed_prec

    # Load the raw data
    temp_df = pd.read_excel(temp_file)
    prec_df = pd.read_excel(prec_file)

    # Get time columns (everything except code and name)
    time_columns = [
        col for col in temp_df.columns if col not in ['code', 'name']]

    # Convert to long format
    temp_long = pd.melt(
        temp_df,
        id_vars=['code', 'name'],
        value_vars=time_columns,
        var_name='date',
        value_name='temperature'
    )

    prec_long = pd.melt(
        prec_df,
        id_vars=['code', 'name'],
        value_vars=time_columns,
        var_name='date',
        value_name='precipitation'
    )

    # Extract year and month
    for df in [temp_long, prec_long]:
        df['year'] = df['date'].str.split('-').str[0].astype(int)
        df['month'] = df['date'].str.split('-').str[1].astype(int)

    # Save processed data
    save_progress(temp_long, 'processed_temp.csv')
    save_progress(prec_long, 'processed_prec.csv')

    return temp_long, prec_long


def calculate_climate_anomalies(temp_long, prec_long):
    """Calculate temperature anomalies and precipitation percent relative to baseline"""
    print(
        f"Calculating anomalies ({BASELINE_START_YEAR}-{BASELINE_END_YEAR} baseline)...")

    # Check for existing anomaly data
    anomaly_temp = load_progress('anomaly_temp.csv')
    anomaly_prec = load_progress('anomaly_prec.csv')

    if anomaly_temp is not None and anomaly_prec is not None:
        print("Loaded anomaly data")
        return anomaly_temp, anomaly_prec

    # Calculate temperature anomalies
    baseline_temp = temp_long[
        (temp_long['year'] >= BASELINE_START_YEAR) &
        (temp_long['year'] <= BASELINE_END_YEAR)
    ]

    # Calculate baseline averages
    baseline_means = baseline_temp.groupby(['code', 'month'])[
        'temperature'].mean().reset_index()
    baseline_means.rename(
        columns={'temperature': 'baseline_temp'}, inplace=True)

    # Join baseline means to original data
    temp_long = pd.merge(
        temp_long,
        baseline_means,
        on=['code', 'month'],
        how='left'
    )

    # Calculate anomalies
    temp_long['temp_anomaly'] = temp_long['temperature'] - \
        temp_long['baseline_temp']

    # Calculate precipitation as percent of normal
    baseline_prec = prec_long[
        (prec_long['year'] >= BASELINE_START_YEAR) &
        (prec_long['year'] <= BASELINE_END_YEAR)
    ]

    baseline_prec_means = baseline_prec.groupby(
        ['code', 'month'])['precipitation'].mean().reset_index()
    baseline_prec_means.rename(
        columns={'precipitation': 'baseline_prec'}, inplace=True)

    prec_long = pd.merge(
        prec_long,
        baseline_prec_means,
        on=['code', 'month'],
        how='left'
    )

    # Calculate precipitation percentage
    prec_long['prec_percent'] = (
        prec_long['precipitation'] / prec_long['baseline_prec'] * 100)

    # Save anomaly data
    save_progress(temp_long, 'anomaly_temp.csv')
    save_progress(prec_long, 'anomaly_prec.csv')

    return temp_long, prec_long


def handle_missing_values(df, method='interpolate'):
    """Handle missing values in climate data"""
    print(f"Handling missing values...")

    df_clean = df.copy()

    if method == 'interpolate':
        # Process each country separately
        for country in df_clean['code'].unique():
            # Get country data and sort by time
            country_data = df_clean[df_clean['code'] == country].copy()
            country_data = country_data.sort_values(['year', 'month'])

            # Interpolate numeric columns
            numeric_cols = country_data.select_dtypes(
                include=[np.number]).columns
            for col in numeric_cols:
                country_data[col] = country_data[col].interpolate(
                    method='linear', limit_direction='both')

            # Update main dataframe
            df_clean.loc[df_clean['code'] == country] = country_data

    # Fill any remaining NaNs with global means
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    return df_clean


def generate_climate_predictions(temp_long, prec_long):
    """Generate climate predictions for all countries"""
    print("Generating climate predictions...")
    start_time = datetime.now()

    # Get countries to process
    all_countries = temp_long['code'].unique()

    # Check for previous progress
    predictions_file = os.path.join(OUTPUT_DIR, 'predictions_in_progress.csv')
    performance_file = os.path.join(OUTPUT_DIR, 'performance_in_progress.csv')
    processed_countries_file = os.path.join(
        OUTPUT_DIR, 'processed_countries.pkl')

    existing_predictions = None
    existing_performance = None
    processed_countries = []

    # First check if we have final predictions (the complete list)
    final_predictions_file = os.path.join(OUTPUT_DIR, 'final_predictions.csv')
    if os.path.exists(final_predictions_file):
        existing_predictions = pd.read_csv(final_predictions_file)
        processed_countries = existing_predictions['code'].unique().tolist()
        print(
            f"Found {len(processed_countries)} countries already processed in final_predictions.csv")
    # If not, check for work in progress files
    elif os.path.exists(predictions_file) and os.path.exists(performance_file) and os.path.exists(processed_countries_file):
        try:
            existing_predictions = pd.read_csv(predictions_file)
            existing_performance = pd.read_csv(performance_file)
            with open(processed_countries_file, 'rb') as f:
                processed_countries = pickle.load(f)
            print(
                f"Loaded previous progress: {len(processed_countries)} countries already processed")
        except:
            existing_predictions = None
            existing_performance = None
            processed_countries = []

    # Select countries to process
    if SELECTED_COUNTRIES:
        selected_countries = [
            c for c in all_countries if c in SELECTED_COUNTRIES and c not in processed_countries]
        print(
            f"Processing {len(selected_countries)} selected countries (+{len(processed_countries)} already done)")
    elif NUM_COUNTRIES is not None and NUM_COUNTRIES < len(all_countries):
        remaining_countries = [
            c for c in all_countries if c not in processed_countries]
        num_to_select = min(
            NUM_COUNTRIES - len(processed_countries), len(remaining_countries))
        if num_to_select > 0:
            selected_countries = np.random.choice(
                remaining_countries, num_to_select, replace=False)
        else:
            selected_countries = []
        print(
            f"Processing {len(selected_countries)} random countries (+{len(processed_countries)} already done)")
    else:
        selected_countries = [
            c for c in all_countries if c not in processed_countries]
        print(
            f"Processing all {len(selected_countries)} remaining countries (+{len(processed_countries)} already done)")

    # Add features to data
    print("Adding features to temperature data...")
    temp_with_features = add_features(temp_long, 'temp_anomaly')

    # Split into train and test
    temp_train = temp_with_features[temp_with_features['year'] < 2018]
    temp_test = temp_with_features[(temp_with_features['year'] >= 2018) &
                                   (temp_with_features['year'] <= 2023)]

    # Future years to predict (2024-2050)
    forecast_steps = (2050 - 2024 + 1) * 12

    # Analyze climate variability patterns
    variability_data = analyze_climate_patterns(temp_long, prec_long)

    # Generate regional variations
    region_variations = generate_regional_variations(
        variability_data, forecast_steps)

    # Initialize prediction arrays
    predictions = [] if existing_predictions is None else existing_predictions.to_dict(
        'records')
    model_performance = [
    ] if existing_performance is None else existing_performance.to_dict('records')

    # Process each country
    for i, country_code in enumerate(selected_countries):
        # Estimate time remaining
        if i > 0:
            elapsed = (datetime.now() -
                       start_time).total_seconds() / 60  # minutes
            avg_time_per_country = elapsed / i
            remaining_countries = len(selected_countries) - i
            est_time = avg_time_per_country * remaining_countries
            print(
                f"Country {i+1}/{len(selected_countries)}: {country_code} (Est. remaining: {est_time:.1f} min)")
        else:
            print(f"Country {i+1}/{len(selected_countries)}: {country_code}")

        try:
            country_name = temp_long[temp_long['code']
                                     == country_code]['name'].iloc[0]
        except:
            country_name = country_code

        # Check for sufficient data
        country_temp_data = temp_with_features[temp_with_features['code']
                                               == country_code]

        if len(country_temp_data) < 60:
            print(
                f"  Skipping {country_code} - insufficient data ({len(country_temp_data)} points)")
            continue

        # Build temperature models
        try:
            temp_models = {}
            temp_models['SARIMA'] = build_sarima_model(
                temp_train, temp_test, 'temp_anomaly', country_code, forecast_steps)
            if prophet_available:
                temp_models['Prophet'] = build_prophet_model(
                    temp_train, temp_test, 'temp_anomaly', country_code, forecast_steps)
            temp_models['GBR'] = build_gbr_model(
                temp_train, temp_test, 'temp_anomaly', country_code, forecast_steps)

            # Filter out failed models
            temp_models = {k: v for k, v in temp_models.items()
                           if v is not None}

            # Skip if all models failed
            if not temp_models:
                print(f"  Skipping {country_code} - all models failed")
                continue

            # Select best model
            best_temp_model = min(temp_models.items(),
                                  key=lambda x: x[1]['test_mse'])

            # Get forecasts
            temp_forecasts = [model['forecast']
                              for model in temp_models.values()]
            temp_test_mses = [model['test_mse']
                              for model in temp_models.values()]

            # Create ensemble or use best model
            if USE_ENSEMBLE:
                temp_ensemble = create_weighted_ensemble(
                    temp_forecasts, temp_test_mses)
            else:
                temp_ensemble = best_temp_model[1]['forecast']

            temp_ensemble = np.array(temp_ensemble)

            # Add climate trend
            temp_ensemble = add_climate_trend(
                temp_ensemble, 2024, country_code)

            # Apply realistic regional variations
            temp_ensemble = apply_regional_variations(
                country_code, temp_ensemble, region_variations, variability_data
            )

            # Calculate confidence intervals for forecasts
            if USE_ENSEMBLE and len(temp_models) > 1:
                # Create weighted confidence intervals for ensemble
                weighted_lower = np.zeros(forecast_steps)
                weighted_upper = np.zeros(forecast_steps)

                # Calculate weights based on model performance
                weights = []
                for model_name, model_details in temp_models.items():
                    weight = 1.0 / max(0.0001, model_details['test_mse'])
                    weights.append(weight)

                # Normalize weights
                weights = [w / sum(weights) for w in weights]

                # Create weighted confidence intervals
                i = 0
                for model_name, model_details in temp_models.items():
                    weighted_lower += weights[i] * model_details['lower']
                    weighted_upper += weights[i] * model_details['upper']
                    i += 1

                # Add small margin for ensemble uncertainty
                ensemble_lower = weighted_lower - 0.1
                ensemble_upper = weighted_upper + 0.1
            else:
                # Use confidence intervals from best model
                ensemble_lower = best_temp_model[1]['lower']
                ensemble_upper = best_temp_model[1]['upper']

                if not isinstance(ensemble_lower, np.ndarray):
                    ensemble_lower = np.array(ensemble_lower)

                if not isinstance(ensemble_upper, np.ndarray):
                    ensemble_upper = np.array(ensemble_upper)

            # Get baseline values for converting to absolute temperatures
            baseline_temps = temp_long[temp_long['code'] == country_code].groupby(
                'month')['baseline_temp'].mean().to_dict()

            # Store model performance
            model_performance.append({
                'code': country_code,
                'name': country_name,
                'variable': 'temperature',
                'best_model': best_temp_model[0],
                'best_model_mse': best_temp_model[1]['test_mse'],
                **{f"{model}_mse": details['test_mse'] for model, details in temp_models.items()}
            })

            # Generate predictions for future months
            start_year, start_month = 2024, 1
            for j in range(forecast_steps):
                year = start_year + (start_month - 1 + j) // 12
                month = ((start_month - 1 + j) % 12) + 1

                # Get baseline temp to convert anomaly to absolute
                baseline_temp = baseline_temps.get(month, sum(
                    baseline_temps.values()) / len(baseline_temps) if baseline_temps else 0)

                # Get confidence interval values
                temp_lower = ensemble_lower[j]
                temp_upper = ensemble_upper[j]

                # Calculate confidence based on interval width
                interval_width = temp_upper - temp_lower

                # Calculate values
                temp_anomaly = temp_ensemble[j]
                actual_temp = baseline_temp + temp_anomaly

                # Store prediction with confidence intervals
                predictions.append({
                    'code': country_code,
                    'country': country_name,
                    'year': year,
                    'month': month,
                    'temperature': actual_temp,
                    'temp_anomaly': temp_anomaly,
                    'temp_lower': baseline_temp + temp_lower,
                    'temp_upper': baseline_temp + temp_upper,
                    'interval_width': round(interval_width, 2),
                    'precipitation': None,
                    'prec_percent': None,
                    'model': f"temp:{best_temp_model[0]}",
                    'is_prediction': True,
                    'prediction_confidence': None
                })

            # Add to processed countries
            processed_countries.append(country_code)

            # Save progress at intervals
            if (i + 1) % SAVE_INTERVAL == 0 or i == len(selected_countries) - 1:
                print(
                    f"Saving progress ({i+1}/{len(selected_countries)} countries)...")
                pd.DataFrame(predictions).to_csv(predictions_file, index=False)
                pd.DataFrame(model_performance).to_csv(
                    performance_file, index=False)
                with open(processed_countries_file, 'wb') as f:
                    pickle.dump(processed_countries, f)

        except Exception as e:
            print(
                f"Error processing {country_code}: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Calculate global temperature by year
    print("Calculating global temperature averages...")
    global_temp_by_year = {}

    # Extract values from predictions
    predictions_df = pd.DataFrame(predictions)
    for year, year_data in predictions_df.groupby('year'):
        global_temp_by_year[year] = year_data['temp_anomaly'].mean()

    # Get historical global averages
    historical_by_year = temp_long.groupby(
        'year')['temp_anomaly'].mean().to_dict()
    for year, avg in historical_by_year.items():
        global_temp_by_year[year] = avg

    historical_global_data = temp_long.groupby('year').agg({
        'temperature': 'mean',
        'temp_anomaly': 'mean'
    }).reset_index()

    # Get the actual baseline temp from the baseline period
    baseline_global_temp = historical_global_data[
        (historical_global_data['year'] >= BASELINE_START_YEAR) &
        (historical_global_data['year'] <= BASELINE_END_YEAR)
    ]['temperature'].mean()

    # Add global predictions
    global_predictions = []
    for year, avg_anomaly in global_temp_by_year.items():
        is_prediction = year >= 2024
        global_predictions.append({
            'code': 'GLOBAL',
            'country': 'Global Average',
            'year': year,
            'month': 7,  # Use July as reference
            'temperature': baseline_global_temp + avg_anomaly,  # Assume 14°C global average
            'temp_anomaly': avg_anomaly,
            'precipitation': None,
            'prec_percent': None,
            'model': 'global_average' if is_prediction else '',
            'is_prediction': is_prediction,
            'prediction_confidence': 'medium' if is_prediction else None,
            'global_avg': avg_anomaly + 0.7  # Add ~0.7°C for pre-industrial adjustment
        })

    # Add global predictions
    predictions.extend(global_predictions)

    # Create final DataFrames
    predictions_df = pd.DataFrame(predictions)
    performance_df = pd.DataFrame(model_performance)

    # Save final results
    predictions_df.to_csv(os.path.join(
        OUTPUT_DIR, 'final_predictions.csv'), index=False)
    performance_df.to_csv(os.path.join(
        OUTPUT_DIR, 'final_performance.csv'), index=False)

    return predictions_df, performance_df


def combine_historical_and_predictions(temp_long, prec_long, predictions_df):
    """Combine historical data with predictions for visualization"""
    print("Combining historical and prediction data...")

    # Get historical temperature data
    historical_temp = temp_long[['code', 'name', 'year',
                                 'month', 'temperature', 'temp_anomaly']].copy()
    historical_temp['is_prediction'] = False
    historical_temp['model'] = ''
    historical_temp['prediction_confidence'] = None
    historical_temp['global_avg'] = None
    historical_temp['interval_width'] = None
    historical_temp['temp_lower'] = None
    historical_temp['temp_upper'] = None

    # Calculate historical global averages
    historical_global = historical_temp.groupby(
        ['year', 'month'])['temp_anomaly'].mean().reset_index()
    historical_global['code'] = 'GLOBAL'
    historical_global['name'] = 'Global Average'
    historical_global['temperature'] = 14.0 + historical_global['temp_anomaly']
    historical_global['is_prediction'] = False
    historical_global['model'] = 'historical_average'
    historical_global['prediction_confidence'] = None
    historical_global['global_avg'] = historical_global['temp_anomaly'] + 0.7
    historical_global['interval_width'] = None

    # Add global historical data
    historical_temp = pd.concat(
        [historical_temp, historical_global], ignore_index=True)

    # Get historical precipitation data (we don't predict this)
    historical_prec = prec_long[['code', 'name', 'year',
                                 'month', 'precipitation', 'prec_percent']].copy()

    # Rename 'name' column to 'country' to match predictions
    historical_temp = historical_temp.rename(columns={'name': 'country'})
    historical_prec = historical_prec.rename(columns={'name': 'country'})

    # Ensure no duplicates
    historical_temp = historical_temp.reset_index(drop=True)
    historical_prec = historical_prec.reset_index(drop=True)

    # If no predictions, use only historical
    if predictions_df.empty:
        print("Warning: No predictions available. Using only historical data.")
        all_temp = historical_temp
        all_prec = historical_prec
    else:

        # Select only the columns we want
        # Keep only needed columns from predictions
        keep_columns = ['code', 'country', 'year', 'month', 'temperature',
                        'temp_anomaly', 'temp_lower', 'temp_upper', 'interval_width',
                        'is_prediction', 'model', 'prediction_confidence', 'global_avg']

        # Ensure predictions_df has all required columns
        for col in keep_columns:
            if col not in predictions_df.columns:
                predictions_df[col] = None

        # Keep only needed columns from predictions
        predictions_temp = predictions_df[keep_columns].copy()

        # Combine historical with predictions
        all_temp = pd.concat([
            historical_temp[keep_columns],
            predictions_temp
        ], ignore_index=True)

        # For precipitation, we only use historical data
        all_prec = historical_prec.copy()

        # Calculate annual averages
        annual_temp = all_temp.groupby(['code', 'country', 'year']).agg({
            'temperature': 'mean',
            'temp_anomaly': 'mean',
            'temp_lower': 'mean',  # Add these
            'temp_upper': 'mean',  # Add these
            'interval_width': 'mean',
            'is_prediction': 'max',
            'model': lambda x: x.mode().iloc[0] if not x.isna().all() else '',
            'prediction_confidence': lambda x: x.mode().iloc[0] if not x.isna().all() else None,
            'global_avg': 'mean'
        }).reset_index()

    # Precipitation annual averages
    annual_prec = all_prec.groupby(['code', 'country', 'year']).agg({
        'precipitation': 'mean',
        'prec_percent': 'mean'
    }).reset_index()

    # Merge temperature and precipitation data
    climate_annual = pd.merge(
        annual_temp,
        annual_prec[['code', 'year', 'precipitation', 'prec_percent']],
        on=['code', 'year'],
        how='left'
    )

    print(
        f"Rows after merge: {len(climate_annual)}, Countries: {climate_annual['code'].nunique()}")

    # Fill any NaN values
    climate_annual['precipitation'] = climate_annual['precipitation'].fillna(0)
    climate_annual['prec_percent'] = climate_annual['prec_percent'].fillna(100)

    # Use temp_anomaly as the map score
    climate_annual['score'] = climate_annual['temp_anomaly']

    print(
        f"Combined data: {len(climate_annual)} rows, {climate_annual['code'].nunique()} countries")
    print(
        f"Years: {climate_annual['year'].min()} to {climate_annual['year'].max()}")

    print(
        f"Final validation: {len(climate_annual)} rows, {climate_annual['code'].nunique()} countries")
    print(
        f"Years: {climate_annual['year'].min()} to {climate_annual['year'].max()}")

    all_years = set(
        range(climate_annual['year'].min(), climate_annual['year'].max() + 1))
    missing_years = all_years - set(climate_annual['year'].unique())
    if missing_years:
        print(f"WARNING: Missing years: {missing_years}")

    return climate_annual

# %%


def main():

    backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(OUTPUT_DIR, f"backup_{backup_timestamp}")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    for file in ['final_predictions.csv', 'model_performance.csv', 'climate_with_predictions.csv']:
        src = os.path.join(OUTPUT_DIR, file)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, file))
            print(f"Backup: {file} to {backup_dir}")

    final_predictions_file = os.path.join(OUTPUT_DIR, 'final_predictions.csv')

    if os.path.exists(final_predictions_file):
        final_preds = pd.read_csv(final_predictions_file)

        # Also check if intermediate files have more recent data
        if os.path.exists(os.path.join(OUTPUT_DIR, 'predictions_in_progress.csv')):
            in_progress = pd.read_csv(os.path.join(
                OUTPUT_DIR, 'predictions_in_progress.csv'))
            if len(in_progress) > len(final_preds):
                print(
                    f"Found more data in predictions_in_progress.csv ({len(in_progress)} rows vs {len(final_preds)}")
                # Use the more complete version
                final_preds = in_progress
                final_preds.to_csv(final_predictions_file, index=False)

        processed_countries = final_preds['code'].unique().tolist()
        print(f"Restored {len(processed_countries)} countries")

    # Set paths to data files
    temp_file = "climate_temp.xlsx"
    prec_file = "climate_prec.xlsx"

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load and process data
    temp_long, prec_long = load_and_process_data(temp_file, prec_file)

    # Calculate climate anomalies
    temp_long, prec_long = calculate_climate_anomalies(temp_long, prec_long)

    # Handle missing values
    temp_long = handle_missing_values(temp_long)
    prec_long = handle_missing_values(prec_long)

    # Generate predictions (only temperature, keep precipitation historical)
    predictions_df, performance_df = generate_climate_predictions(
        temp_long, prec_long)

    # Combine historical and prediction data
    visualization_data = combine_historical_and_predictions(
        temp_long, prec_long, predictions_df)

    # Calculate statistical confidence levels
    predictions_df = calculate_statistical_confidence(
        predictions_df, temp_long, performance_df)

    # Save results
    output_file = os.path.join(OUTPUT_DIR, 'climate_with_predictions.csv')
    performance_file = os.path.join(OUTPUT_DIR, 'model_performance.csv')

    visualization_data.to_csv(output_file, index=False)
    performance_df.to_csv(performance_file, index=False)

    print("\nDone! Generated files:")
    print(f"- {output_file} - Ready for visualization")
    print(f"- {performance_file} - Model evaluation metrics")

    # Show model performance summary
    if not performance_df.empty:
        temp_models = performance_df[performance_df['variable']
                                     == 'temperature']['best_model'].value_counts()
        print("\nBest temperature models:")
        for model, count in temp_models.items():
            print(f"  {model}: {count} countries")


if __name__ == "__main__":
    main()
