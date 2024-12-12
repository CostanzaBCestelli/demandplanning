import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import statsmodels.api as sm
import holidays

class LeatherGoodsForecast:
    def __init__(self, country_code='IT'):
        self.scaler = StandardScaler()
        self.prophet_model = None
        self.rf_model = None
        self.sarima_model = None
        self.eu_holidays = holidays.CountryHoliday(country_code)
        self.historical_errors_ = None
    
    def prepare_features(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_holiday'] = df['date'].apply(lambda x: x in self.eu_holidays)
        for period in [365.25, 182.625, 91.3125]:
            for n in range(1, 3):
                df[f'sin_{int(period)}_{n}'] = np.sin(2 * n * np.pi * df['day_of_year'] / period)
                df[f'cos_{int(period)}_{n}'] = np.cos(2 * n * np.pi * df['day_of_year'] / period)
        return df

   def _generate_holiday_df(self, start_date, end_date):
    # Convert Timestamps to date
    start_date = start_date.date()
    end_date = end_date.date()
    holiday_dates = [d for d in self.eu_holidays if start_date <= d <= end_date]
    holiday_df = pd.DataFrame({
        'ds': pd.to_datetime(holiday_dates),
        'holiday': 'EU_holiday'
    })
    return holiday_df


    def train_prophet(self, df):
        prophet_df = df[['date', 'demand']].copy()
        prophet_df.columns = ['ds', 'y']
        holiday_df = self._generate_holiday_df(prophet_df['ds'].min(), prophet_df['ds'].max())
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holiday_df,
            seasonality_mode='multiplicative'
        )
        self.prophet_model.fit(prophet_df)

    def train_random_forest(self, X, y):
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.rf_model.fit(X, y)

    def train_sarima(self, df):
        self.sarima_model = sm.tsa.statespace.SARIMAX(
            df['demand'],
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

    def ensemble_forecast(self, future_dates, features):
        prophet_future = pd.DataFrame({'ds': future_dates})
        prophet_forecast = self.prophet_model.predict(prophet_future)['yhat'].values
        rf_forecast = self.rf_model.predict(features)
        sarima_forecast = self.sarima_model.forecast(len(future_dates)).values
        ensemble_forecast = 0.4*prophet_forecast + 0.3*rf_forecast + 0.3*sarima_forecast
        return ensemble_forecast

    def calculate_uncertainty(self, predictions, historical_errors):
        if historical_errors is None or len(historical_errors)<2:
            lower_bound = predictions*0.9
            upper_bound = predictions*1.1
        else:
            error_std = np.std(historical_errors)
            lower_bound = predictions - 1.96*error_std
            upper_bound = predictions + 1.96*error_std
        return lower_bound, upper_bound

    def fit(self, historical_data, validation_size=12):
        if 'date' not in historical_data.columns or 'demand' not in historical_data.columns:
            raise ValueError("historical_data must contain 'date' and 'demand' columns.")
        df = self.prepare_features(historical_data).sort_values('date')
        if len(df)<=validation_size:
            raise ValueError("Not enough data for requested validation size.")
        train_df = df.iloc[:-validation_size].copy()
        val_df = df.iloc[-validation_size:].copy()
        self.train_prophet(train_df)
        feature_cols = [c for c in df.columns if c not in ['date','demand']]
        X_train = train_df[feature_cols]
        y_train = train_df['demand']
        self.train_random_forest(X_train, y_train)
        self.train_sarima(train_df)
        val_features = val_df[feature_cols]
        val_dates = val_df['date']
        val_pred = self.ensemble_forecast(val_dates, val_features)
        val_actual = val_df['demand'].values
        self.historical_errors_ = val_pred - val_actual
        return self

    def get_historical_errors(self):
        return self.historical_errors_

    def predict(self, future_dates):
        future_df = pd.DataFrame({'date': future_dates})
        future_features = self.prepare_features(future_df)
        feature_cols = [c for c in future_features.columns if c not in ['date','demand']]
        predictions = self.ensemble_forecast(future_dates, future_features[feature_cols])
        historical_errors = self.get_historical_errors()
        lower_bound, upper_bound = self.calculate_uncertainty(predictions, historical_errors)
        return pd.DataFrame({
            'date': future_dates,
            'forecast': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
