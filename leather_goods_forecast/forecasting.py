import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import statsmodels.api as sm
import holidays

class LeatherGoodsForecast:
    """
    A forecasting class that utilizes multiple models (Prophet, Random Forest, SARIMA)
    to predict future demand for leather goods. The approach combines classical time series models
    with machine learning for improved accuracy.
    """

    def __init__(self, country_code='IT'):
        """
        Initialize the LeatherGoodsForecast instance.

        Args:
            country_code (str): ISO code of the country to source holidays from.
        """
        self.scaler = StandardScaler()
        self.prophet_model = None
        self.rf_model = None
        self.sarima_model = None
        self.eu_holidays = holidays.CountryHoliday(country_code)
        self.historical_errors_ = None

    def prepare_features(self, df):
        """
        Prepare features for the model including:
        - Time-based features (month, quarter, year, day_of_week)
        - Holiday flags
        - Fourier terms for seasonality

        Args:
            df (pd.DataFrame): DataFrame with at least a 'date' column and 'demand'.

        Returns:
            pd.DataFrame: Enhanced DataFrame with new feature columns.
        """
        df = df.copy()
        if 'date' not in df.columns:
            raise ValueError("DataFrame must contain a 'date' column.")

        df['date'] = pd.to_datetime(df['date'])

        # Extract temporal features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear

        # Create holiday flags
        df['is_holiday'] = df['date'].apply(lambda x: x in self.eu_holidays)

        # Create Fourier terms for seasonality
        # Periods: annual (~365.25 days), semi-annual (~182.625 days), quarterly (~91.3125 days)
        for period in [365.25, 182.625, 91.3125]:
            for n in range(1, 3):
                df[f'sin_{int(period)}_{n}'] = np.sin(2 * n * np.pi * df['day_of_year'] / period)
                df[f'cos_{int(period)}_{n}'] = np.cos(2 * n * np.pi * df['day_of_year'] / period)

        return df

    def _generate_holiday_df(self, start_date, end_date):
        """
        Generate a DataFrame of holidays for Prophet from the holidays library.

        Args:
            start_date (pd.Timestamp): Start date of the historical data range.
            end_date (pd.Timestamp): End date of the historical data range.

        Returns:
            pd.DataFrame: DataFrame with columns ['ds', 'holiday'] suitable for Prophet.
        """
        # Convert timestamps to date for comparison with holiday dates (datetime.date)
        start_date = start_date.date()
        end_date = end_date.date()
        holiday_dates = [d for d in self.eu_holidays if start_date <= d <= end_date]

        holiday_df = pd.DataFrame({
            'ds': pd.to_datetime(holiday_dates),
            'holiday': 'EU_holiday'
        })
        return holiday_df

    def train_prophet(self, df):
        """
        Train a Prophet model for baseline forecasting.

        Args:
            df (pd.DataFrame): Must contain ['date', 'demand'].
        """
        if 'demand' not in df.columns:
            raise ValueError("DataFrame must contain a 'demand' column.")

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
        """
        Train a Random Forest model using prepared features.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable (demand).
        """
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.rf_model.fit(X, y)

    def train_sarima(self, df):
        """
        Train a SARIMA model for time series forecasting.

        Args:
            df (pd.DataFrame): DataFrame with 'demand' column.
        """
        if 'demand' not in df.columns:
            raise ValueError("DataFrame must contain a 'demand' column.")

        self.sarima_model = sm.tsa.statespace.SARIMAX(
            df['demand'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

    def ensemble_forecast(self, future_dates, features):
        """
        Create an ensemble forecast combining Prophet, Random Forest, and SARIMA predictions.

        Args:
            future_dates (array-like): Future dates for forecasting.
            features (pd.DataFrame): Feature matrix for future dates.

        Returns:
            np.ndarray: Ensemble forecast array.
        """
        # Prophet forecast
        prophet_future = pd.DataFrame({'ds': future_dates})
        prophet_forecast = self.prophet_model.predict(prophet_future)['yhat'].values

        # Random Forest forecast
        rf_forecast = self.rf_model.predict(features)

        # SARIMA forecast
        sarima_forecast = self.sarima_model.forecast(len(future_dates)).values

        # Weighted ensemble
        ensemble_forecast = (
            0.4 * prophet_forecast +
            0.3 * rf_forecast +
            0.3 * sarima_forecast
        )
        return ensemble_forecast

    def calculate_uncertainty(self, predictions, historical_errors):
        """
        Calculate prediction intervals based on historical errors.

        Args:
            predictions (np.ndarray): Mean predictions.
            historical_errors (np.ndarray): Array of historical forecast errors.

        Returns:
            (np.ndarray, np.ndarray): lower_bound and upper_bound arrays.
        """
        if historical_errors is None or len(historical_errors) < 2:
            # Inadequate data to compute std, return dummy bounds
            lower_bound = predictions * 0.9
            upper_bound = predictions * 1.1
        else:
            error_std = np.std(historical_errors)
            lower_bound = predictions - 1.96 * error_std
            upper_bound = predictions + 1.96 * error_std
        return lower_bound, upper_bound

    def fit(self, historical_data, validation_size=12):
        """
        Train all models using historical data and compute historical errors via a validation set.

        Args:
            historical_data (pd.DataFrame): Must contain ['date', 'demand'].
            validation_size (int): Number of validation periods.

        Returns:
            LeatherGoodsForecast: self
        """
        if 'date' not in historical_data.columns or 'demand' not in historical_data.columns:
            raise ValueError("historical_data must contain 'date' and 'demand' columns.")

        df = self.prepare_features(historical_data)
        df = df.sort_values('date')

        if len(df) <= validation_size:
            raise ValueError("Not enough data for the requested validation size.")

        train_df = df.iloc[:-validation_size].copy()
        val_df = df.iloc[-validation_size:].copy()

        # Train Prophet
        self.train_prophet(train_df)

        # Train Random Forest
        feature_cols = [col for col in df.columns if col not in ['date', 'demand']]
        X_train = train_df[feature_cols]
        y_train = train_df['demand']
        self.train_random_forest(X_train, y_train)

        # Train SARIMA
        self.train_sarima(train_df)

        # Forecast on validation set to get errors
        val_features = val_df[feature_cols]
        val_dates = val_df['date']
        val_pred = self.ensemble_forecast(val_dates, val_features)
        val_actual = val_df['demand'].values

        # Store historical errors
        self.historical_errors_ = val_pred - val_actual
        return self

    def get_historical_errors(self):
        """
        Return stored historical forecast errors from the validation set.

        Returns:
            np.ndarray: Array of historical errors.
        """
        return self.historical_errors_

    def predict(self, future_dates):
        """
        Generate forecasts for future dates using the ensemble of models.

        Args:
            future_dates (pd.DatetimeIndex): Future dates to forecast.

        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'forecast', 'lower_bound', 'upper_bound'].
        """
        future_df = pd.DataFrame({'date': future_dates})
        future_features = self.prepare_features(future_df)
        feature_cols = [col for col in future_features.columns if col not in ['date', 'demand']]

        predictions = self.ensemble_forecast(future_dates, future_features[feature_cols])
        historical_errors = self.get_historical_errors()
        lower_bound, upper_bound = self.calculate_uncertainty(predictions, historical_errors)

        return pd.DataFrame({
            'date': future_dates,
            'forecast': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
