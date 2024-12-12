import pandas as pd
import numpy as np
from leather_goods_forecast.forecasting import LeatherGoodsForecast

def test_model_fit_and_predict():
    dates = pd.date_range('2020-01-01', periods=36, freq='M')
    demand = np.random.randint(50,150,len(dates))
    historical_data = pd.DataFrame({'date':dates, 'demand':demand})
    model = LeatherGoodsForecast()
    model.fit(historical_data, validation_size=6)
    
    future_dates = pd.date_range(start=dates.max()+pd.offsets.MonthBegin(), periods=3, freq='M')
    forecasts = model.predict(future_dates)
    
    assert len(forecasts) == 3
    assert all(col in forecasts.columns for col in ['date','forecast','lower_bound','upper_bound'])
    assert not forecasts['forecast'].isnull().any()
