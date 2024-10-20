import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.linear_model import LogisticRegression
import os

# Gапку для изображений
if not os.path.exists('images'):
    os.makedirs('images')

def main():
    # 1. Считывание и преобработка исходного ряда
    data_frame = pd.read_excel('Данные v2.xlsx', sheet_name='Бр_дневка - 3 (основной)', dtype={'дата': str, 'направление': str, 'выход': float})
    data_frame['дата'] = pd.to_datetime(data_frame['дата'], dayfirst=True)
    sorted_data_frame = data_frame.sort_values(by='дата', ascending=True).reset_index(drop=True)

    # График временного ряда
    plt.figure(figsize=(14, 6))
    plt.plot(sorted_data_frame['дата'], sorted_data_frame['выход'])
    plt.title('Временной ряд')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.savefig('images/1.time_series.png')
    plt.close()

    # 1.2. Сглаживание данных
    smoothed_values = sorted_data_frame['выход'].rolling(window=30, center=True).mean().fillna(sorted_data_frame['выход'])
    
    plt.figure(figsize=(14, 6))
    plt.plot(sorted_data_frame['дата'], smoothed_values)
    plt.title('Сглаженные данные')
    plt.xlabel('Дата')
    plt.ylabel('Сглаженное значение')
    plt.grid(True)
    plt.savefig('images/2.smoothed_data.png')
    plt.close()

    # 1.3. Выделение компоненты тренда
    time_index = np.arange(len(smoothed_values)).reshape(-1, 1)
    smoothed_y_values = smoothed_values.values
    polynomial_features = PolynomialFeatures(degree=6)
    time_index_poly = polynomial_features.fit_transform(time_index)
    
    polynomial_model = LinearRegression().fit(time_index_poly, smoothed_y_values)
    trend_values = polynomial_model.predict(time_index_poly)
    
    plt.figure(figsize=(14, 6))
    plt.plot(sorted_data_frame['дата'], smoothed_values, label='Сглаженный ряд', alpha=0.5)
    plt.plot(sorted_data_frame['дата'], trend_values, label='Тренд', color='red')
    plt.title('Выделение тренда из сглаженного ряда')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/3.trend_extraction.png')
    plt.close()

    # 1.4. Удаление тренда из сглаженных данных
    detrended_smoothed_data = smoothed_values - trend_values
    
    plt.figure(figsize=(14, 6))
    plt.plot(sorted_data_frame['дата'], detrended_smoothed_data)
    plt.title('Сглаженные данные без тренда')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.savefig('images/4.smoothed_without_trend.png')
    plt.close()

    # 2. Прогноз
    # 2.1. Прогноз тренда
    forecast_dates_df = pd.read_excel('Данные v2.xlsx', sheet_name='Прогноз')
    forecast_dates_df['дата'] = pd.to_datetime(forecast_dates_df['дата'], dayfirst=True)
    
    future_dates_list = forecast_dates_df['дата']

    future_time_index = np.arange(len(detrended_smoothed_data) + len(future_dates_list)).reshape(-1, 1)
    future_time_index_poly = polynomial_features.transform(future_time_index)
    
    future_trend_values = polynomial_model.predict(future_time_index_poly)

    trend_forecast_df = pd.DataFrame({
        'дата': future_dates_list,
        'forecast': future_trend_values[-len(future_dates_list):]  # Только последние прогнозы, соответствующие будущим датам
    })

    plt.figure(figsize=(14, 6))
    plt.plot(sorted_data_frame['дата'], trend_values, label='Исходная компонента тренда', color='blue')
    plt.plot(trend_forecast_df['дата'], trend_forecast_df['forecast'], label='Предсказанная компонента тренда', color='red')
    plt.title('Прогноз компоненты тренда')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/5.trend_forecast.png')
    plt.close()

    # 2.2. Прогноз оставшегося ряда
    # Используется SARIMA, которая учитывает сезонность данных, поэтому сезонную компоненту выделять не нужно.
    sarima_model = auto_arima(detrended_smoothed_data,
                                seasonal=True,  # Учитываем сезонность
                                m=4,           # Период сезонности
                                stepwise=True,  # Используем пошаговый поиск
                                trace=True,     # Выводить прогресс
                                suppress_warnings=True,  # Подавлять предупреждения
                                max_p=3, max_q=3, max_P=2, max_Q=2,  # Ограничения на поиск
                                d=1, D=1)       # Дифференцирование

    print(f"Лучшие параметры SARIMA: {sarima_model.order} x {sarima_model.seasonal_order}")

    fitted_sarima_model = SARIMAX(detrended_smoothed_data,
                                    order=(0,1,0),
                                    seasonal_order=(2, 1, 0, 4)).fit()

    sarima_forecast_values = fitted_sarima_model.get_forecast(steps=len(future_dates_list)).predicted_mean

    plt.figure(figsize=(14, 6))
    plt.plot(sorted_data_frame['дата'], detrended_smoothed_data, label='Исходные данные без тренда', color='blue')
    plt.plot(future_dates_list, sarima_forecast_values.values, label='Предсказания SARIMA', color='red')
    plt.title('SARIMA')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/6.forecast_sarima.png')
    plt.close()

    # 2.3. Объединение предсказаний
    combined_forecast_sarima_values = sarima_forecast_values.values + trend_forecast_df['forecast']

    plt.figure(figsize=(14, 6))
    plt.plot(sorted_data_frame['дата'], sorted_data_frame['выход'], label='Исходный ряд', color='blue')
    plt.plot(future_dates_list, combined_forecast_sarima_values, label='Предсказание', color='red')
    plt.title('Полное предсказание')
    plt.xlabel('Данные')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.legend()
    plt.savefig('images/7.full_forecast.png')
    plt.close()

    # 3. Классификация
    merged_dataframe = pd.concat([sorted_data_frame, forecast_dates_df], ignore_index=True)
    merged_dataframe['change'] = merged_dataframe['выход'].diff()  # Изменение
    merged_dataframe['percent_change'] = merged_dataframe['выход'].pct_change(fill_method=None)  # Процентное изменение
    merged_dataframe['rolling_mean_5'] = merged_dataframe['выход'].rolling(window=5).mean()  # Скользящее среднее на 5 шагов
    merged_dataframe['rolling_std_5'] = merged_dataframe['выход'].rolling(window=5).std()  # Скользящее стандартное отклонение

    merged_dataframe = merged_dataframe.dropna()

    merged_dataframe['direction_label'] = merged_dataframe['направление'].apply(lambda x: 1 if x == 'л' else 0)
    training_size = len(merged_dataframe) - len(future_dates_list)
    training_data = merged_dataframe.iloc[:training_size]  # Данные для обучения (исходный ряд)
    future_prediction_data = merged_dataframe.iloc[training_size:]  # Данные для прогноза (будущие даты)

    X_train_features = training_data[['change', 'percent_change', 'rolling_mean_5', 'rolling_std_5']]
    y_train_labels = training_data['direction_label']
    X_future_features = future_prediction_data[['change', 'percent_change', 'rolling_mean_5', 'rolling_std_5']]

    logistic_regression_model = LogisticRegression(C=0.1, max_iter=100, penalty='l2', solver='liblinear')
    logistic_regression_model.fit(X_train_features, y_train_labels)  
    future_direction_predictions = logistic_regression_model.predict(X_future_features)

    # 4. Сохранение результатов в JSON файлы.
    with open('forecast_value.json', 'w') as file:
        json.dump(combined_forecast_sarima_values.tolist(), file)

    with open('forecast_class.json', 'w') as file:
        json.dump(future_direction_predictions.tolist(), file)

if __name__ == "__main__":
    main()