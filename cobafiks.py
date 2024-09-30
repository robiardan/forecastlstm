import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go

# Fungsi untuk menghitung interval kepercayaan menggunakan bootstrap
def calculate_confidence_interval(model, last_data, num_periods, scaler, n_bootstrap=100):
    forecasts = []

    for _ in range(n_bootstrap):
        noisy_data = last_data + np.random.normal(0, 0.01, last_data.shape)
        forecast = []
        data = noisy_data.copy()

        for _ in range(num_periods):
            next_pred = model.predict(np.expand_dims(data, axis=0))
            forecast.append(next_pred[0, 0])
            data = np.roll(data, -1, axis=0)
            data[-1] = next_pred[0, 0]

        forecasts.append(np.array(forecast))

    forecasts = np.array(forecasts)
    mean_forecast = np.mean(forecasts, axis=0)
    lower_bound = np.percentile(forecasts, 2.5, axis=0)
    upper_bound = np.percentile(forecasts, 97.5, axis=0)

    return mean_forecast, lower_bound, upper_bound

# Sidebar Navigation
menu = st.sidebar.selectbox("Menu", ["Home","CPO Analysis", "Upload Data"])

if menu == "Home":
    st.markdown(
        """
        <style>
        /* CSS untuk menutupi seluruh halaman termasuk sidebar */
        .css-1d391kg {
            background: rgba(0, 0, 0, 0); /* Transparent background to ensure parallax effect */
            margin: 0;
            padding: 0;
            height: 100vh; /* Ensure full height */
            position: relative;
            overflow: hidden;
        }

        /* CSS untuk area sidebar */
        .css-1d391kg .css-1v3fvcr {
            background: rgba(0, 0, 0, 0.5); /* Semi-transparent sidebar */
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
            z-index: 1; /* Ensure it’s above the parallax background */
        }

        /* CSS untuk area konten */
        .css-1d391kg .css-ffhzg2 {
            background: rgba(0, 0, 0, 0.5); /* Semi-transparent overlay on content */
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
            z-index: 2; /* Ensure it’s above the parallax background */
        }

        /* Parallax effect CSS */
        .parallax {
            height: 100vh;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            z-index: -1; /* Ensure it’s behind everything else */
        }

        .content {
            color: white;
            text-align: center;
            padding: 150px;
            font-size: 24px;
            position: relative;
            z-index: 2; /* Ensure content is above the background */
        }
        </style>

        <div class="parallax"></div>
        <div class="content">
            <h1>Welcome to the CPO Production Prediction App</h1>
            <p>Use the menu on the left to explore more features.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif menu == "CPO Analysis":
    # Judul aplikasi
    st.title('CPO Production Prediction')

    # Baca data dari file lokal (ubah path sesuai lokasi file Anda)
    file_path = 'Data TA.xlsx'  # Ganti dengan path ke file Excel Anda
    try:
        df = pd.read_excel(file_path)
        df = df.drop(['Date'], axis=1)  # Drop non-numeric columns

        # Pastikan data memiliki kolom 'Target' untuk peramalan
        if 'Target' not in df.columns:
            st.error("File Excel harus memiliki kolom 'Target'")
        else:
            if df.empty:
                st.error("DataFrame kosong. Pastikan file Excel memiliki data.")
            else:
                # Deskripsi Data
                st.subheader('Data Description')
                st.write(df.describe())

                # Time Series Plot dengan Plotly
                st.subheader('CPO Production Time Series')
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df['Target'], mode='lines', name='CPO Production', line=dict(color='blue')))
                fig.update_layout(title='CPO Production Time Series',
                                  xaxis_title='Time',
                                  yaxis_title='Production')
                st.plotly_chart(fig)

                # Normalisasi data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df)

                # Pembagian data menjadi pelatihan dan pengujian
                training_size = int(len(scaled_data) * 0.8)
                train_data = scaled_data[:training_size]
                test_data = scaled_data[training_size:]

                # Mempersiapkan data dengan window size (timesteps)
                def create_dataset(dataset, time_step=6):
                    x, y = [], []
                    for i in range(time_step, len(dataset)):
                        x.append(dataset[i-time_step:i, 0])
                        y.append(dataset[i, 0])
                    return np.array(x), np.array(y)

                time_step = 6
                x_train, y_train = create_dataset(train_data, time_step)
                x_test, y_test = create_dataset(test_data, time_step)

                # Reshape data agar sesuai dengan input LSTM
                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

                # Load model
                try:
                    best_model = tf.keras.models.load_model('best_model_w6.h5', custom_objects={'DTypePolicy': tf.keras.mixed_precision.Policy})
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.stop()

                # Prediksi pada data pengujian
                y_test_pred = best_model.predict(x_test).flatten()

                # Denormalisasi data aktual dan prediksi
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                y_test_pred_denormalized = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

                # Hitung MAPE untuk data pengujian setelah denormalisasi
                test_mape_denormalized = mean_absolute_percentage_error(y_test_actual, y_test_pred_denormalized)
                st.write(f"Test MAPE (Denormalized): {test_mape_denormalized*100:.2f}%")

                # Plot LSTM Predictions vs Original dengan Plotly
                st.subheader('LSTM Predictions vs Original')
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=np.arange(len(y_test_actual)), y=y_test_actual, mode='lines', name='Original Production', line=dict(color='blue')))
                fig2.add_trace(go.Scatter(x=np.arange(len(y_test_pred_denormalized)), y=y_test_pred_denormalized.flatten(), mode='lines', name='Predicted Production', line=dict(color='red')))
                fig2.update_layout(title='LSTM Predictions vs Original Production',
                                   xaxis_title='Time',
                                   yaxis_title='Production')
                st.plotly_chart(fig2)

                # Forecasting beberapa periode ke depan
                forecast_periods = st.slider('Select number of future periods to predict:', min_value=1, max_value=24, value=6, step=1)

                # Ambil data terakhir dari data uji untuk memulai forecasting
                last_sequence = x_test[-1]  # Data terakhir dari x_test
                forecast_results = []

                # Lakukan forecasting
                for _ in range(forecast_periods):
                    # Prediksi untuk periode berikutnya
                    next_value = best_model.predict(last_sequence.reshape(1, time_step, 1)).flatten()[0]
                    forecast_results.append(next_value)

                    # Update last_sequence untuk prediksi berikutnya
                    last_sequence = np.append(last_sequence[1:], next_value)

                # Denormalisasi hasil forecast
                forecast_results_denormalized = scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1)).flatten()

                # Plot hasil forecast dengan Plotly
                fig3 = go.Figure()

                # Actual values
                fig3.add_trace(go.Scatter(
                    x=np.arange(len(y_test_actual)),
                    y=y_test_actual,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                ))

                # Predicted values
                fig3.add_trace(go.Scatter(
                    x=np.arange(len(y_test_pred_denormalized)),
                    y=y_test_pred_denormalized,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='orange')
                ))

                # Forecasted values
                fig3.add_trace(go.Scatter(
                    x=np.arange(len(y_test_actual), len(y_test_actual) + forecast_periods),
                    y=forecast_results_denormalized,
                    mode='lines',
                    name='Forecasted',
                    line=dict(color='red')
                ))

                fig3.update_layout(
                    title='Actual, Predicted, and Forecasted CPO Production',
                    xaxis_title='Time Steps',
                    yaxis_title='CPO Production'
                )

                st.plotly_chart(fig3)


                st.write(f"Forecasted values for the next {forecast_periods} periods:")
                st.write(forecast_results_denormalized)

    except Exception as e:
        st.error(f"An error occurred: {e}")


elif menu == "Upload Data":
    # Judul aplikasi
    st.title('Your Data Prediction')

    # Upload file Excel
    uploaded_file = st.file_uploader("Choose an Excel file. (rename kolom menjadi Target)", type="xlsx")
    if uploaded_file is not None:
        try:
            # Membaca file Excel
            df = pd.read_excel(uploaded_file)
            df = df.drop(['Date'], axis=1)  # Drop non-numeric columns
            # Pastikan data memiliki kolom 'Target' untuk peramalan
            if 'Target' not in df.columns:
                st.error("File Excel harus memiliki kolom 'Target'")
            else:
                if df.empty:
                    st.error("DataFrame kosong. Pastikan file Excel memiliki data.")
                else:
                    # Deskripsi Data
                    st.subheader('Data Description')
                    st.write(df.describe())

                    # Time Series Plot dengan Plotly
                    st.subheader('Time Series Plot')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=df['Target'], mode='lines', name='CPO Production', line=dict(color='blue')))
                    fig.update_layout(title='Time Series Plot',
                                    xaxis_title='Time',
                                    yaxis_title='Target')
                    st.plotly_chart(fig)

                    # Normalisasi data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(df)

                    # Pembagian data menjadi pelatihan dan pengujian
                    training_size = int(len(scaled_data) * 0.8)
                    train_data = scaled_data[:training_size]
                    test_data = scaled_data[training_size:]

                    # Mempersiapkan data dengan window size (timesteps)
                    def create_dataset(dataset, time_step=6):
                        x, y = [], []
                        for i in range(time_step, len(dataset)):
                            x.append(dataset[i-time_step:i, 0])
                            y.append(dataset[i, 0])
                        return np.array(x), np.array(y)

                    time_step = 6
                    x_train, y_train = create_dataset(train_data, time_step)
                    x_test, y_test = create_dataset(test_data, time_step)

                    # Reshape data agar sesuai dengan input LSTM
                    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

                    # Load model
                    try:
                        best_model = tf.keras.models.load_model('best_model_w6.h5', custom_objects={'DTypePolicy': tf.keras.mixed_precision.Policy})
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        st.stop()

                    # Prediksi pada data pengujian
                    y_test_pred = best_model.predict(x_test).flatten()

                    # Denormalisasi data aktual dan prediksi
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_test_pred_denormalized = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

                    # Hitung MAPE untuk data pengujian setelah denormalisasi
                    test_mape_denormalized = mean_absolute_percentage_error(y_test_actual, y_test_pred_denormalized)
                    st.write(f"Test MAPE (Denormalized): {test_mape_denormalized*100:.2f}%")

                    # Plot LSTM Predictions vs Original dengan Plotly
                    st.subheader('LSTM Predictions vs Original')
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=np.arange(len(y_test_actual)), y=y_test_actual, mode='lines', name='Original Production', line=dict(color='blue')))
                    fig2.add_trace(go.Scatter(x=np.arange(len(y_test_pred_denormalized)), y=y_test_pred_denormalized.flatten(), mode='lines', name='Predicted Production', line=dict(color='red')))
                    fig2.update_layout(title='LSTM Predictions vs Original Production',
                                    xaxis_title='Time',
                                    yaxis_title='Production')
                    st.plotly_chart(fig2)

                    # Forecasting beberapa periode ke depan
                    forecast_periods = st.slider('Select number of future periods to predict:', min_value=1, max_value=24, value=6, step=1)

                    # Ambil data terakhir dari data uji untuk memulai forecasting
                    last_sequence = x_test[-1]  # Data terakhir dari x_test
                    forecast_results = []

                    # Lakukan forecasting
                    for _ in range(forecast_periods):
                        # Prediksi untuk periode berikutnya
                        next_value = best_model.predict(last_sequence.reshape(1, time_step, 1)).flatten()[0]
                        forecast_results.append(next_value)

                        # Update last_sequence untuk prediksi berikutnya
                        last_sequence = np.append(last_sequence[1:], next_value)

                    # Denormalisasi hasil forecast
                    forecast_results_denormalized = scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1)).flatten()

                    # Plot hasil forecast dengan Plotly
                    fig3 = go.Figure()

                    # Actual values
                    fig3.add_trace(go.Scatter(
                        x=np.arange(len(y_test_actual)),
                        y=y_test_actual,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))

                    # Predicted values
                    fig3.add_trace(go.Scatter(
                        x=np.arange(len(y_test_pred_denormalized)),
                        y=y_test_pred_denormalized,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='orange')
                    ))

                    # Forecasted values
                    fig3.add_trace(go.Scatter(
                        x=np.arange(len(y_test_actual), len(y_test_actual) + forecast_periods),
                        y=forecast_results_denormalized,
                        mode='lines',
                        name='Forecasted',
                        line=dict(color='red')
                    ))

                    fig3.update_layout(
                        title='Actual, Predicted, and Forecasted CPO Production',
                        xaxis_title='Time Steps',
                        yaxis_title='CPO Production'
                    )

                    st.plotly_chart(fig3)


                    st.write(f"Forecasted values for the next {forecast_periods} periods:")
                    st.write(forecast_results_denormalized)

        except Exception as e:
            st.error(f"An error occurred: {e}")
