from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)


@app.route('/api/bitcoin-data', methods=['GET'])
def get_bitcoin_data():

    try:

        # =====================================================
        # 1. LOAD BITCOIN DATASET
        # =====================================================

        df = pd.read_csv("bitcoin_new.csv")

        # Convert Date column
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Convert numeric columns
        numeric_columns = [
            "Price",
            "Open",
            "High",
            "Low",
            "Volume"
        ]

        for column in numeric_columns:

            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        # Remove invalid dates
        df = df.dropna(subset=["Date"])

        # Sort by date
        df = df.sort_values("Date")

        # Remove duplicate dates
        df = df.drop_duplicates(
            subset=["Date"],
            keep="last"
        )

        # Set Date as index
        df = df.set_index("Date")

        # Remove rows without Price
        df = df.dropna(
            subset=["Price"]
        )

        # =====================================================
        # 2. 30-DAY MOVING AVERAGE
        # =====================================================

        df["MA_30"] = (
            df["Price"]
            .rolling(window=30)
            .mean()
        )

        # =====================================================
        # 3. PRICE DATA
        # =====================================================

        price_data = df["Price"].astype(float)

        # Make sure data is sorted
        price_data = price_data.sort_index()

        # =====================================================
        # 4. TRAIN / TEST SPLIT
        # =====================================================

        train_size = int(
            len(price_data) * 0.80
        )

        train = price_data.iloc[
            :train_size
        ]

        test = price_data.iloc[
            train_size:
        ]

        # =====================================================
        # 5. TRAIN ARIMA MODEL
        # =====================================================

        model = ARIMA(
            train,
            order=(1, 0, 1)
        )

        model_fit = model.fit()

        # =====================================================
        # 6. TEST PREDICTION
        # =====================================================

        predictions = model_fit.forecast(
            steps=len(test)
        )

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        actual = np.asarray(
            test,
            dtype=float
        )

        # =====================================================
        # 7. MODEL PERFORMANCE
        # =====================================================

        mae = mean_absolute_error(
            actual,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predictions
            )
        )

        # =====================================================
        # 8. TRAIN FINAL MODEL USING ALL DATA
        # =====================================================

        final_model = ARIMA(
            price_data,
            order=(1, 0, 1)
        )

        final_model_fit = (
            final_model.fit()
        )

        # =====================================================
        # 9. FUTURE BITCOIN FORECAST
        # =====================================================

        forecast_steps = 10

        future_forecast = (
            final_model_fit.forecast(
                steps=forecast_steps
            )
        )

        future_forecast = np.asarray(
            future_forecast,
            dtype=float
        )

        # =====================================================
        # 10. FUTURE DATES
        # =====================================================

        last_date = price_data.index[-1]

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_steps,
            freq="D"
        )

        # =====================================================
        # 11. CURRENT BITCOIN PRICE
        # =====================================================

        current_price = float(
            price_data.iloc[-1]
        )

        # =====================================================
        # 12. UP / DOWN PREDICTION
        # =====================================================

        forecast_data = []

        previous_price = current_price

        for i in range(forecast_steps):

            predicted_price = float(
                future_forecast[i]
            )

            change = (
                predicted_price
                - previous_price
            )

            percentage_change = (
                change
                / previous_price
            ) * 100

            # Direction
            if percentage_change > 0.10:

                direction = "UP"

            elif percentage_change < -0.10:

                direction = "DOWN"

            else:

                direction = "STABLE"

            forecast_data.append({

                "date":
                    future_dates[i].strftime(
                        "%Y-%m-%d"
                    ),

                "predicted_price":
                    round(
                        predicted_price,
                        2
                    ),

                "previous_price":
                    round(
                        previous_price,
                        2
                    ),

                "change":
                    round(
                        change,
                        2
                    ),

                "change_percent":
                    round(
                        percentage_change,
                        2
                    ),

                "direction":
                    direction
            })

            previous_price = (
                predicted_price
            )

        # =====================================================
        # 13. OVERALL PREDICTION
        # =====================================================

        final_predicted_price = float(
            future_forecast[-1]
        )

        overall_change = (
            final_predicted_price
            - current_price
        )

        overall_percentage = (
            overall_change
            / current_price
        ) * 100

        if overall_percentage > 0.10:

            overall_direction = "UP"

        elif overall_percentage < -0.10:

            overall_direction = "DOWN"

        else:

            overall_direction = "STABLE"

        # =====================================================
        # 14. HISTORICAL DATA
        # =====================================================

        historical_dates = (
            df.index
            .strftime("%Y-%m-%d")
            .tolist()
        )

        historical_prices = (
            df["Price"]
            .round(2)
            .tolist()
        )

        moving_average = (
            df["MA_30"]
            .round(2)
            .replace(
                {np.nan: None}
            )
            .tolist()
        )

        # =====================================================
        # 15. TEST EVALUATION DATA
        # =====================================================

        test_dates = (
            test.index
            .strftime("%Y-%m-%d")
            .tolist()
        )

        # =====================================================
        # 16. FINAL JSON RESPONSE
        # =====================================================

        data = {

            # -------------------------
            # Current price
            # -------------------------

            "current_price":
                round(
                    current_price,
                    2
                ),

            # -------------------------
            # Prediction summary
            # -------------------------

            "prediction_summary": {

                "direction":
                    overall_direction,

                "predicted_price":
                    round(
                        final_predicted_price,
                        2
                    ),

                "change":
                    round(
                        overall_change,
                        2
                    ),

                "change_percent":
                    round(
                        overall_percentage,
                        2
                    )
            },

            # -------------------------
            # Dataset information
            # -------------------------

            "dataset": {

                "total_records":
                    len(df),

                "start_date":
                    df.index[0].strftime(
                        "%Y-%m-%d"
                    ),

                "last_date":
                    df.index[-1].strftime(
                        "%Y-%m-%d"
                    )
            },

            # -------------------------
            # Historical prices
            # -------------------------

            "historical": {

                "dates":
                    historical_dates,

                "prices":
                    historical_prices,

                "ma_30":
                    moving_average
            },

            # -------------------------
            # Model evaluation
            # -------------------------

            "test_evaluation": {

                "dates":
                    test_dates,

                "actual":
                    actual.tolist(),

                "predicted":
                    predictions.tolist()
            },

            # -------------------------
            # Future forecast
            # -------------------------

            "forecast": {

                "dates":
                    future_dates.strftime(
                        "%Y-%m-%d"
                    ).tolist(),

                "predicted":
                    [
                        round(
                            float(x),
                            2
                        )
                        for x in future_forecast
                    ],

                "direction":
                    [
                        item["direction"]
                        for item in forecast_data
                    ],

                "change_percent":
                    [
                        item["change_percent"]
                        for item in forecast_data
                    ],

                "details":
                    forecast_data
            },

            # -------------------------
            # Model metrics
            # -------------------------

            "metrics": {

                "mae":
                    round(
                        mae,
                        2
                    ),

                "rmse":
                    round(
                        rmse,
                        2
                    )
            }
        }

        return jsonify(data)

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# =============================================================
# START FLASK SERVER
# =============================================================

if __name__ == '__main__':

    print(
        "\n======================================"
    )

    print(
        " Bitcoin Prediction API"
    )

    print(
        "======================================"
    )

    print(
        "Dataset: bitcoin.csv"
    )

    print(
        "API: http://localhost:5000/api/bitcoin-data"
    )

    print(
        "Server: http://localhost:5000"
    )

    print(
        "======================================\n"
    )

    app.run(
        debug=True,
        port=5000
    )