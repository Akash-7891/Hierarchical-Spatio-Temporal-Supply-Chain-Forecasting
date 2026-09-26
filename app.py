import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error




st.title("Hierarchical Supply Chain Forecasting")

st.write(
    "Forecast store-level sales and reconcile them "
    "into region and national level forecasts."
)



df = pd.read_csv("sales_data.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.dropna()

df = df.sort_values(
    ["region", "store", "date"]
)




df["day_of_week"] = df["date"].dt.dayofweek

df["lag_1"] = df.groupby(
    ["region", "store"]
)["sales"].shift(1)

df["lag_7"] = df.groupby(
    ["region", "store"]
)["sales"].shift(7)

# Remove rows where lag values are not available
df = df.dropna().reset_index(drop=True)




st.subheader("Sample Data")

st.dataframe(df.head())



# SELECT REGION


st.subheader("Choose a Region")

regions = sorted(df["region"].unique())

selected_region = st.selectbox(
    "Select Region:",
    regions
)



# FORECAST BUTTON


if st.button("Generate Forecast"):

    # SELECTED REGION DATA
    

    region_df = df[
        df["region"] == selected_region
    ].copy()


 
    # TRAIN TEST SPLIT
    

    cutoff_date = region_df["date"].quantile(
        0.8,
        interpolation="nearest"
    )

    train = region_df[
        region_df["date"] < cutoff_date
    ]

    test = region_df[
        region_df["date"] >= cutoff_date
    ].copy()


  
    # FEATURES
   

    features = [
        "day_of_week",
        "lag_1",
        "lag_7"
    ]


 
    # LINEAR REGRESSION MODEL
   

    model = LinearRegression()

    model.fit(
        train[features],
        train["sales"]
    )


    # STORE LEVEL FORECAST
   
    test["predicted_sales"] = model.predict(
        test[features]
    )

    # Avoid negative sales predictions
    test["predicted_sales"] = np.maximum(
        test["predicted_sales"],
        0
    )


    st.subheader("Store Level Forecast")

    st.dataframe(
        test[
            [
                "date",
                "region",
                "store",
                "sales",
                "predicted_sales"
            ]
        ]
    )



    # ACCURACY
   

    mae = mean_absolute_error(
        test["sales"],
        test["predicted_sales"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["sales"],
            test["predicted_sales"]
        )
    )


    st.subheader("Forecast Accuracy")

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        round(mae, 2)
    )

    col2.metric(
        "RMSE",
        round(rmse, 2)
    )


   
    # RECONCILIATION FUNCTION
   

    def reconcile_forecasts(data):

        # Store-level forecasts
        store_forecast = data.groupby(
            ["date", "region", "store"]
        )["predicted_sales"].sum().reset_index()


        # Region-level forecast
        region_forecast = store_forecast.groupby(
            ["date", "region"]
        )["predicted_sales"].sum().reset_index()


        # National-level forecast
        national_forecast = region_forecast.groupby(
            "date"
        )["predicted_sales"].sum().reset_index()


        return (
            store_forecast,
            region_forecast,
            national_forecast
        )


   
    # APPLY RECONCILIATION
    

    (
        store_forecast,
        region_forecast,
        national_forecast
    ) = reconcile_forecasts(test)


   
    # REGION ACTUAL SALES
    

    region_actual = test.groupby(
        ["date", "region"]
    )["sales"].sum().reset_index()

    region_actual = region_actual.rename(
        columns={
            "sales": "actual_sales"
        }
    )


    # Merge actual and forecast
    region_result = pd.merge(
        region_actual,
        region_forecast,
        on=["date", "region"]
    )

    region_result = region_result.rename(
        columns={
            "predicted_sales":
            "reconciled_forecast"
        }
    )


    # REGION LEVEL
    

    st.subheader(
        "Region Level Reconciled Forecast"
    )

    st.dataframe(
        region_result
    )


    # --------------------------------------------------
    # NATIONAL ACTUAL SALES
    # --------------------------------------------------

    national_actual = df[
        df["date"].isin(
            national_forecast["date"]
        )
    ]

    national_actual = national_actual.groupby(
        "date"
    )["sales"].sum().reset_index()

    national_actual = national_actual.rename(
        columns={
            "sales": "actual_sales"
        }
    )


    # Merge national actual + forecast
    national_result = pd.merge(
        national_actual,
        national_forecast,
        on="date"
    )

    national_result = national_result.rename(
        columns={
            "predicted_sales":
            "reconciled_forecast"
        }
    )


    # NATIONAL LEVEL
 

    st.subheader(
        "National Level Reconciled Forecast"
    )

    st.dataframe(
        national_result
    )


    
    # HIERARCHY CHECK
    

    st.subheader(
        "Hierarchy Reconciliation Check"
    )

    hierarchy_check = region_forecast.groupby(
        "date"
    )["predicted_sales"].sum().reset_index()

    hierarchy_check = hierarchy_check.merge(
        national_forecast,
        on="date"
    )

    hierarchy_check = hierarchy_check.rename(
        columns={
            "predicted_sales_x":
            "Sum_of_Region_Forecasts",

            "predicted_sales_y":
            "National_Forecast"
        }
    )

    hierarchy_check["Difference"] = (
        hierarchy_check["Sum_of_Region_Forecasts"]
        - hierarchy_check["National_Forecast"]
    )

    st.dataframe(
        hierarchy_check
    )


    # REGION GRAPH
    

    st.subheader(
        "Region: Actual vs Forecast"
    )

    fig, ax = plt.subplots()

    ax.plot(
        region_result["date"],
        region_result["actual_sales"],
        label="Actual Sales"
    )

    ax.plot(
        region_result["date"],
        region_result["reconciled_forecast"],
        label="Reconciled Forecast"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    ax.set_title(
        selected_region +
        " Sales Forecast"
    )

    ax.legend()

    st.pyplot(fig)


    
    # NATIONAL GRAPH
 

    st.subheader(
        "National: Actual vs Forecast"
    )

    fig2, ax2 = plt.subplots()

    ax2.plot(
        national_result["date"],
        national_result["actual_sales"],
        label="Actual Sales"
    )

    ax2.plot(
        national_result["date"],
        national_result["reconciled_forecast"],
        label="Reconciled Forecast"
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sales")

    ax2.set_title(
        "National Sales Forecast"
    )

    ax2.legend()

    st.pyplot(fig2)



    # HIERARCHY DISPLAY
 

    st.subheader("Forecast Hierarchy")

    st.write(
        "Store → Region → National"
    )

    st.success(
        "Store forecasts have been aggregated into "
        "region forecasts and then into the national forecast."
    )


