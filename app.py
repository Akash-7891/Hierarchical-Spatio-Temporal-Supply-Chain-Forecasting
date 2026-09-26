import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hierarchical Supply Chain Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("Hierarchical Supply Chain Forecasting")

st.write(
    "Forecast SKU-level sales and reconcile them through "
    "Store → Region → National hierarchy."
)


# ============================================================
# LOAD CSV
# CSV columns:
# date, region, store, sku, sales
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("sales_data.csv")

    required_columns = [
        "date",
        "region",
        "store",
        "sku",
        "sales"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in CSV: {missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "region",
            "store",
            "sku",
            "sales"
        ]
    )

    df = df.sort_values(
        ["region", "store", "sku", "date"]
    ).reset_index(drop=True)

    return df


try:
    df = load_data()

except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

# SKU-level lag features
group_columns = [
    "region",
    "store",
    "sku"
]

df["lag_1"] = (
    df.groupby(group_columns)["sales"]
    .shift(1)
)

df["lag_7"] = (
    df.groupby(group_columns)["sales"]
    .shift(7)
)

df["rolling_7"] = (
    df.groupby(group_columns)["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

# Rows without enough history cannot be used for forecasting
df = df.dropna(
    subset=[
        "lag_1",
        "lag_7",
        "rolling_7"
    ]
).reset_index(drop=True)


# ============================================================
# DATA SUMMARY
# ============================================================

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Rows",
    len(df)
)

col2.metric(
    "Regions",
    df["region"].nunique()
)

col3.metric(
    "Stores",
    df["store"].nunique()
)

col4.metric(
    "SKUs",
    df["sku"].nunique()
)

st.subheader("Sample Data")
st.dataframe(
    df.head(20),
    use_container_width=True
)


# ============================================================
# REGION SELECTION
# ============================================================

st.subheader("Choose a Region")

regions = sorted(
    df["region"].unique()
)

selected_region = st.selectbox(
    "Select Region:",
    regions
)


# ============================================================
# FORECAST BUTTON
# ============================================================

if st.button(
    "Generate Forecast",
    type="primary"
):

    # ========================================================
    # FEATURES
    # ========================================================

    features = [
        "day_of_week",
        "day_of_month",
        "month",
        "week_of_year",
        "lag_1",
        "lag_7",
        "rolling_7"
    ]


    # ========================================================
    # TIME-BASED TRAIN / TEST SPLIT
    # ========================================================

    unique_dates = sorted(
        df["date"].unique()
    )

    if len(unique_dates) < 2:
        st.error(
            "Not enough dates available for train/test forecasting."
        )
        st.stop()

    split_index = int(
        len(unique_dates) * 0.80
    )

    split_index = max(
        1,
        min(
            split_index,
            len(unique_dates) - 1
        )
    )

    cutoff_date = unique_dates[split_index]


    train = df[
        df["date"] < cutoff_date
    ].copy()

    test = df[
        df["date"] >= cutoff_date
    ].copy()


    # ========================================================
    # LINEAR REGRESSION MODEL
    # Train on ALL regions/SKUs so national reconciliation
    # represents the complete hierarchy.
    # ========================================================

    model = LinearRegression()

    model.fit(
        train[features],
        train["sales"]
    )


    # ========================================================
    # SKU LEVEL FORECAST
    # ========================================================

    test["predicted_sales"] = model.predict(
        test[features]
    )

    # Sales cannot be negative
    test["predicted_sales"] = np.maximum(
        test["predicted_sales"],
        0
    )


    # ========================================================
    # SKU LEVEL FORECAST DISPLAY
    # ========================================================

    st.subheader(
        "SKU Level Forecast"
    )

    selected_region_sku = test[
        test["region"] == selected_region
    ][
        [
            "date",
            "region",
            "store",
            "sku",
            "sales",
            "predicted_sales"
        ]
    ].copy()

    st.dataframe(
        selected_region_sku,
        use_container_width=True
    )


    # ========================================================
    # FORECAST ACCURACY
    # Selected Region
    # ========================================================

    region_test = test[
        test["region"] == selected_region
    ].copy()

    if len(region_test) > 0:

        mae = mean_absolute_error(
            region_test["sales"],
            region_test["predicted_sales"]
        )

        rmse = np.sqrt(
            mean_squared_error(
                region_test["sales"],
                region_test["predicted_sales"]
            )
        )

        st.subheader(
            f"Forecast Accuracy - {selected_region}"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "MAE",
            round(mae, 2)
        )

        col2.metric(
            "RMSE",
            round(rmse, 2)
        )


    # ========================================================
    # RECONCILIATION
    #
    # SKU → Store → Region → National
    # ========================================================

    def reconcile_forecasts(data):

        # ----------------------------------------------
        # SKU level
        # ----------------------------------------------

        sku_forecast = (
            data.groupby(
                [
                    "date",
                    "region",
                    "store",
                    "sku"
                ],
                as_index=False
            )["predicted_sales"]
            .sum()
        )

        # ----------------------------------------------
        # Store level
        # ----------------------------------------------

        store_forecast = (
            sku_forecast.groupby(
                [
                    "date",
                    "region",
                    "store"
                ],
                as_index=False
            )["predicted_sales"]
            .sum()
        )

        # ----------------------------------------------
        # Region level
        # ----------------------------------------------

        region_forecast = (
            store_forecast.groupby(
                [
                    "date",
                    "region"
                ],
                as_index=False
            )["predicted_sales"]
            .sum()
        )

        # ----------------------------------------------
        # National level
        # ----------------------------------------------

        national_forecast = (
            region_forecast.groupby(
                "date",
                as_index=False
            )["predicted_sales"]
            .sum()
        )

        return (
            sku_forecast,
            store_forecast,
            region_forecast,
            national_forecast
        )


    (
        sku_forecast,
        store_forecast,
        region_forecast,
        national_forecast
    ) = reconcile_forecasts(test)


    # ========================================================
    # SELECTED REGION ACTUAL SALES
    # ========================================================

    region_actual = (
        test.groupby(
            [
                "date",
                "region"
            ],
            as_index=False
        )["sales"]
        .sum()
        .rename(
            columns={
                "sales": "actual_sales"
            }
        )
    )


    # ========================================================
    # REGION ACTUAL + FORECAST
    # ========================================================

    region_result = pd.merge(
        region_actual,
        region_forecast,
        on=[
            "date",
            "region"
        ],
        how="inner"
    )

    region_result = region_result.rename(
        columns={
            "predicted_sales":
            "reconciled_forecast"
        }
    )


    selected_region_result = region_result[
        region_result["region"] == selected_region
    ].copy()


    # ========================================================
    # REGION LEVEL DISPLAY
    # ========================================================

    st.subheader(
        "Region Level Reconciled Forecast"
    )

    st.dataframe(
        selected_region_result,
        use_container_width=True
    )


    # ========================================================
    # NATIONAL ACTUAL SALES
    # ========================================================

    national_actual = (
        test.groupby(
            "date",
            as_index=False
        )["sales"]
        .sum()
        .rename(
            columns={
                "sales": "actual_sales"
            }
        )
    )


    # ========================================================
    # NATIONAL ACTUAL + FORECAST
    # ========================================================

    national_result = pd.merge(
        national_actual,
        national_forecast,
        on="date",
        how="inner"
    )

    national_result = national_result.rename(
        columns={
            "predicted_sales":
            "reconciled_forecast"
        }
    )


    # ========================================================
    # NATIONAL LEVEL DISPLAY
    # ========================================================

    st.subheader(
        "National Level Reconciled Forecast"
    )

    st.dataframe(
        national_result,
        use_container_width=True
    )


    # ========================================================
    # HIERARCHY RECONCILIATION CHECK
    # ========================================================

    st.subheader(
        "Hierarchy Reconciliation Check"
    )

    hierarchy_check = (
        region_forecast.groupby(
            "date",
            as_index=False
        )["predicted_sales"]
        .sum()
        .rename(
            columns={
                "predicted_sales":
                "Sum_of_Region_Forecasts"
            }
        )
    )

    hierarchy_check = hierarchy_check.merge(
        national_forecast.rename(
            columns={
                "predicted_sales":
                "National_Forecast"
            }
        ),
        on="date",
        how="inner"
    )

    hierarchy_check["Difference"] = (
        hierarchy_check[
            "Sum_of_Region_Forecasts"
        ]
        -
        hierarchy_check[
            "National_Forecast"
        ]
    )

    st.dataframe(
        hierarchy_check,
        use_container_width=True
    )


    # ========================================================
    # STORE → REGION CHECK
    # ========================================================

    st.subheader(
        "Store to Region Reconciliation Check"
    )

    store_region_check = (
        store_forecast.groupby(
            [
                "date",
                "region"
            ],
            as_index=False
        )["predicted_sales"]
        .sum()
        .rename(
            columns={
                "predicted_sales":
                "Sum_of_Store_Forecasts"
            }
        )
    )

    store_region_check = store_region_check.merge(
        region_forecast.rename(
            columns={
                "predicted_sales":
                "Region_Forecast"
            }
        ),
        on=[
            "date",
            "region"
        ],
        how="inner"
    )

    store_region_check["Difference"] = (
        store_region_check[
            "Sum_of_Store_Forecasts"
        ]
        -
        store_region_check[
            "Region_Forecast"
        ]
    )

    st.dataframe(
        store_region_check,
        use_container_width=True
    )


    # ========================================================
    # SKU → STORE CHECK
    # ========================================================

    st.subheader(
        "SKU to Store Reconciliation Check"
    )

    sku_store_check = (
        sku_forecast.groupby(
            [
                "date",
                "region",
                "store"
            ],
            as_index=False
        )["predicted_sales"]
        .sum()
        .rename(
            columns={
                "predicted_sales":
                "Sum_of_SKU_Forecasts"
            }
        )
    )

    sku_store_check = sku_store_check.merge(
        store_forecast.rename(
            columns={
                "predicted_sales":
                "Store_Forecast"
            }
        ),
        on=[
            "date",
            "region",
            "store"
        ],
        how="inner"
    )

    sku_store_check["Difference"] = (
        sku_store_check[
            "Sum_of_SKU_Forecasts"
        ]
        -
        sku_store_check[
            "Store_Forecast"
        ]
    )

    st.dataframe(
        sku_store_check,
        use_container_width=True
    )


    # ========================================================
    # REGION GRAPH
    # ========================================================

    st.subheader(
        f"{selected_region}: Actual vs Forecast"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        selected_region_result["date"],
        selected_region_result["actual_sales"],
        label="Actual Sales"
    )

    ax.plot(
        selected_region_result["date"],
        selected_region_result["reconciled_forecast"],
        label="Reconciled Forecast"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    ax.set_title(
        f"{selected_region} Sales Forecast"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # NATIONAL GRAPH
    # ========================================================

    st.subheader(
        "National: Actual vs Forecast"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

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

    ax2.grid(
        alpha=0.3
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)


    # ========================================================
    # FORECAST HIERARCHY
    # ========================================================

    st.subheader(
        "Forecast Hierarchy"
    )

    st.write(
        "SKU → Store → Region → National"
    )

    st.success(
        "SKU forecasts have been aggregated into store "
        "forecasts, then region forecasts, and finally "
        "the national forecast."
    )


    # ========================================================
    # RECONCILIATION STATUS
    # ========================================================

    max_difference = abs(
        hierarchy_check["Difference"]
    ).max()

    if max_difference < 0.000001:

        st.success(
            "✓ Hierarchy reconciliation passed: "
            "sum of region forecasts equals national forecast."
        )

    else:

        st.warning(
            "Hierarchy reconciliation difference detected."
        )
