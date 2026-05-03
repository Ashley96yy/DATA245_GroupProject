from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent


@st.cache(allow_output_mutation=True)
def load_artifact(filename):
    return joblib.load(BASE_DIR / filename)


def show_sample(sample):
    st.subheader("Input Row")
    st.dataframe(sample)


def yuyao_demo():
    model = load_artifact("yuyao_tuned_rf_pipeline.pkl")
    st.header("Yuyao Ding: Weather-Normalized Site EUI Prediction")
    st.caption("Saved tuned random forest pipeline")

    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=1983, step=1)
    property_gfa = st.number_input("Property GFA - Self-Reported (ft²)", min_value=1.0, value=50000.0, step=1000.0)
    number_of_buildings = st.number_input("Number of Buildings", min_value=1, value=1, step=1)
    primary_property_type = st.selectbox(
        "Primary Property Type - Self Selected",
        ["Multifamily Housing", "Office", "Hotel", "K-12 School", "Retail Store", "Non-Refrigerated Warehouse"],
    )
    city = st.selectbox("City", ["New York", "Brooklyn", "Queens", "Bronx", "Staten Island"])

    sample = pd.DataFrame([{
        "Year Built": year_built,
        "Property GFA - Self-Reported (ft²)": property_gfa,
        "Number of Buildings": number_of_buildings,
        "Primary Property Type - Self Selected": primary_property_type,
        "City": city,
    }])
    show_sample(sample)

    if st.button("Predict EUI"):
        prediction = model.predict(sample)[0]
        st.success(f"Predicted Weather-Normalized Site EUI: {prediction:.2f} kBtu/ft²")


def ying_demo():
    model = load_artifact("ying_tuned_rf_pipeline.pkl")
    st.header("Ying Zhu: ENERGY STAR Score Prediction")
    st.caption("Saved tuned random forest pipeline")

    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=1980, step=1)
    number_of_buildings = st.number_input("Number of Buildings", min_value=1, value=1, step=1)
    occupancy = st.number_input("Occupancy", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
    floor_area = st.number_input("Largest Property Use Type - Gross Floor Area (ft²)", min_value=1.0, value=50000.0, step=1000.0)
    electricity = st.number_input("Electricity Use - Grid Purchase (kBtu)", min_value=0.0, value=1500000.0, step=50000.0)
    natural_gas = st.number_input("Natural Gas Use (kBtu)", min_value=0.0, value=800000.0, step=50000.0)
    property_use = st.selectbox(
        "Largest Property Use Type",
        ["Multifamily Housing", "Office", "Hotel", "K-12 School", "Retail Store", "Non-Refrigerated Warehouse"],
    )

    sample = pd.DataFrame([{
        "Year Built": year_built,
        "Number of Buildings": number_of_buildings,
        "Occupancy": occupancy,
        "Largest Property Use Type": property_use,
        "Largest Property Use Type - Gross Floor Area (ft²)": floor_area,
        "Electricity Use - Grid Purchase (kBtu)": electricity,
        "Natural Gas Use (kBtu)": natural_gas,
        "Electricity_missing": int(electricity == 0),
        "NaturalGas_missing": int(natural_gas == 0),
    }])
    show_sample(sample)

    if st.button("Predict ENERGY STAR Score"):
        prediction = model.predict(sample)[0]
        st.success(f"Predicted ENERGY STAR Score: {prediction:.2f}")


def chao_demo():
    artifact = load_artifact("chao_kmeans_cluster_artifact.pkl")
    st.header("Chao Zheng: Building Performance Clustering")
    st.caption("Saved KMeans artifact with scaler and preprocessing thresholds")

    eui = st.number_input("Weather Normalized Site EUI (kBtu/ft²)", min_value=0.1, value=75.0, step=5.0)
    water = st.number_input("Water Use (All Water Sources) (kgal)", min_value=0.1, value=1200.0, step=100.0)
    gfa = st.number_input("Property GFA - Calculated (Buildings and Parking) (ft²)", min_value=1.0, value=50000.0, step=1000.0)

    water_intensity = water / gfa
    energy_wins = min(max(eui, artifact["energy_lower"]), artifact["energy_upper"])
    water_wins = min(max(water_intensity, artifact["water_lower"]), artifact["water_upper"])
    sample = pd.DataFrame([{
        "energy_wins": energy_wins,
        "log_water_intensity": np.log1p(water_wins),
    }])
    show_sample(sample)

    if st.button("Assign Cluster"):
        sample_scaled = artifact["scaler"].transform(sample)
        cluster = int(artifact["model"].predict(sample_scaled)[0])
        cluster_name = artifact["cluster_names"].get(cluster, "Unknown")
        st.success(f"Assigned cluster: {cluster} - {cluster_name}")


def jim_demo():
    model = load_artifact("jim_tuned_rf_pipeline.pkl")
    st.header("Jim He: GHG Emissions Intensity Prediction")
    st.caption("Saved tuned random forest pipeline")

    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=1980, step=1, key="jim_year")
    number_of_buildings = st.number_input("Number of Buildings", min_value=1, value=1, step=1, key="jim_buildings")
    occupancy = st.number_input("Occupancy", min_value=0.0, max_value=100.0, value=95.0, step=1.0, key="jim_occupancy")
    model_gfa = st.number_input("Model GFA (ft²)", min_value=1.0, value=50000.0, step=1000.0, key="jim_gfa")
    total_fuel = st.number_input("Total Fuel Use (kBtu)", min_value=1.0, value=4250000.0, step=50000.0, key="jim_fuel")
    electricity_share = st.slider("Electricity Share", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    natural_gas_share = st.slider("Natural Gas Share", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    property_type = st.selectbox(
        "Primary Property Type - Self Selected",
        ["Multifamily Housing", "Office", "Hotel", "K-12 School", "Retail Store", "Non-Refrigerated Warehouse"],
        key="jim_property_type",
    )
    largest_use_type = st.selectbox(
        "Largest Property Use Type",
        ["Multifamily Housing", "Office", "Hotel", "K-12 School", "Retail Store", "Non-Refrigerated Warehouse"],
        key="jim_largest_use_type",
    )
    city = st.selectbox("City", ["New York", "Brooklyn", "Queens", "Bronx", "Staten Island"], key="jim_city")

    fuel_intensity = total_fuel / model_gfa
    largest_use_share = 0.9
    sample = pd.DataFrame([{
        "Calendar Year": 2024,
        "Year Built": year_built,
        "Number of Buildings": number_of_buildings,
        "Occupancy": occupancy,
        "Model GFA (ft²)": model_gfa,
        "Largest Use Share of GFA": largest_use_share,
        "Fuel Use Intensity (kBtu/ft²)": fuel_intensity,
        "Log Model GFA": np.log1p(model_gfa),
        "Log Total Fuel Use": np.log1p(total_fuel),
        "Primary Property Type - Self Selected": property_type,
        "Largest Property Use Type": largest_use_type,
        "City": city,
        "electricity_grid_purchase_share": electricity_share,
        "natural_gas_share": natural_gas_share,
        "district_steam_share": 0.0,
        "district_hot_water_share": 0.0,
        "district_chilled_water_share": 0.0,
        "fuel_oil_1_share": 0.0,
        "fuel_oil_2_share": 0.0,
        "fuel_oil_4_share": 0.0,
        "fuel_oil_5_6_share": 0.0,
        "diesel_2_share": 0.0,
        "propane_share": 0.0,
        "kerosene_share": 0.0,
        "electricity_grid_purchase_missing": int(electricity_share == 0),
        "natural_gas_missing": int(natural_gas_share == 0),
        "district_steam_missing": 1,
        "district_hot_water_missing": 1,
        "district_chilled_water_missing": 1,
        "fuel_oil_1_missing": 1,
        "fuel_oil_2_missing": 1,
        "fuel_oil_4_missing": 1,
        "fuel_oil_5_6_missing": 1,
        "diesel_2_missing": 1,
        "propane_missing": 1,
        "kerosene_missing": 1,
    }])
    show_sample(sample)

    if st.button("Predict GHG Emissions Intensity"):
        prediction = model.predict(sample)[0]
        st.success(f"Predicted GHG Emissions Intensity: {prediction:.2f} kgCO2e/ft²")


st.set_page_config(page_title="DATA245 Saved Model Demo", layout="centered")
st.title("DATA245 Group Project Saved Model Demo")
st.caption("Loads saved artifacts and runs lightweight prediction or cluster-assignment checks.")

demo_choice = st.sidebar.selectbox(
    "Choose demo",
    [
        "Yuyao - EUI Prediction",
        "Ying - ENERGY STAR Prediction",
        "Chao - Clustering",
        "Jim - GHG Emissions Prediction",
    ],
)

if demo_choice.startswith("Yuyao"):
    yuyao_demo()
elif demo_choice.startswith("Ying"):
    ying_demo()
elif demo_choice.startswith("Chao"):
    chao_demo()
else:
    jim_demo()

st.caption("All outputs are generated from saved artifacts without retraining.")
