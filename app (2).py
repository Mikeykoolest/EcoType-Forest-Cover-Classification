import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -------------------------------
# LOAD MODEL & ENCODER
# -------------------------------
model = pickle.load(open("forest_cover_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="EcoType Predictor", layout="centered")

st.title("🌲 EcoType: Forest Cover Type Prediction")
st.markdown("Predict forest type using environmental and cartographic features")

# -------------------------------
# INPUT SECTION WITH DESCRIPTIONS
# -------------------------------
st.header("📥 Enter Environmental Details")

col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input("Elevation (meters)", 0, 5000, 2500,
        help="Height above sea level")

    aspect = st.slider("Aspect (0–360°)", 0, 360, 180,
        help="Direction the slope faces (0 = North, 180 = South)")

    slope = st.number_input("Slope (degrees)", 0, 60, 10,
        help="Steepness of terrain")

    hd_hydrology = st.number_input("Distance to Water (Horizontal)", 0, 5000, 100,
        help="Distance to nearest water source")

    vd_hydrology = st.number_input("Distance to Water (Vertical)", -500, 500, 0,
        help="Elevation difference from water source")

    hd_roadways = st.number_input("Distance to Roads", 0, 5000, 1000,
        help="Distance to nearest road")

with col2:
    hillshade_9am = st.slider("Hillshade 9 AM", 0, 255, 200,
        help="Sunlight intensity in morning")

    hillshade_noon = st.slider("Hillshade Noon", 0, 255, 220,
        help="Sunlight intensity at noon")

    hillshade_3pm = st.slider("Hillshade 3 PM", 0, 255, 180,
        help="Sunlight intensity in afternoon")

    hd_fire = st.number_input("Distance to Fire Points", 0, 5000, 1000,
        help="Distance to nearest wildfire point")

    wilderness = st.selectbox("Wilderness Area", [0,1,2,3],
        help="Type of protected wilderness zone")

    soil = st.number_input("Soil Type (0–39)", 0, 39, 10,
        help="Soil classification index")

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
aspect_sin = np.sin(np.radians(aspect))
aspect_cos = np.cos(np.radians(aspect))

hydrology_distance = np.sqrt(hd_hydrology**2 + vd_hydrology**2)
hillshade_mean = (hillshade_9am + hillshade_noon + hillshade_3pm) / 3

elevation_slope = elevation / (slope + 1)
hydrology_ratio = hd_hydrology / (abs(vd_hydrology) + 1)

# -------------------------------
# FINAL FEATURE ARRAY (18)
# -------------------------------
features = np.array([[ 
    elevation, aspect, slope,
    hd_hydrology, vd_hydrology, hd_roadways,
    hillshade_9am, hillshade_noon, hillshade_3pm,
    hd_fire, wilderness, soil,
    aspect_sin, aspect_cos,
    hydrology_distance, hillshade_mean,
    elevation_slope, hydrology_ratio
]])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Forest Type"):

    prediction = model.predict(features)
    result = encoder.inverse_transform(prediction)[0]

    probs = model.predict_proba(features)[0]

    st.success(f"🌳 Predicted Forest Type: **{result}**")

    # -------------------------------
    # PROBABILITY DISPLAY
    # -------------------------------
    st.subheader("📊 Prediction Confidence")

    class_labels = encoder.classes_
    prob_df = pd.DataFrame({
        "Forest Type": class_labels,
        "Probability": probs
    }).sort_values(by="Probability", ascending=False)

    st.dataframe(prob_df)

    # -------------------------------
    # BAR CHART
    # -------------------------------
    st.subheader("📈 Probability Distribution")

    st.bar_chart(prob_df.set_index("Forest Type"))
