import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# Load model and scaler
model = tf.keras.models.load_model("movie_model.keras")
scaler = joblib.load("scaler.pkl")

st.title(" Box Office Prediction App")
st.write("Enter the movie features below to predict box office revenue:")

# Input fields
budget = st.number_input("Budget (in million $)", min_value=1.0, step=0.5)
runtime = st.number_input("Runtime (in minutes)", min_value=60, step=1)
popularity = st.number_input("Popularity Score", min_value=0.0, step=0.1)
vote_average = st.number_input("Vote Average (IMDb)", min_value=0.0, max_value=10.0, step=0.1)

if st.button(" Predict Revenue"):
    # Prepare input
    features = np.array([[budget, runtime, popularity, vote_average]])
    scaled = scaler.transform(features)

    # Predict log-revenue and convert back
    prediction = model.predict(scaled)
    predicted_revenue = np.expm1(prediction[0][0])  # convert from log scale
    st.success(f" Predicted Box Office Revenue: **${predicted_revenue:,.2f} million**")

    # ROI calculation
    roi = predicted_revenue / budget if budget > 0 else 0

    # --- ROI Classification ---
    if roi < 1.0:
        category, color = "❌ Flop", "#FF4B4B"          # Red
    elif roi < 2.5:
        category, color = "⚪ Average", "#F1C40F"       # Yellow
    elif roi < 4.0:
        category, color = "🟢 Hit", "#2ECC71"           # Green
    else:
        category, color = "🟣 Blockbuster", "#9B59B6"   # Purple

    # Clamp ROI and ensure valid float for progress bar
    safe_roi = 0.0 if np.isnan(roi) or np.isinf(roi) or roi < 0 else min(float(roi) / 5, 1.0)

    # Styled category display
    st.markdown(
        f"###  Performance Category: <span style='color:{color}'>{category}</span>",
        unsafe_allow_html=True
    )

    # Custom ROI bar
    progress_html = f"""
    <div style='background-color:#ddd; border-radius:12px; height:25px; width:100%;'>
        <div style='background:linear-gradient(90deg, #FF4B4B, #F1C40F, #2ECC71, #9B59B6);
                    width:{safe_roi*100}%;
                    height:100%;
                    border-radius:12px;'>
        </div>
    </div>
    <p style='text-align:center; font-weight:bold; margin-top:5px;'>
        ROI: {roi:.2f}×
    </p>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

    # Optional visualization
    st.subheader(" Feature Values Overview")
    df = pd.DataFrame({
        "Feature": ["Budget", "Runtime", "Popularity", "Vote Average"],
        "Value": [budget, runtime, popularity, vote_average]
    })
    fig = px.bar(df, x="Feature", y="Value", title="Entered Feature Values", color="Feature")
    st.plotly_chart(fig)
