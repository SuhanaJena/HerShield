# HerShield — Intelligent Women Safety & Risk Analytics

HerShield is a women-safety analytics application built using Python, Streamlit, data analytics, geospatial analysis, and machine learning.

The project explores crime patterns, identifies geographic risk areas, provides location-based safety insights, and offers experimental image screening for incident reports.

## Features

- **Safety Dashboard:** Overview of crime data and safety indicators.
- **Crime Analytics:** Explore crime patterns and statistical trends.
- **Hotspot Detection:** Identify geographic clusters using machine learning.
- **Safety Heatmap:** Visualize crime-related geographic patterns.
- **Anomaly Detection:** Identify unusual patterns in the dataset.
- **Risk Prediction:** Explore location-related risk indicators.
- **Safe Route Analysis:** Compare routes using available geographic and risk information.
- **GPS Integration:** Access browser location functionality.
- **Incident Image Screening:** Examine image metadata, duplicate indicators, and an experimental deep-learning model.
- **Interactive Maps:** Visualize geographic information using Folium.

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Plotly
- Folium and Streamlit-Folium
- Shapely
- Pillow

## Project Structure

```text
HerShield/
├── app.py
├── requirements.txt
├── create_anomalies.py
├── create_clusters.py
├── create_features.py
├── create_risk_features.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── maps/
│   └── real_world_evaluation/
├── models/
│   ├── image_authenticity_model.keras
│   ├── train_image_model.py
│   └── evaluate_real_world.py
└── modules/
    ├── gps.py
    └── gps_risk.py

