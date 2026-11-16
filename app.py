import os
import json
import warnings
import math
import hashlib
from datetime import datetime
from io import BytesIO
from urllib.parse import quote_plus

from PIL import Image, ExifTags

from sklearn.ensemble import IsolationForest

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import folium

from streamlit_folium import st_folium
from modules.gps import display_gps
from modules.gps_risk import analyze_gps_location

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HerShield | Women Safety & Risk Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

SESSION_DEFAULTS = {
    "gps_latitude": None,
    "gps_longitude": None,
    "gps_accuracy": None,
    "gps_available": False,
    "travel_destination_candidates": [],
    "travel_destination": None,
    "travel_destination_lat": None,
    "travel_destination_lon": None,
    "travel_destination_display": None,
    "travel_destination_candidates": [],
    "travel_route": None,
    "travel_transport": None,
    "travel_origin_transport": None,
    "travel_destination_transport": None,
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MAP_DIR = os.path.join(DATA_DIR, "maps", "states")

CLEAN_FILE = os.path.join(PROCESSED_DIR, "hershield_crime_2023_clean.csv")
FEATURE_FILE = os.path.join(PROCESSED_DIR, "hershield_features_2023.csv")
RISK_FILE = os.path.join(PROCESSED_DIR, "hershield_risk_features_2023.csv")
CLUSTER_FILE = os.path.join(PROCESSED_DIR, "hershield_clusters_2023.csv")
ANOMALY_FILE = os.path.join(PROCESSED_DIR, "hershield_anomalies_2023.csv")
REPORT_VERIFICATION_DIR = os.path.join(DATA_DIR, "report_verification")
IMAGE_FINGERPRINT_FILE = os.path.join(REPORT_VERIFICATION_DIR, "image_fingerprints.csv")
IMAGE_MODEL_FILE = os.path.join(BASE_DIR, "models", "image_authenticity_model.keras")


# ============================================================
# EXTERNAL REAL-WORLD DATA SERVICES
# ============================================================

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HTTP_HEADERS = {
    "User-Agent": "HerShield-Women-Safety-Analytics/1.0"
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #11151c;
    }

    [data-testid="stMetric"] {
        background-color: #171c24;
        border: 1px solid #2b313c;
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #aab2bf !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    p, label {
        color: #d5d9e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Unable to read data file: {path}")
        st.error(str(exc))
        return pd.DataFrame()


def format_number(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.0f}"
    except Exception:
        return str(value)


def format_decimal(value, decimals=2):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def get_risk_category(score):
    try:
        score = float(score)
    except Exception:
        return "Unavailable"

    if score < 33.33:
        return "Lower Indicator"
    if score < 66.67:
        return "Moderate Indicator"
    return "Higher Indicator"


def risk_color(score):
    try:
        score = float(score)
    except Exception:
        return "#6c757d"

    if score < 33.33:
        return "#2ecc71"
    if score < 66.67:
        return "#f1c40f"
    return "#e74c3c"


def create_metric(label, value, help_text=None):
    st.metric(label=label, value=value, help=help_text)


def get_anomaly_result(value):
    if value is None:
        return "Unavailable"

    try:
        if pd.isna(value):
            return "Unavailable"
    except Exception:
        pass

    text_value = str(value).strip().lower()

    if text_value in {"-1", "true", "anomaly", "outlier"}:
        return "Anomaly"

    if text_value in {"1", "false", "normal", "0"}:
        return "Normal"

    return "Normal"


def calculate_average_hash(image, hash_size=8):
    """Create a lightweight perceptual fingerprint for duplicate screening."""
    grayscale = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    mean_value = pixels.mean()
    bits = pixels >= mean_value
    return "".join("1" if bit else "0" for bit in bits.flatten())


def hamming_distance(hash_one, hash_two):
    if not isinstance(hash_one, str) or not isinstance(hash_two, str):
        return None
    if len(hash_one) != len(hash_two):
        return None
    return sum(a != b for a, b in zip(hash_one, hash_two))


@st.cache_resource(show_spinner=False)
def load_image_authenticity_model(model_path, modified_time):
    """Load an optional user-trained Keras binary image-forensics model."""
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False)


def get_image_exif(image):
    """Read available EXIF metadata without treating missing metadata as fraud."""
    result = {}
    try:
        exif = image.getexif()
        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name in {"DateTime", "DateTimeOriginal", "DateTimeDigitized", "Make", "Model", "Software"}:
                result[tag_name] = str(value)
    except Exception:
        pass
    return result


def get_image_fingerprint_history():
    if not os.path.exists(IMAGE_FINGERPRINT_FILE):
        return pd.DataFrame(columns=["sha256", "average_hash", "recorded_at"])
    try:
        history = pd.read_csv(IMAGE_FINGERPRINT_FILE)
        expected = ["sha256", "average_hash", "recorded_at"]
        for column in expected:
            if column not in history.columns:
                history[column] = ""
        return history[expected]
    except Exception:
        return pd.DataFrame(columns=["sha256", "average_hash", "recorded_at"])


def state_to_filename(state_name):
    name = str(state_name).strip().lower()

    replacements = {
        "andaman and nicobar islands": "andaman-and-nicobar-islands",
        "andhra pradesh": "andhra-pradesh",
        "arunachal pradesh": "arunachal-pradesh",
        "assam": "assam",
        "bihar": "bihar",
        "chandigarh": "chandigarh",
        "chhattisgarh": "chhattisgarh",
        "dadra and nagar haveli and daman and diu": "dnh-and-dd",
        "delhi": "delhi",
        "goa": "goa",
        "gujarat": "gujarat",
        "haryana": "haryana",
        "himachal pradesh": "himachal-pradesh",
        "jammu and kashmir": "jammu-and-kashmir",
        "jharkhand": "jharkhand",
        "karnataka": "karnataka",
        "kerala": "kerala",
        "ladakh": "ladakh",
        "lakshadweep": "lakshadweep",
        "madhya pradesh": "madhya-pradesh",
        "maharashtra": "maharashtra",
        "manipur": "manipur",
        "meghalaya": "meghalaya",
        "mizoram": "mizoram",
        "nagaland": "nagaland",
        "odisha": "odisha",
        "puducherry": "puducherry",
        "punjab": "punjab",
        "rajasthan": "rajasthan",
        "sikkim": "sikkim",
        "tamil nadu": "tamil-nadu",
        "telangana": "telangana",
        "tripura": "tripura",
        "uttar pradesh": "uttar-pradesh",
        "uttarakhand": "uttarakhand",
        "west bengal": "west-bengal",
    }

    return replacements.get(name, name.replace(" ", "-"))


def load_geojson(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088

    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


# ============================================================
# REAL-WORLD TRAVEL FUNCTIONS
# ============================================================

def _normalise_geocode_result(lat, lon, display_name, result_type="", osm_id="", source=""):
    try:
        return {
            "lat": float(lat),
            "lon": float(lon),
            "display_name": str(display_name),
            "type": str(result_type or ""),
            "osm_id": str(osm_id or ""),
            "class": "",
            "source": source,
        }
    except (TypeError, ValueError):
        return None


def _deduplicate_geocode_results(results):
    unique = []
    seen = set()

    for item in results:
        if not item:
            continue

        key = (round(float(item["lat"]), 5), round(float(item["lon"]), 5))
        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def _is_reasonably_in_india(lat, lon):
    """Reject clearly non-Indian geocoding results."""
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False
    return 6.0 <= lat <= 37.5 and 68.0 <= lon <= 97.5


def _search_nominatim(search_query):
    params = {
        "q": search_query,
        "format": "jsonv2",
        "limit": 8,
        "addressdetails": 1,
        "dedupe": 1,
        "countrycodes": "in",
    }

    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers={
            **HTTP_HEADERS,
            "Accept-Language": "en-IN,en;q=0.9",
        },
        timeout=15,
    )
    response.raise_for_status()

    results = response.json()
    if not isinstance(results, list):
        return []

    parsed = []
    for result in results:
        if not result.get("lat") or not result.get("lon"):
            continue

        if not _is_reasonably_in_india(
            result.get("lat"),
            result.get("lon"),
        ):
            continue

        item = _normalise_geocode_result(
            result["lat"],
            result["lon"],
            result.get("display_name", search_query),
            result.get("type", ""),
            result.get("osm_id", ""),
            "Nominatim / OpenStreetMap",
        )
        if item:
            item["class"] = result.get("class", "")
            parsed.append(item)

    return parsed


def _search_photon(search_query):
    """Fallback real-world geocoder using Photon/OpenStreetMap data."""
    response = requests.get(
        "https://photon.komoot.io/api/",
        params={
            "q": search_query,
            "limit": 8,
            "lang": "en",
            "bbox": "68.0,6.0,97.5,37.5",
        },
        headers=HTTP_HEADERS,
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()
    features = data.get("features", []) if isinstance(data, dict) else []
    parsed = []

    for feature in features:
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        if len(coordinates) < 2:
            continue

        lon = coordinates[0]
        lat = coordinates[1]
        if not _is_reasonably_in_india(lat, lon):
            continue

        properties = feature.get("properties", {})
        parts = [
            properties.get("name"),
            properties.get("street"),
            properties.get("city"),
            properties.get("state"),
            properties.get("country"),
        ]
        display_name = ", ".join(
            str(part) for part in parts if part
        ) or search_query

        item = _normalise_geocode_result(
            lat,
            lon,
            display_name,
            properties.get("type", ""),
            properties.get("osm_id", ""),
            "Photon / OpenStreetMap",
        )
        if item:
            parsed.append(item)

    return parsed

def geocode_destination(query):
    """Find a real destination in India using OpenStreetMap geocoding."""
    original_query = str(query).strip()
    if not original_query:
        return []

    queries = [
        f"{original_query}, Vijayawada, Andhra Pradesh, India",
        f"{original_query}, Andhra Pradesh, India",
        f"{original_query}, India",
    ]
    queries = list(dict.fromkeys(queries))
    all_results = []

    for search_query in queries:
        try:
            results = _search_nominatim(search_query)
            all_results.extend(results)
            if all_results:
                return _deduplicate_geocode_results(all_results)[:8]
        except requests.exceptions.RequestException:
            break

    for search_query in queries:
        try:
            results = _search_photon(search_query)
            all_results.extend(results)
            if all_results:
                return _deduplicate_geocode_results(all_results)[:8]
        except requests.exceptions.RequestException:
            continue

    return _deduplicate_geocode_results(all_results)[:8]


@st.cache_data(ttl=900, show_spinner=False)
def get_driving_route(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    Retrieve a real road route from the public OSRM routing service.
    """

    coordinates = (
        f"{origin_lon},{origin_lat};"
        f"{dest_lon},{dest_lat}"
    )

    url = f"{OSRM_URL}/{coordinates}"

    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
        "alternatives": "true",
    }

    response = requests.get(
        url,
        params=params,
        headers=HTTP_HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("code") != "Ok":
        return {
            "success": False,
            "message": data.get("message", "Route not found."),
        }

    routes = data.get("routes", [])

    if not routes:
        return {
            "success": False,
            "message": "No route was returned.",
        }

    parsed_routes = []

    for index, route in enumerate(routes[:3], start=1):
        parsed_routes.append(
            {
                "route_number": index,
                "distance_km": route.get("distance", 0) / 1000,
                "duration_min": route.get("duration", 0) / 60,
                "geometry": route.get("geometry"),
            }
        )

    return {
        "success": True,
        "routes": parsed_routes,
        "source": "OSRM",
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_nearby_transport(latitude, longitude, radius=2000, area="Starting Point"):
    """Find mapped transport features around a coordinate using OpenStreetMap."""
    columns = ["name", "type", "latitude", "longitude", "distance_km", "area"]
    query = f"""
    [out:json][timeout:20];
    (
      nwr(around:{int(radius)},{float(latitude)},{float(longitude)})["highway"="bus_stop"];
      nwr(around:{int(radius)},{float(latitude)},{float(longitude)})["public_transport"~"platform|stop_position|station"];
      nwr(around:{int(radius)},{float(latitude)},{float(longitude)})["railway"~"station|halt|tram_stop|subway_entrance"];
      nwr(around:{int(radius)},{float(latitude)},{float(longitude)})["amenity"~"bus_station|taxi"];
    );
    out center tags;
    """
    try:
        response = requests.post(
            OVERPASS_URL, data=query, headers=HTTP_HEADERS, timeout=25
        )
        response.raise_for_status()
        elements = response.json().get("elements", [])
    except (requests.exceptions.RequestException, ValueError):
        return pd.DataFrame(columns=columns)

    rows = []
    for element in elements:
        tags = element.get("tags", {})
        lat, lon = element.get("lat"), element.get("lon")
        if lat is None or lon is None:
            center = element.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue

        name = tags.get("name") or tags.get("name:en") or "Unnamed transport point"
        transport_type = (
            tags.get("railway") or tags.get("amenity") or
            tags.get("public_transport") or tags.get("highway") or "transport"
        )
        rows.append({
            "name": str(name),
            "type": str(transport_type).replace("_", " ").title(),
            "latitude": float(lat),
            "longitude": float(lon),
            "distance_km": haversine_km(latitude, longitude, lat, lon),
            "area": area,
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    return (
        result.sort_values("distance_km")
        .drop_duplicates(subset=["name", "type", "latitude", "longitude"])
        .head(30)
        .reset_index(drop=True)
    )


def build_travel_map(
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    destination_name,
    route,
    transport_df,
):
    """Display the exact GPS point, selected destination and returned route."""

    origin = [float(origin_lat), float(origin_lon)]
    destination = [float(destination_lat), float(destination_lon)]

    travel_map = folium.Map(
        location=origin,
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )

    # English-labelled street basemap. The route itself comes from OSRM.
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        name="English Street Map",
        attr=(
            "Tiles © Esri — Source: Esri, Garmin, FAO, NOAA, USGS, "
            "OpenStreetMap contributors, and the GIS User Community"
        ),
        overlay=False,
        control=False,
    ).add_to(travel_map)

    folium.Marker(
        origin,
        tooltip="Current GPS Location",
        popup=folium.Popup(
            f"<b>Starting Point</b><br>"
            f"Latitude: {origin[0]:.6f}<br>"
            f"Longitude: {origin[1]:.6f}",
            max_width=320,
        ),
        icon=folium.Icon(
            color="blue",
            icon="location-arrow",
            prefix="fa",
        ),
    ).add_to(travel_map)

    folium.Marker(
        destination,
        tooltip="Selected Destination",
        popup=folium.Popup(
            f"<b>Destination</b><br>{destination_name}<br>"
            f"Latitude: {destination[0]:.6f}<br>"
            f"Longitude: {destination[1]:.6f}",
            max_width=380,
        ),
        icon=folium.Icon(
            color="red",
            icon="flag",
            prefix="fa",
        ),
    ).add_to(travel_map)

    # Fit exactly to the returned OSRM road geometry.
    route_points = [origin, destination]
    if route and route.get("geometry"):
        geometry = route["geometry"]
        coordinates = geometry.get("coordinates", [])
        points = [
            [float(coord[1]), float(coord[0])]
            for coord in coordinates
            if isinstance(coord, (list, tuple)) and len(coord) >= 2
        ]
        if points:
            route_points = points

        folium.GeoJson(
            geometry,
            name="Real OSRM Road Route",
            style_function=lambda feature: {
                "color": "#1976d2",
                "weight": 6,
                "opacity": 0.9,
            },
            tooltip="Real road route returned by OSRM",
        ).add_to(travel_map)

    if transport_df is not None and not transport_df.empty:
        for _, row in transport_df.head(30).iterrows():
            transport_type = str(row["type"])
            if "Railway" in transport_type:
                icon_color, icon_name = "green", "train"
            elif "Bus" in transport_type:
                icon_color, icon_name = "orange", "bus"
            elif "Taxi" in transport_type:
                icon_color, icon_name = "purple", "taxi"
            else:
                icon_color, icon_name = "cadetblue", "info-sign"

            popup = (
                f"<b>{row['name']}</b><br>"
                f"Type: {transport_type}<br>"
                f"Near: {row.get('area', 'Journey')}<br>"
                f"Distance from that endpoint: {row['distance_km']:.2f} km"
            )
            folium.Marker(
                [float(row["latitude"]), float(row["longitude"])],
                tooltip=str(row["name"]),
                popup=folium.Popup(popup, max_width=320),
                icon=folium.Icon(
                    color=icon_color,
                    icon=icon_name,
                    prefix="fa",
                ),
            ).add_to(travel_map)

    travel_map.fit_bounds(
        route_points,
        padding=(30, 30),
        max_zoom=16,
    )

    return travel_map


# ============================================================
# LOAD DATA
# ============================================================

clean_df = safe_read_csv(CLEAN_FILE)
features_df = safe_read_csv(FEATURE_FILE)
risk_df = safe_read_csv(RISK_FILE)
cluster_df = safe_read_csv(CLUSTER_FILE)
anomaly_df = safe_read_csv(ANOMALY_FILE)


# ============================================================
# DATA PREPARATION
# ============================================================

if not features_df.empty:

    if "state" in features_df.columns:
        features_df["state"] = (
            features_df["state"].astype(str).str.strip()
        )

    if not risk_df.empty and "state" in risk_df.columns:

        risk_df["state"] = (
            risk_df["state"].astype(str).str.strip()
        )

        risk_columns = ["state"]

        for column in [
            "risk_indicator",
            "risk_score",
            "risk_category",
        ]:
            if column in risk_df.columns:
                risk_columns.append(column)

        features_df = features_df.merge(
            risk_df[
                list(dict.fromkeys(risk_columns))
            ],
            on="state",
            how="left",
            suffixes=("", "_risk"),
        )

    if "risk_indicator" in features_df.columns:
        features_df["risk_indicator"] = pd.to_numeric(
            features_df["risk_indicator"],
            errors="coerce",
        )
        features_df["risk_score"] = features_df["risk_indicator"]

    elif "risk_score" in features_df.columns:
        features_df["risk_score"] = pd.to_numeric(
            features_df["risk_score"],
            errors="coerce",
        )
        features_df["risk_indicator"] = features_df["risk_score"]

    else:
        fallback = [
            "total_rate",
            "rape_rate",
            "dowry_rate",
            "cruelty_rate",
            "kidnapping_rate",
            "assault_rate",
            "cyber_rate",
            "pocso_rate",
        ]

        available = [
            col for col in fallback
            if col in features_df.columns
        ]

        if available:
            normalized = features_df[available].copy()

            for column in available:
                minimum = normalized[column].min()
                maximum = normalized[column].max()

                if maximum != minimum:
                    normalized[column] = (
                        (normalized[column] - minimum)
                        / (maximum - minimum)
                    ) * 100
                else:
                    normalized[column] = 0

            features_df["risk_score"] = normalized.mean(axis=1)
            features_df["risk_indicator"] = features_df["risk_score"]

    if "risk_score" in features_df.columns:
        features_df["risk_category"] = features_df["risk_score"].apply(
            get_risk_category
        )

    if not cluster_df.empty and "state" in cluster_df.columns:
        cluster_df["state"] = (
            cluster_df["state"].astype(str).str.strip()
        )

        columns = ["state"]

        for column in ["cluster", "cluster_label"]:
            if column in cluster_df.columns:
                columns.append(column)

        features_df = features_df.merge(
            cluster_df[
                list(dict.fromkeys(columns))
            ],
            on="state",
            how="left",
            suffixes=("", "_cluster"),
        )

    if not anomaly_df.empty and "state" in anomaly_df.columns:
        anomaly_df["state"] = (
            anomaly_df["state"].astype(str).str.strip()
        )

        columns = ["state"]

        for column in ["anomaly", "anomaly_score"]:
            if column in anomaly_df.columns:
                columns.append(column)

        features_df = features_df.merge(
            anomaly_df[
                list(dict.fromkeys(columns))
            ],
            on="state",
            how="left",
            suffixes=("", "_anomaly"),
        )


# ============================================================
# ISOLATION FOREST FALLBACK
# ============================================================

# If the processed anomaly file is missing or lacks usable labels, fit the
# model from available numeric crime-rate features in the feature dataset.
if not features_df.empty and "state" in features_df.columns:
    anomaly_aliases = ["is_anomaly", "prediction", "anomaly_label"]
    if "anomaly" not in features_df.columns:
        for alias in anomaly_aliases:
            if alias in features_df.columns:
                features_df = features_df.rename(columns={alias: "anomaly"})
                break

    if "anomaly" not in features_df.columns or features_df["anomaly"].isna().all():
        candidate_features = [
            "total_rate", "rape_rate", "dowry_rate", "cruelty_rate",
            "kidnapping_rate", "assault_rate", "cyber_rate", "pocso_rate",
        ]
        available = [c for c in candidate_features if c in features_df.columns]
        if len(available) >= 2 and len(features_df) >= 3:
            matrix = features_df[available].apply(pd.to_numeric, errors="coerce")
            matrix = matrix.replace([np.inf, -np.inf], np.nan)
            matrix = matrix.dropna(axis=1, how="all")
            if matrix.shape[1] >= 2:
                matrix = matrix.fillna(matrix.median()).fillna(0)
                try:
                    isolation_model = IsolationForest(
                        n_estimators=200, contamination="auto", random_state=42
                    )
                    features_df["anomaly"] = isolation_model.fit_predict(matrix)
                    features_df["anomaly_score"] = isolation_model.decision_function(matrix)
                except Exception as exc:
                    warnings.warn(f"Isolation Forest training failed: {exc}")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ HerShield")
    st.caption("Women Safety & Risk Analytics")
    st.divider()

    page_labels = {
        "Dashboard": "🏠 Dashboard",
        "Safety Map": "🗺️ Safety Map",
        "Live GPS": "🛡️ Live Safety",
        "Safe Travel": "🧭 Safe Travel",
        "Crime Analytics": "📊 Crime Analytics",
        "ML Insights": "🤖 ML Insights",
        "Report Verification": "🔎 Report Verification",
        "About": "ℹ️ About",
    }

    page_options = list(page_labels.keys())

    page = st.radio(
        "Navigation",
        page_options,
        format_func=lambda item: page_labels[item],
    )

    st.divider()

    st.caption("Data Source")
    st.caption("NCRB Crime in India 2023")
    st.caption("Official Government of India data")


# ============================================================
# DATA CHECK
# ============================================================

if features_df.empty:
    st.error("Processed data could not be loaded.")
    st.info(
        "Make sure the required files exist inside "
        "data/processed/."
    )
    st.stop()


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🛡️ HerShield Dashboard")

    st.markdown(
        """
        **HerShield** is a data-driven women safety analytics
        platform using official NCRB 2023 crime statistics,
        geospatial visualization and machine learning.
        """
    )

    st.divider()

    total_cases = (
        features_df["total_cases"].sum()
        if "total_cases" in features_df.columns
        else 0
    )

    total_states = len(features_df)

    avg_rate = (
        features_df["total_rate"].mean()
        if "total_rate" in features_df.columns
        else 0
    )

    avg_risk = (
        features_df["risk_score"].mean()
        if "risk_score" in features_df.columns
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric(
            "Total Reported Cases",
            format_number(total_cases),
        )

    with col2:
        create_metric(
            "States / UTs",
            format_number(total_states),
        )

    with col3:
        create_metric(
            "Average Crime Rate",
            format_decimal(avg_rate, 2),
        )

    with col4:
        create_metric(
            "Average Risk Indicator",
            format_decimal(avg_risk, 2),
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📊 States / UTs by Total Reported Cases")

        if "total_cases" in features_df.columns:
            top_cases = (
                features_df[
                    ["state", "total_cases"]
                ]
                .sort_values(
                    "total_cases",
                    ascending=False,
                )
                .head(10)
            )

            fig = px.bar(
                top_cases,
                x="total_cases",
                y="state",
                orientation="h",
                title="Top 10 by Total Cases",
                labels={
                    "total_cases": "Cases",
                    "state": "State / UT",
                },
            )

            fig.update_layout(
                template="plotly_dark",
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("⚠️ Risk Indicator by State / UT")

        risk_chart = (
            features_df[
                ["state", "risk_score"]
            ]
            .sort_values(
                "risk_score",
                ascending=False,
            )
            .head(10)
        )

        fig = px.bar(
            risk_chart,
            x="risk_score",
            y="state",
            orientation="h",
            title="Highest Risk Indicators",
            labels={
                "risk_score": "Risk Indicator",
                "state": "State / UT",
            },
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("📈 Risk Indicator Distribution")

    category_counts = (
        features_df["risk_category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = ["Category", "Count"]

    fig = px.pie(
        category_counts,
        names="Category",
        values="Count",
        hole=0.45,
        title="Risk Indicator Categories",
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, width="stretch")

    st.info(
        "Risk indicators are analytical values derived from "
        "selected NCRB 2023 crime-rate features. They are not "
        "guarantees of whether a location is safe or unsafe."
    )


# ============================================================
# PAGE 2 — SAFETY MAP
# ============================================================

elif page == "Safety Map":

    st.title("🗺️ India Safety Risk Map")

    st.markdown(
        """
        This map visualizes the **state/UT-level analytical
        risk indicator** derived from NCRB 2023 data.
        """
    )

    st.info(
        "The map represents historical state/UT-level data. "
        "It is not a live street-level crime map."
    )

    india_map = folium.Map(
        location=[22.5, 79.0],
        zoom_start=5,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=False,
        attr="© OpenStreetMap contributors",
    ).add_to(india_map)

    for _, row in features_df.iterrows():

        state = str(row["state"])
        score = row.get("risk_score", np.nan)
        category = row.get(
            "risk_category",
            get_risk_category(score),
        )

        geo_path = os.path.join(
            MAP_DIR,
            state_to_filename(state) + ".geojson",
        )

        if not os.path.exists(geo_path):
            continue

        geojson_data = load_geojson(geo_path)

        if geojson_data is None:
            continue

        state_color = risk_color(score)

        state_style = {
            "color": "#ffffff",
            "weight": 1,
            "fillColor": state_color,
            "fillOpacity": 0.65,
            "opacity": 1,
        }

        tooltip_text = (
            f"{state} | Risk Indicator: "
            f"{format_decimal(score, 2)} | {category}"
        )

        popup_html = f"""
        <div style="font-family:Arial; width:250px;">
            <h4>{state}</h4>
            <b>Risk Indicator:</b> {format_decimal(score, 2)}
            <br><br>
            <b>Category:</b> {category}
            <br><br>
            <b>Data:</b> NCRB 2023
        </div>
        """

        folium.GeoJson(
            geojson_data,
            name=state,
            style_function=lambda feature, style=state_style: style,
            highlight_function=lambda feature: {
                "weight": 3,
                "color": "#ffffff",
                "fillOpacity": 0.85,
            },
            tooltip=folium.Tooltip(tooltip_text),
            popup=folium.Popup(
                popup_html,
                max_width=350,
            ),
        ).add_to(india_map)

    st_folium(
        india_map,
        width="100%",
        height=650,
        returned_objects=[],
    )

    st.divider()

    st.subheader("📌 Risk Indicator Legend")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("🟢 Lower Indicator\n\n0 – 33.32")

    with c2:
        st.warning("🟡 Moderate Indicator\n\n33.33 – 66.66")

    with c3:
        st.error("🔴 Higher Indicator\n\n66.67 – 100")


# ============================================================
# PAGE 3 — LIVE GPS
# ============================================================

elif page == "Live GPS":

    st.title("📍 Live GPS Location Analysis")

    st.markdown(
        """
        Use your browser's current location to connect the
        location with HerShield's historical NCRB 2023
        state-level crime analytics and machine-learning results.
        """
    )

    st.info(
        "Privacy: GPS coordinates are used only for the current "
        "analysis. They are not written into the NCRB dataset."
    )

    st.divider()

    gps_data = display_gps()

    if gps_data is None:

        st.warning(
            "Waiting for GPS location. Allow location access "
            "in your browser."
        )

    else:

        latitude = None
        longitude = None
        accuracy = None

        if isinstance(gps_data, dict):

            latitude = gps_data.get("latitude")
            longitude = gps_data.get("longitude")
            accuracy = gps_data.get("accuracy")

        elif isinstance(gps_data, (list, tuple)):

            if len(gps_data) >= 2:
                latitude = gps_data[0]
                longitude = gps_data[1]

            if len(gps_data) >= 3:
                accuracy = gps_data[2]

        if latitude is None or longitude is None:

            st.warning(
                "GPS coordinates were not received. "
                "Please allow location access and try again."
            )

        else:

            try:

                latitude = float(latitude)
                longitude = float(longitude)

                st.session_state["gps_latitude"] = latitude
                st.session_state["gps_longitude"] = longitude
                st.session_state["gps_accuracy"] = accuracy
                st.session_state["gps_available"] = True

                st.subheader("📍 Current GPS Location")

                c1, c2, c3 = st.columns(3)

                with c1:
                    create_metric(
                        "Latitude",
                        f"{latitude:.5f}",
                    )

                with c2:
                    create_metric(
                        "Longitude",
                        f"{longitude:.5f}",
                    )

                with c3:
                    create_metric(
                        "GPS Accuracy",
                        (
                            f"{float(accuracy):.1f} m"
                            if accuracy is not None
                            else "Unavailable"
                        ),
                    )

                st.divider()

                st.subheader("🗺️ Your Current Location")

                gps_map = folium.Map(
                    location=[latitude, longitude],
                    zoom_start=13,
                    tiles="OpenStreetMap",
                    control_scale=True,
                )

                folium.Marker(
                    [latitude, longitude],
                    tooltip="Current GPS Location",
                    popup=folium.Popup(
                        f"""
                        <b>Current Location</b><br><br>
                        Latitude: {latitude:.5f}<br>
                        Longitude: {longitude:.5f}
                        """,
                        max_width=300,
                    ),
                    icon=folium.Icon(
                        color="blue",
                        icon="location-arrow",
                        prefix="fa",
                    ),
                ).add_to(gps_map)

                st_folium(
                    gps_map,
                    width="100%",
                    height=450,
                    returned_objects=[],
                )

                st.divider()

                st.subheader("🛡️ Live Safety Assessment")

                gps_analysis = analyze_gps_location(
                    latitude,
                    longitude,
                )

                if not gps_analysis.get("success", False):

                    st.error(
                        gps_analysis.get(
                            "message",
                            "Unable to analyze this location.",
                        )
                    )

                else:

                    detected_state = gps_analysis.get(
                        "state",
                        "Unknown",
                    )

                    risk_score = gps_analysis.get(
                        "risk_score",
                        None,
                    )

                    risk_category = gps_analysis.get(
                        "risk_category",
                        "Unavailable",
                    )

                    geographic_method = gps_analysis.get(
                        "geographic_method",
                        "GeoJSON point-in-polygon",
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        create_metric(
                            "Latitude",
                            f"{latitude:.5f}",
                        )

                    with c2:
                        create_metric(
                            "Longitude",
                            f"{longitude:.5f}",
                        )

                    with c3:
                        create_metric(
                            "Detected State / UT",
                            detected_state,
                        )

                    st.info(
                        f"📍 GPS point matched to **{detected_state}** "
                        f"using **{geographic_method}**."
                    )

                    st.markdown("### 📊 Historical Risk Context")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        create_metric(
                            "Risk Indicator",
                            format_decimal(risk_score, 2),
                        )

                    with c2:
                        create_metric(
                            "Risk Category",
                            risk_category,
                        )

                    with c3:
                        create_metric(
                            "Data Source",
                            "NCRB 2023",
                        )

                    if risk_score is not None:

                        score = float(risk_score)

                        if score < 33.33:
                            st.success(
                                f"🟢 Historical analytical risk "
                                f"indicator for **{detected_state}**: "
                                f"**{score:.2f}**."
                            )

                        elif score < 66.67:
                            st.warning(
                                f"🟡 Historical analytical risk "
                                f"indicator for **{detected_state}**: "
                                f"**{score:.2f}**."
                            )

                        else:
                            st.error(
                                f"🔴 Historical analytical risk "
                                f"indicator for **{detected_state}**: "
                                f"**{score:.2f}**."
                            )

                    st.caption(
                        "This is historical NCRB 2023 state/UT-level "
                        "analysis. It is not a live crime report and "
                        "does not determine whether the exact GPS "
                        "location is currently safe or unsafe."
                    )

                    st.divider()

                    st.subheader(
                        f"📈 {detected_state} Historical Crime Profile"
                    )

                    matching_rows = features_df[
                        features_df["state"].astype(str).str.lower()
                        == detected_state.lower()
                    ]

                    if not matching_rows.empty:

                        state_row = matching_rows.iloc[0]

                        profile_items = [
                            ("Total Cases", "total_cases", "number"),
                            ("Total Crime Rate", "total_rate", "decimal"),
                            ("Rape Cases", "rape_cases", "number"),
                            ("Rape Rate", "rape_rate", "decimal"),
                            ("Dowry Deaths", "dowry_deaths", "number"),
                            ("Dowry Rate", "dowry_rate", "decimal"),
                            ("Cruelty Cases", "cruelty_cases", "number"),
                            ("Cruelty Rate", "cruelty_rate", "decimal"),
                            ("Kidnapping Cases", "kidnapping_cases", "number"),
                            ("Kidnapping Rate", "kidnapping_rate", "decimal"),
                            ("Assault Cases", "assault_cases", "number"),
                            ("Assault Rate", "assault_rate", "decimal"),
                            ("Cyber Crime Cases", "cyber_cases", "number"),
                            ("Cyber Crime Rate", "cyber_rate", "decimal"),
                            ("POCSO Cases", "pocso_cases", "number"),
                            ("POCSO Rate", "pocso_rate", "decimal"),
                        ]

                        # Historical NCRB crime-rate profile chart
                        rate_items = [
                            ("Total Crime", "total_rate"),
                            ("Rape", "rape_rate"),
                            ("Dowry Death", "dowry_rate"),
                            ("Cruelty", "cruelty_rate"),
                            ("Kidnapping", "kidnapping_rate"),
                            ("Assault", "assault_rate"),
                            ("Cyber Crime", "cyber_rate"),
                            ("POCSO", "pocso_rate"),
                        ]

                        chart_rows = []
                        for label, column in rate_items:
                            if column not in features_df.columns:
                                continue
                            value = pd.to_numeric(
                                state_row.get(column),
                                errors="coerce",
                            )
                            if pd.notna(value):
                                chart_rows.append({
                                    "Crime Category": label,
                                    "Rate": float(value),
                                })

                        if chart_rows:
                            profile_chart = px.bar(
                                pd.DataFrame(chart_rows),
                                x="Crime Category",
                                y="Rate",
                                text="Rate",
                                title="NCRB 2023 Historical Crime Rates",
                            )
                            profile_chart.update_traces(
                                texttemplate="%{text:.2f}",
                                textposition="outside",
                            )
                            profile_chart.update_layout(
                                template="plotly_dark",
                                height=430,
                                margin=dict(l=20, r=20, t=70, b=80),
                                xaxis_title="Crime Category",
                                yaxis_title="Reported rate",
                                uniformtext_minsize=9,
                                uniformtext_mode="hide",
                            )
                            st.plotly_chart(
                                profile_chart,
                                use_container_width=True,
                                key="live_safety_crime_profile",
                            )
                        else:
                            st.info(
                                "No crime-rate fields were available for this state in the NCRB 2023 feature dataset."
                            )

                        # Keep the case counts visible below the chart.
                        count_items = [
                            ("Total Cases", "total_cases"),
                            ("Rape Cases", "rape_cases"),
                            ("Dowry Deaths", "dowry_deaths"),
                            ("Cruelty Cases", "cruelty_cases"),
                            ("Kidnapping Cases", "kidnapping_cases"),
                            ("Assault Cases", "assault_cases"),
                            ("Cyber Crime Cases", "cyber_cases"),
                            ("POCSO Cases", "pocso_cases"),
                        ]

                        rows = []
                        for label, column in count_items:
                            if column not in features_df.columns:
                                continue
                            value = pd.to_numeric(
                                state_row.get(column),
                                errors="coerce",
                            )
                            if pd.notna(value):
                                rows.append({
                                    "Metric": label,
                                    "Cases": int(round(float(value))),
                                })

                        if rows:
                            st.dataframe(
                                pd.DataFrame(rows),
                                use_container_width=True,
                                hide_index=True,
                            )

                    st.divider()

                    st.subheader(
                        "🤖 Supporting Machine Learning Insights"
                    )

                    if not matching_rows.empty:

                        state_row = matching_rows.iloc[0]
                        c1, c2 = st.columns(2)

                        with c1:

                            cluster_value = state_row.get(
                                "cluster",
                                None,
                            )

                            if cluster_value is None or pd.isna(cluster_value):
                                cluster_display = "Unavailable"
                            else:
                                try:
                                    cluster_display = str(
                                        int(float(cluster_value))
                                    )
                                except Exception:
                                    cluster_display = str(cluster_value)

                            st.metric(
                                "K-Means Cluster",
                                cluster_display,
                            )

                            st.caption(
                                "Cluster numbers are statistical labels "
                                "and do not mean safe or unsafe."
                            )

                        with c2:

                            # Read the anomaly result directly from the
                            # dedicated anomaly dataset. Different generated
                            # versions may call the prediction column
                            # anomaly, anomaly_label, is_anomaly, or similar.
                            anomaly_value = None

                            if not anomaly_df.empty and "state" in anomaly_df.columns:
                                anomaly_matches = anomaly_df[
                                    anomaly_df["state"].astype(str).str.strip().str.lower()
                                    == detected_state.strip().lower()
                                ]

                                if not anomaly_matches.empty:
                                    anomaly_row = anomaly_matches.iloc[0]
                                    candidate_columns = [
                                        "anomaly",
                                        "anomaly_label",
                                        "is_anomaly",
                                        "anomaly_prediction",
                                        "prediction",
                                    ]

                                    for candidate in candidate_columns:
                                        if candidate in anomaly_df.columns:
                                            anomaly_value = anomaly_row.get(candidate)
                                            if pd.notna(anomaly_value):
                                                break

                                    if anomaly_value is None:
                                        for column in anomaly_df.columns:
                                            lower_column = str(column).lower()
                                            if "anomaly" in lower_column and "score" not in lower_column:
                                                value = anomaly_row.get(column)
                                                if pd.notna(value):
                                                    anomaly_value = value
                                                    break

                            if anomaly_value is None:
                                anomaly_value = state_row.get("anomaly", None)

                            anomaly_result = get_anomaly_result(anomaly_value)

                            st.metric(
                                "Isolation Forest",
                                anomaly_result,
                            )

                            st.caption(
                                "An anomaly means an unusual feature "
                                "pattern, not a safety verdict."
                            )

                    st.divider()

                    st.subheader(
                        "🧠 How Live Safety Analysis Works"
                    )

                    st.markdown(
                        f"""
                        **Step 1 — Live GPS**

                        The browser provides the current latitude and longitude
                        after location permission is granted.

                        **Step 2 — Geographic Boundary Matching**

                        The GPS point is matched to a geographic boundary using
                        point-in-polygon analysis.

                        **Step 3 — Historical Crime Context**

                        The detected state/UT is matched with NCRB 2023 statistics.

                        **Step 4 — Risk Indicator**

                        HerShield retrieves the calculated historical risk indicator.

                        **Step 5 — Machine Learning Context**

                        K-Means and Isolation Forest provide additional statistical
                        context based on historical crime-rate patterns.

                        """
                    )

                    st.info(
                        "GPS identifies where the user is now. NCRB 2023 "
                        "provides historical state/UT-level data. The Safe "
                        "Travel module below uses the saved GPS position "
                        "with real map, route and mapped transportation data."
                    )

            except Exception as exc:

                st.error(
                    "An error occurred while processing the GPS location."
                )

                st.code(str(exc))


# ============================================================
# PAGE 4 — SAFE TRAVEL
# ============================================================

elif page == "Safe Travel":

    st.title("🧭 Safe Travel")

    st.markdown(
        """
        Plan a journey from your **current browser GPS location**
        to a destination using real-world map, route and mapped
        transportation information.
        """
    )

    st.caption(
        "OpenStreetMap data is community-maintained and may be incomplete "
        "or outdated in some locations."
    )

    st.divider()

    # --------------------------------------------------------
    # STARTING LOCATION
    # --------------------------------------------------------

    st.subheader("📍 Starting Location")

    gps_available = st.session_state.get(
        "gps_available",
        False,
    )

    origin_lat = st.session_state.get(
        "gps_latitude"
    )

    origin_lon = st.session_state.get(
        "gps_longitude"
    )

    origin_accuracy = st.session_state.get(
        "gps_accuracy"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        create_metric(
            "GPS Status",
            "Available" if gps_available else "Not Available",
        )

    with c2:
        create_metric(
            "Latitude",
            (
                f"{float(origin_lat):.5f}"
                if origin_lat is not None
                else "—"
            ),
        )

    with c3:
        create_metric(
            "Longitude",
            (
                f"{float(origin_lon):.5f}"
                if origin_lon is not None
                else "—"
            ),
        )

    if origin_accuracy is not None:
        st.caption(
            f"Current GPS accuracy: approximately "
            f"{float(origin_accuracy):.1f} m"
        )

    if not gps_available or origin_lat is None or origin_lon is None:

        st.warning(
            "Your current GPS is not available in this session. "
            "Open **🛡️ Live Safety**, allow browser location access, "
            "wait until the GPS coordinates appear, and then return "
            "to **🧭 Safe Travel**."
        )

        st.stop()

    st.success(
        "Your current GPS location is ready as the starting point."
    )

    st.divider()

    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    st.subheader("🎯 Destination")

    destination = st.text_input(
        "Enter your destination",
        placeholder="Example: Vijayawada Railway Station",
        help=(
            "Enter a place, landmark, address, railway station, "
            "bus station, college, hospital or other destination."
        ),
    )

    analyze_travel = st.button(
        "🔎 Analyze Travel",
        type="primary",
        use_container_width=True,
    )

    if analyze_travel:

        if not destination.strip():

            st.warning(
                "Please enter a destination."
            )

        else:

            with st.spinner(
                "Searching OpenStreetMap for the destination..."
            ):

                try:

                    destination_results = geocode_destination(
                        destination.strip()
                    )

                    if not destination_results:

                        st.session_state[
                            "travel_destination_candidates"
                        ] = []

                        st.warning(
                            "No destination matches were returned by OpenStreetMap."
                        )

                        st.info(
                            "Try a more specific place name or address, "
                            "for example: Vijayawada Railway Station, "
                            "Benz Circle Vijayawada, VIT-AP University, "
                            "or Vijayawada Bus Station."
                        )

                    else:

                        st.session_state[
                            "travel_destination_candidates"
                        ] = destination_results

                        st.session_state[
                            "travel_destination"
                        ] = destination.strip()

                        st.session_state[
                            "travel_destination_lat"
                        ] = None

                        st.session_state[
                            "travel_destination_lon"
                        ] = None

                        st.session_state[
                            "travel_destination_display"
                        ] = None

                        st.session_state[
                            "travel_route"
                        ] = None

                        st.session_state["travel_transport"] = None
                        st.session_state["travel_origin_transport"] = None
                        st.session_state["travel_destination_transport"] = None

                        st.success(
                            f"Found {len(destination_results)} "
                            "possible destination match(es)."
                        )

                except requests.exceptions.Timeout:

                    st.error(
                        "The OpenStreetMap geocoding service took too long "
                        "to respond. Please try again."
                    )

                except requests.exceptions.RequestException as exc:

                    st.error(
                        "The destination geocoding service could not be reached."
                    )
                    st.caption(
                        "Please check your internet connection and try again."
                    )
                    st.code(str(exc))

                except Exception as exc:

                    st.error(
                        "An unexpected destination-search error occurred."
                    )
                    st.code(str(exc))

    # --------------------------------------------------------
    # DESTINATION MATCH SELECTION
    # --------------------------------------------------------

    destination_candidates = st.session_state.get(
        "travel_destination_candidates",
        [],
    )

    if destination_candidates:

        st.divider()
        st.subheader("📍 Select the Correct Destination")
        st.caption(
            "Several real OpenStreetMap matches may be returned. Choose the exact place before calculating the route."
        )

        destination_labels = [
            result["display_name"]
            for result in destination_candidates
        ]

        selected_label = st.selectbox(
            "OpenStreetMap matches",
            destination_labels,
            key="selected_destination_label",
            help=(
                "Choose the exact place you want to travel to. "
                "HerShield will calculate the route only after you select it."
            ),
        )

        selected_result = next(
            (
                result
                for result in destination_candidates
                if result["display_name"] == selected_label
            ),
            None,
        )

        if selected_result is not None:

            match_col1, match_col2 = st.columns(2)

            with match_col1:
                create_metric(
                    "Destination Latitude",
                    f"{selected_result['lat']:.5f}",
                )

            with match_col2:
                create_metric(
                    "Destination Longitude",
                    f"{selected_result['lon']:.5f}",
                )

            calculate_route = st.button(
                "🛣️ Calculate Real Route",
                type="primary",
                use_container_width=True,
                key="calculate_selected_route",
            )

            if calculate_route:

                with st.spinner(
                    "Calculating the real road route and nearby transport..."
                ):

                    try:

                        dest_lat = float(selected_result["lat"])
                        dest_lon = float(selected_result["lon"])
                        dest_display = selected_result["display_name"]

                        route_result = get_driving_route(
                            float(origin_lat),
                            float(origin_lon),
                            dest_lat,
                            dest_lon,
                        )

                        if not route_result.get("success", False):

                            st.error(
                                route_result.get(
                                    "message",
                                    "A road route could not be calculated.",
                                )
                            )

                        else:

                            origin_transport = get_nearby_transport(
                                float(origin_lat), float(origin_lon),
                                radius=2000, area="Starting Point",
                            )
                            destination_transport = get_nearby_transport(
                                dest_lat, dest_lon,
                                radius=2000, area="Destination",
                            )
                            transport_df = pd.concat(
                                [origin_transport, destination_transport],
                                ignore_index=True,
                            )
                            if not transport_df.empty:
                                transport_df = transport_df.drop_duplicates(
                                    subset=["name", "type", "latitude", "longitude", "area"]
                                ).reset_index(drop=True)
                            st.session_state["travel_origin_transport"] = origin_transport
                            st.session_state["travel_destination_transport"] = destination_transport

                            st.session_state[
                                "travel_destination_lat"
                            ] = dest_lat

                            st.session_state[
                                "travel_destination_lon"
                            ] = dest_lon

                            st.session_state[
                                "travel_destination_display"
                            ] = dest_display

                            st.session_state[
                                "travel_route"
                            ] = route_result

                            st.session_state[
                                "travel_transport"
                            ] = transport_df

                            st.success(
                                "Real route calculated successfully."
                            )

                    except requests.exceptions.Timeout:

                        st.error(
                            "The routing or transportation service took too long "
                            "to respond. Please try again."
                        )

                    except requests.exceptions.RequestException as exc:

                        st.error(
                            "A real-world map service could not be reached."
                        )
                        st.code(str(exc))

                    except Exception as exc:

                        st.error(
                            "An unexpected travel-analysis error occurred."
                        )
                        st.code(str(exc))

    # --------------------------------------------------------
    # DISPLAY SAVED TRAVEL RESULT
    # --------------------------------------------------------

    saved_destination = st.session_state.get(
        "travel_destination"
    )

    saved_dest_lat = st.session_state.get(
        "travel_destination_lat"
    )

    saved_dest_lon = st.session_state.get(
        "travel_destination_lon"
    )

    saved_dest_display = st.session_state.get(
        "travel_destination_display"
    )

    saved_route = st.session_state.get(
        "travel_route"
    )

    saved_transport = st.session_state.get(
        "travel_transport"
    )

    if (
        saved_destination
        and saved_dest_lat is not None
        and saved_dest_lon is not None
        and saved_route
        and saved_route.get("success")
    ):

        st.divider()

        st.subheader("🎯 Destination Found")

        st.success(
            f"**{saved_dest_display}**"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            create_metric(
                "Destination Latitude",
                f"{float(saved_dest_lat):.5f}",
            )

        with c2:
            create_metric(
                "Destination Longitude",
                f"{float(saved_dest_lon):.5f}",
            )

        with c3:
            straight_distance = haversine_km(
                float(origin_lat),
                float(origin_lon),
                float(saved_dest_lat),
                float(saved_dest_lon),
            )

            create_metric(
                "Straight-line Distance",
                f"{straight_distance:.2f} km",
            )

        st.divider()

        st.subheader("🛣️ Real Road Route Options")

        routes = saved_route.get("routes", [])

        route_rows = []

        for route in routes:

            route_rows.append(
                {
                    "Route": f"Route {route['route_number']}",
                    "Distance": f"{route['distance_km']:.2f} km",
                    "Estimated Driving Time": (
                        f"{route['duration_min']:.0f} min"
                    ),
                }
            )

        if route_rows:

            route_table_html = [
                '<div style="width:100%;overflow-x:auto;">',
                '<table style="width:100%;border-collapse:collapse;background:#111111;color:#ffffff;">',
                '<thead><tr>',
                '<th style="text-align:left;padding:12px;border-bottom:1px solid #444;">Route</th>',
                '<th style="text-align:left;padding:12px;border-bottom:1px solid #444;">Distance</th>',
                '<th style="text-align:left;padding:12px;border-bottom:1px solid #444;">Estimated Driving Time</th>',
                '</tr></thead><tbody>'
            ]

            for row in route_rows:
                route_table_html.append(
                    '<tr>'
                    f'<td style="padding:10px;border-bottom:1px solid #333;">{row["Route"]}</td>'
                    f'<td style="padding:10px;border-bottom:1px solid #333;">{row["Distance"]}</td>'
                    f'<td style="padding:10px;border-bottom:1px solid #333;">{row["Estimated Driving Time"]}</td>'
                    '</tr>'
                )

            route_table_html.append('</tbody></table></div>')
            st.markdown(''.join(route_table_html), unsafe_allow_html=True)

        selected_route = routes[0] if routes else None

        st.caption(
            "Route distance and duration are routing estimates. "
            "They can change with traffic, road conditions and routing data."
        )

        st.divider()

        st.subheader("🗺️ Travel Map")

        travel_map = build_travel_map(
            float(origin_lat),
            float(origin_lon),
            float(saved_dest_lat),
            float(saved_dest_lon),
            saved_dest_display,
            selected_route,
            saved_transport,
        )

        st_folium(
            travel_map,
            width="100%",
            height=650,
            returned_objects=[],
        )

        st.divider()

        st.subheader("🚌 Nearby Transportation")
        st.caption(
            "Mapped public-transport locations within 2 km of the starting point "
            "and destination. Results depend on OpenStreetMap coverage."
        )

        if saved_transport is None or saved_transport.empty:
            st.warning(
                "No mapped transport features were returned near either endpoint. "
                "This does not mean transport is unavailable; local map data may be incomplete."
            )
        else:
            display_transport = saved_transport[
                ["area", "name", "type", "distance_km"]
            ].copy()
            display_transport.columns = [
                "Nearby endpoint", "Name", "Type", "Distance from endpoint (km)"
            ]
            display_transport["Distance from endpoint (km)"] = (
                display_transport["Distance from endpoint (km)"].round(2)
            )
            st.dataframe(display_transport, width="stretch", hide_index=True)

        st.caption(
            "Transport points are OpenStreetMap-mapped locations, not live vehicle "
            "positions or confirmation that a service is operating."
        )

        st.divider()
        st.subheader("🚕 Ride Booking Shortcuts")
        pickup_lat = float(origin_lat)
        pickup_lon = float(origin_lon)
        drop_lat = float(saved_dest_lat)
        drop_lon = float(saved_dest_lon)
        drop_label = quote_plus(str(saved_dest_display))
        uber_url = (
            "https://m.uber.com/ul/?action=setPickup"
            f"&pickup[latitude]={pickup_lat}&pickup[longitude]={pickup_lon}"
            f"&dropoff[latitude]={drop_lat}&dropoff[longitude]={drop_lon}"
            f"&dropoff[nickname]={drop_label}"
        )
        maps_url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={pickup_lat},{pickup_lon}"
            f"&destination={drop_lat},{drop_lon}&travelmode=driving"
        )
        ride_col1, ride_col2, ride_col3 = st.columns(3)
        with ride_col1:
            st.link_button("🚕 Open Uber", uber_url, width="stretch")
        with ride_col2:
            st.link_button("🛵 Open Rapido", "https://www.rapido.bike/", width="stretch")
        with ride_col3:
            st.link_button("🗺️ Google Maps", maps_url, width="stretch")
        st.caption(
            "Uber may open with the pickup and destination prefilled, depending on device/app support. "
            "Rapido opens its service; enter or confirm pickup and drop-off in the app. "
            "Availability and fares are determined by the ride provider."
        )

        st.divider()

        st.subheader("🛡️ Historical Safety Context")

        travel_risk = analyze_gps_location(
            float(origin_lat),
            float(origin_lon),
        )

        if travel_risk.get("success"):

            detected_state = travel_risk.get(
                "state",
                "Unknown",
            )

            risk_score = travel_risk.get(
                "risk_score"
            )

            risk_category = travel_risk.get(
                "risk_category",
                "Unavailable",
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                create_metric(
                    "Current State / UT",
                    detected_state,
                )

            with c2:
                create_metric(
                    "Historical Risk Indicator",
                    format_decimal(risk_score, 2),
                )

            with c3:
                create_metric(
                    "Historical Category",
                    risk_category,
                )

            st.info(
                "This safety context describes historical NCRB 2023 "
                "state/UT-level crime patterns. It is not a live route "
                "safety score and does not guarantee that the journey "
                "or any place along it is safe."
            )

        else:

            st.warning(
                "Historical safety context could not be calculated "
                "for the current GPS position."
            )

        st.divider()

        st.subheader("ℹ️ Data Sources Used")

        st.markdown(
            """
            - **Starting point:** your browser's current GPS coordinates.
            - **Destination:** OpenStreetMap Nominatim geocoding.
            - **Road route:** OSRM routing service.
            - **Nearby transportation:** OpenStreetMap data queried through Overpass API.
            - **Historical safety context:** NCRB Crime in India 2023 data.
            """
        )

        st.warning(
            "Important: HerShield combines real map/route data with "
            "historical crime analytics. It does not provide a real-time "
            "crime alert, police dispatch service, or guarantee of safety."
        )


# ============================================================
# PAGE 5 — CRIME ANALYTICS
# ============================================================

elif page == "Crime Analytics":

    st.title("📊 Crime Analytics")

    st.markdown(
        """
        Explore major crime categories contained in the
        NCRB 2023 dataset.
        """
    )

    st.divider()

    available_rate_columns = [
        column
        for column in [
            "total_rate",
            "rape_rate",
            "dowry_rate",
            "cruelty_rate",
            "kidnapping_rate",
            "acid_attack_rate",
            "assault_rate",
            "trafficking_rate",
            "cyber_rate",
            "pocso_rate",
        ]
        if column in features_df.columns
    ]

    if not available_rate_columns:

        st.warning("No crime-rate columns are available.")

    else:

        selected_rate = st.selectbox(
            "Select Crime Rate",
            available_rate_columns,
        )

        chart_data = (
            features_df[
                ["state", selected_rate]
            ]
            .sort_values(
                selected_rate,
                ascending=False,
            )
            .head(15)
        )

        fig = px.bar(
            chart_data,
            x=selected_rate,
            y="state",
            orientation="h",
            title=(
                "Top States / UTs by "
                + selected_rate.replace("_", " ").title()
            ),
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("🔗 Crime Rate Correlation")

    correlation_columns = [
        column
        for column in [
            "total_rate",
            "rape_rate",
            "dowry_rate",
            "cruelty_rate",
            "kidnapping_rate",
            "acid_attack_rate",
            "assault_rate",
            "trafficking_rate",
            "cyber_rate",
            "pocso_rate",
        ]
        if column in features_df.columns
    ]

    if len(correlation_columns) >= 2:

        corr = features_df[correlation_columns].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Crime Rate Correlation Matrix",
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE 6 — ML INSIGHTS
# ============================================================

elif page == "ML Insights":

    st.title("🤖 ML Insights")

    st.markdown(
        """
        HerShield uses machine-learning methods to identify
        patterns in the historical crime-rate dataset.
        """
    )

    st.info(
        "ML outputs describe statistical patterns in historical "
        "data. They are not safety verdicts."
    )

    if "cluster" in features_df.columns:

        st.subheader("🔵 K-Means Clustering")

        cluster_summary = (
            features_df["cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        cluster_summary.columns = [
            "Cluster",
            "Number of States / UTs",
        ]

        st.dataframe(
            cluster_summary,
            width="stretch",
            hide_index=True,
        )

        x_options = [
            col
            for col in [
                "total_rate",
                "rape_rate",
                "cruelty_rate",
                "assault_rate",
                "cyber_rate",
                "pocso_rate",
            ]
            if col in features_df.columns
        ]

        if len(x_options) >= 2:

            c1, c2 = st.columns(2)

            with c1:
                x_axis = st.selectbox(
                    "X-axis",
                    x_options,
                    key="ml_x_axis",
                )

            with c2:
                y_axis = st.selectbox(
                    "Y-axis",
                    x_options,
                    index=1,
                    key="ml_y_axis",
                )

            fig = px.scatter(
                features_df,
                x=x_axis,
                y=y_axis,
                color="cluster",
                hover_name="state",
                title="K-Means State / UT Clusters",
            )

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, width="stretch")

    else:

        st.warning("K-Means cluster information is not available.")

    st.divider()

    st.subheader("🟣 Isolation Forest")

    if "anomaly" not in features_df.columns:

        st.warning(
            "Isolation Forest could not be trained because the dataset does not "
            "contain at least two usable numeric crime-rate features."
        )

    else:

        anomaly_columns = ["state", "anomaly"]
        if "anomaly_score" in features_df.columns:
            anomaly_columns.append("anomaly_score")
        anomaly_display = features_df[anomaly_columns].copy()

        anomaly_display["Result"] = (
            anomaly_display["anomaly"]
            .apply(get_anomaly_result)
        )

        display_columns = ["state", "Result"]
        if "anomaly_score" in anomaly_display.columns:
            display_columns.append("anomaly_score")
        st.dataframe(
            anomaly_display[display_columns],
            width="stretch",
            hide_index=True,
        )

        result_counts = (
            anomaly_display["Result"]
            .value_counts()
            .reset_index()
        )

        result_counts.columns = ["Result", "Count"]

        fig = px.pie(
            result_counts,
            names="Result",
            values="Count",
            hole=0.45,
            title="Normal vs Anomalous States / UTs",
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE 7 — AI CRIME REPORT IMAGE VERIFICATION
# ============================================================

elif page == "Report Verification":

    st.title("🔎 AI-Based Crime Report Verification")
    st.markdown(
        "Upload a photograph attached to a reported incident for preliminary "
        "image screening. This module checks image metadata and duplicate "
        "indicators and provides an experimental AI-generated-image screening "
        "signal when a trained model is installed."
    )

    st.warning(
        "This tool cannot determine whether a crime actually occurred or prove "
        "that a report is fake. It only identifies indicators that may deserve "
        "human review. Never automatically reject or penalise a report based on "
        "this result."
    )

    with st.form("crime_report_verification_form"):
        report_description = st.text_area(
            "Incident description",
            placeholder="Describe what was reported. Avoid including unnecessary personal details.",
            height=110,
        )
        report_location = st.text_input(
            "Reported location (optional)",
            placeholder="Area, city or landmark",
        )
        report_datetime = st.text_input(
            "Reported date and time (optional)",
            placeholder="Example: 2026-11-16 20:30",
        )
        evidence_file = st.file_uploader(
            "Upload incident photograph",
            type=["jpg", "jpeg", "png", "webp"],
            help="The uploaded image is analysed in memory and is not saved by this page.",
        )
        save_fingerprint = st.checkbox(
            "With my authorisation, save only this image's fingerprint to help identify repeated uploads later (the photograph itself will not be saved).",
            value=False,
        )
        run_verification = st.form_submit_button(
            "Analyse Evidence",
            type="primary",
            width="stretch",
        )

    if run_verification:

        if evidence_file is None:
            st.error("Please upload an image before starting the analysis.")

        else:
            image_bytes = evidence_file.getvalue()
            sha256_hash = hashlib.sha256(image_bytes).hexdigest()

            try:
                image = Image.open(BytesIO(image_bytes))
                image.load()
                image_rgb = image.convert("RGB")
            except Exception:
                st.error("The uploaded file could not be read as a valid image.")
                image = None
                image_rgb = None

            if image is not None:

                st.divider()
                st.subheader("1. Uploaded evidence")

                preview_col, details_col = st.columns([1, 1])
                with preview_col:
                    st.image(image_rgb, caption="Uploaded evidence", width="stretch")
                with details_col:
                    st.write(f"**File name:** {evidence_file.name}")
                    st.write(f"**Format:** {image.format or 'Unknown'}")
                    st.write(f"**Dimensions:** {image.width} × {image.height} pixels")
                    st.write(f"**Colour mode:** {image.mode}")
                    st.write(f"**SHA-256:** `{sha256_hash[:24]}…`")

                st.subheader("2. Metadata screening")
                exif_data = get_image_exif(image)
                if exif_data:
                    st.success("Some EXIF metadata is available.")
                    st.dataframe(
                        pd.DataFrame(
                            [{"Metadata field": key, "Value": value} for key, value in exif_data.items()]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.info(
                        "No readable EXIF metadata was found. This is common for images "
                        "shared through messaging apps or social media and is not evidence of fraud."
                    )

                # Compare against fingerprints previously saved with explicit consent.
                st.subheader("3. Duplicate-image screening")
                current_ahash = calculate_average_hash(image_rgb)
                history = get_image_fingerprint_history()
                exact_match = False
                near_match = None

                if not history.empty:
                    exact_match = bool((history["sha256"].astype(str) == sha256_hash).any())
                    distances = history["average_hash"].astype(str).apply(
                        lambda value: hamming_distance(current_ahash, value)
                    )
                    valid_distances = distances.dropna()
                    if not valid_distances.empty:
                        minimum_distance = int(valid_distances.min())
                        if minimum_distance <= 5:
                            near_match = minimum_distance

                if exact_match:
                    st.warning(
                        "An exact match was found in the previously consented fingerprint records. "
                        "This may indicate a repeated upload; it does not prove the report is false."
                    )
                elif near_match is not None:
                    st.warning(
                        f"A visually similar image was found (perceptual hash distance: {near_match}). "
                        "A reviewer should check whether the image is reused or legitimately related."
                    )
                else:
                    st.success("No matching image was found in the available fingerprint records.")

                if save_fingerprint:
                    os.makedirs(REPORT_VERIFICATION_DIR, exist_ok=True)
                    new_record = pd.DataFrame([{
                        "sha256": sha256_hash,
                        "average_hash": current_ahash,
                        "recorded_at": datetime.now().isoformat(timespec="seconds"),
                    }])
                    updated_history = pd.concat([history, new_record], ignore_index=True)
                    updated_history.to_csv(IMAGE_FINGERPRINT_FILE, index=False)
                    st.success(
                        "Image fingerprint saved with your consent. The uploaded photograph was not saved."
                    )

                st.subheader("4. Deep Learning image analysis")

                if os.path.exists(IMAGE_MODEL_FILE):
                    try:
                        model = load_image_authenticity_model(
                            IMAGE_MODEL_FILE,
                            os.path.getmtime(IMAGE_MODEL_FILE),
                        )

                        # The CNN expects 32 x 32 RGB images.
                        # Its internal Rescaling layer handles 0-255 conversion.
                        model_input = image_rgb.resize(
                            (32, 32),
                            Image.Resampling.LANCZOS,
                        )

                        model_array = np.asarray(
                            model_input,
                            dtype=np.float32,
                        )
                        model_array = np.expand_dims(model_array, axis=0)

                        prediction = np.asarray(
                            model.predict(model_array, verbose=0)
                        ).squeeze()

                        if prediction.size != 1:
                            st.error(
                                "The model did not return the expected single "
                                "REAL-class output."
                            )
                        else:
                            probability_real = float(prediction)

                            if 0 <= probability_real <= 1:
                                ai_generated_score = 1.0 - probability_real

                                # Experimental class boundary, not proof of authenticity.
                                if ai_generated_score >= 0.5:
                                    model_label = "FAKE-like"
                                else:
                                    model_label = "REAL-like"

                                st.markdown("#### CNN screening result")

                                st.metric(
                                    "Experimental model output",
                                    model_label,
                                )

                                st.metric(
                                    "AI-generated screening score",
                                    f"{ai_generated_score * 100:.1f}%",
                                )

                                st.caption(
                                    "This score is derived from the model's output "
                                    "for the FAKE training class. It is not a calibrated "
                                    "probability that the uploaded photograph is "
                                    "AI-generated, manipulated, or fraudulent."
                                )

                                if ai_generated_score >= 0.5:
                                    st.warning(
                                        "The model output is closer to its FAKE "
                                        "training class. This image may be referred "
                                        "for human review, but this result does not "
                                        "prove that the image is fake."
                                    )
                                else:
                                    st.info(
                                        "The model output is closer to its REAL "
                                        "training class. This does not prove that "
                                        "the image is genuine or that it depicts "
                                        "the reported incident."
                                    )

                                with st.expander(
                                    "View CNN model evaluation and limitations"
                                ):
                                    st.markdown("**Independent CIFAKE test results**")

                                    st.metric(
                                        "Overall test accuracy",
                                        "93.78%",
                                    )

                                    st.caption(
                                        "Evaluated on 20,000 CIFAKE test images. "
                                        "This is dataset performance, NOT the "
                                        "confidence or accuracy for this uploaded image."
                                    )

                                    st.markdown("**Class-wise accuracy**")
                                    st.write("- FAKE images: 92.50%")
                                    st.write("- REAL images: 95.07%")

                                    st.markdown("**Confusion matrix**")
                                    st.markdown(
                                        "| Actual class | Predicted FAKE | Predicted REAL |\n"
                                        "|---|---:|---:|\n"
                                        "| FAKE | 9,250 | 750 |\n"
                                        "| REAL | 493 | 9,507 |"
                                    )

                                    st.markdown("**Observed classification errors**")
                                    st.write(
                                        "- 493 REAL test images were incorrectly "
                                        "classified as FAKE."
                                    )
                                    st.write(
                                        "- 750 FAKE test images were incorrectly "
                                        "classified as REAL."
                                    )

                                    st.warning(
                                        "Important limitation: this CNN was trained "
                                        "on CIFAKE, which contains 32 x 32 CIFAR-10 "
                                        "real images and synthetic images. Real-world "
                                        "incident photographs can differ substantially "
                                        "from this dataset. The 0.5 threshold is only "
                                        "the model's experimental class boundary."
                                    )

                                    st.info(
                                        "Use this output only as an additional "
                                        "screening signal for human reviewers. "
                                        "Never automatically reject, penalise, or "
                                        "dismiss an incident report based on it."
                                    )

                            else:
                                st.error(
                                    "The model returned a value outside the expected "
                                    "0-1 range."
                                )

                    except ImportError:
                        st.warning(
                            "A model file exists, but TensorFlow is not installed. "
                            "Install the project's optional TensorFlow dependency "
                            "to enable inference."
                        )
                    except Exception as model_error:
                        st.error(
                            f"The Deep Learning model could not analyse this image: "
                            f"{model_error}"
                        )
                else:
                    st.info(
                        "Deep Learning inference is unavailable because the trained "
                        "model file was not found at "
                        "`models/image_authenticity_model.keras`. "
                        "The model must accept 32 x 32 RGB images and return one "
                        "REAL-class score. The metadata and duplicate checks remain "
                        "available without the model."
                    )

                st.subheader("5. Report review notes")
                if report_description.strip():
                    st.write("**Submitted description:**", report_description)
                if report_location.strip():
                    st.write("**Reported location:**", report_location)
                if report_datetime.strip():
                    st.write("**Reported date/time:**", report_datetime)

                if exif_data.get("DateTimeOriginal") and report_datetime.strip():
                    st.info(
                        "A capture timestamp is present in the image metadata. A reviewer may "
                        "compare it with the reported time, keeping in mind that EXIF metadata "
                        "can be edited, stripped or set incorrectly."
                    )

                st.divider()
                st.subheader("Final screening outcome")
                st.info(
                    "Preliminary screening completed. Send the report for human verification. "
                    "A photograph alone cannot establish whether an incident occurred, where it "
                    "occurred, or who was involved. Preserve the original evidence and follow "
                    "appropriate privacy and legal procedures."
                )

    else:
        st.caption(
            "For privacy, uploaded photographs are processed in memory and are not stored by "
            "this page. Only an image fingerprint is stored if you explicitly select the consent option."
        )


# ============================================================
# PAGE 8 — ABOUT
# ============================================================

elif page == "About":

    st.title("📚 About HerShield")

    st.markdown(
        """
        ## HerShield — Intelligent Women Safety & Risk Analytics

        HerShield combines historical crime analytics, geospatial
        processing, machine learning and real-world map services.

        ## Historical Crime Data

        The core crime dataset is the official **Crime in India 2023**
        data published through the Government of India's Open Government
        Data platform.

        The NCRB data used by this application is aggregated at the
        state/UT level and is historical rather than live.

        ## Machine Learning

        ### K-Means

        Groups states/UTs according to similarities in selected
        historical crime-rate features.

        ### Isolation Forest

        Identifies statistically unusual combinations of selected
        historical crime-rate features.

        ### Risk Indicator

        A normalized analytical indicator based on selected historical
        crime-rate features.

        ## AI-Based Crime Report Verification

        The Report Verification page provides preliminary image screening
        for photographs submitted with an incident report. It checks available
        EXIF metadata, compares image fingerprints when a user explicitly
        consents to fingerprint storage, and can run an optional trained Keras
        image-forensics model. The uploaded photograph is not stored by this
        page. Deep Learning inference requires a separately trained compatible
        model file. Screening results are indicators for human review, not
        proof that a report is genuine or fake.

        ## Live GPS

        The Live Safety page obtains the browser's current GPS
        coordinates after the user grants location permission.

        The coordinates are matched to a geographic boundary using
        point-in-polygon analysis.

        ## Safe Travel

        Safe Travel uses:

        1. Current browser GPS as the starting point.
        2. OpenStreetMap Nominatim for user-triggered destination geocoding.
        3. OSRM for road-route calculations.
        4. OpenStreetMap / Overpass for nearby mapped transportation.
        5. NCRB 2023 for historical safety context.

        ## Important Limitations

        HerShield does **not** claim to provide:

        - live crime incident detection
        - exact-GPS crime prediction
        - individual danger prediction
        - guaranteed safe routes
        - police emergency dispatch
        - guaranteed real-time transport availability

        Map and transportation information depends on external services
        and the completeness of OpenStreetMap data.

        Historical NCRB statistics should not be interpreted as current
        street-level conditions.
        """
    )

    st.divider()

    st.subheader("📊 Current Dataset Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        create_metric(
            "States / UTs",
            format_number(len(features_df)),
        )

    with c2:
        create_metric(
            "Columns",
            format_number(len(features_df.columns)),
        )

    with c3:
        create_metric(
            "Average Risk Indicator",
            format_decimal(
                features_df["risk_score"].mean(),
                2,
            ),
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HerShield — Intelligent Women Safety & Risk Analytics"
)

st.caption(
    "Historical data: NCRB Crime in India 2023 | "
    "Map data: OpenStreetMap | "
    "Risk indicator and report-image screening are analytical aids, not proof or guarantees."
)