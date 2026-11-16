import json
import os

import pandas as pd
from shapely.geometry import Point, shape


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RISK_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hershield_risk_features_2023.csv",
)

GEOJSON_FILE = os.path.join(
    BASE_DIR,
    "data",
    "maps",
    "india_state_current.geojson",
)


# ============================================================
# LOAD NCRB RISK DATA
# ============================================================

def load_risk_data():
    """
    Load the existing NCRB 2023 state/UT risk indicators.

    IMPORTANT:
    These are historical state/UT-level indicators.
    They are NOT live crime measurements.
    """

    if not os.path.exists(RISK_FILE):
        return pd.DataFrame()

    try:
        data = pd.read_csv(RISK_FILE)
    except Exception:
        return pd.DataFrame()

    if "state" in data.columns:
        data["state"] = (
            data["state"]
            .astype(str)
            .str.strip()
        )

    return data


# ============================================================
# LOAD INDIA STATE BOUNDARIES
# ============================================================

def load_state_boundaries():
    """
    Load the India state/UT GeoJSON file.

    Returns:
        List of dictionaries containing:
        - state name
        - geometry
    """

    if not os.path.exists(GEOJSON_FILE):
        return []

    try:
        with open(
            GEOJSON_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            geojson = json.load(file)

    except Exception:
        return []

    features = geojson.get(
        "features",
        [],
    )

    boundaries = []

    for feature in features:

        properties = feature.get(
            "properties",
            {},
        )

        geometry_data = feature.get(
            "geometry"
        )

        if not geometry_data:
            continue

        # Different GeoJSON sources may use different
        # property names. NAME_1 is used by this file.
        state_name = (
            properties.get("NAME_1")
            or properties.get("name")
            or properties.get("NAME")
        )

        if not state_name:
            continue

        try:
            geometry = shape(
                geometry_data
            )
        except Exception:
            continue

        boundaries.append(
            {
                "state": str(
                    state_name
                ).strip(),
                "geometry": geometry,
            }
        )

    return boundaries


# ============================================================
# NORMALIZE STATE NAMES
# ============================================================

def normalize_state_name(
    state_name,
):
    """
    Normalize common state/UT naming differences
    between the GeoJSON and NCRB dataset.
    """

    if state_name is None:
        return None

    name = str(
        state_name
    ).strip()

    replacements = {
        "Andaman and Nicobar": (
            "Andaman and Nicobar Islands"
        ),
        "Andaman & Nicobar": (
            "Andaman and Nicobar Islands"
        ),
        "NCT of Delhi": "Delhi",
        "Delhi": "Delhi",
        "Pondicherry": "Puducherry",
    }

    return replacements.get(
        name,
        name,
    )


# ============================================================
# FIND STATE FROM GPS
# ============================================================

def find_state_from_gps(
    latitude,
    longitude,
):
    """
    Determine the state/UT containing the GPS coordinate
    using actual GeoJSON administrative boundaries.

    This replaces the previous nearest-state-center
    approximation.
    """

    boundaries = load_state_boundaries()

    if not boundaries:
        return None

    point = Point(
        float(longitude),
        float(latitude),
    )

    # First try strict containment.
    for boundary in boundaries:

        geometry = boundary["geometry"]

        try:
            if geometry.contains(point):

                return normalize_state_name(
                    boundary["state"]
                )

        except Exception:
            continue

    # Boundary points can sometimes fall exactly on
    # the polygon edge. Covers() handles that case.
    for boundary in boundaries:

        geometry = boundary["geometry"]

        try:
            if geometry.covers(point):

                return normalize_state_name(
                    boundary["state"]
                )

        except Exception:
            continue

    return None


# ============================================================
# FIND NCRB RISK RECORD
# ============================================================

def get_state_risk(
    state_name,
):
    """
    Retrieve the historical NCRB risk indicator
    for the identified state/UT.
    """

    data = load_risk_data()

    if data.empty:
        return None

    if "state" not in data.columns:
        return None

    target = normalize_state_name(
        state_name
    )

    normalized_states = (
        data["state"]
        .astype(str)
        .str.strip()
        .apply(normalize_state_name)
        .str.lower()
    )

    matches = data[
        normalized_states
        == str(target).lower()
    ]

    if matches.empty:
        return None

    record = matches.iloc[0]

    result = {
        "state": target,
    }

    # --------------------------------------------------------
    # Historical risk indicator
    # --------------------------------------------------------

    if "risk_indicator" in data.columns:

        try:
            result["risk_score"] = float(
                record["risk_indicator"]
            )

        except Exception:
            result["risk_score"] = None

    else:
        result["risk_score"] = None

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    if "risk_category" in data.columns:

        category = record[
            "risk_category"
        ]

        if pd.notna(category):

            result["risk_category"] = str(
                category
            ).strip()

        else:

            result["risk_category"] = (
                "Unavailable"
            )

    else:

        score = result["risk_score"]

        if score is None:

            result["risk_category"] = (
                "Unavailable"
            )

        elif score < 33.33:

            result["risk_category"] = (
                "Lower Indicator"
            )

        elif score < 66.67:

            result["risk_category"] = (
                "Moderate Indicator"
            )

        else:

            result["risk_category"] = (
                "Higher Indicator"
            )

    return result


# ============================================================
# COMPLETE GPS RISK ANALYSIS
# ============================================================

def analyze_gps_location(
    latitude,
    longitude,
):
    """
    Analyze a live GPS coordinate.

    Current implementation:

        GPS
          ↓
        Actual state/UT boundary
          ↓
        Historical NCRB 2023 indicator

    IMPORTANT:
    The NCRB dataset is historical and state/UT-level.
    Therefore this function does NOT claim to provide
    live street-level crime prediction.

    The risk score currently represents the historical
    NCRB-derived state/UT indicator associated with the
    user's current geographic location.
    """

    # ========================================================
    # VALIDATE COORDINATES
    # ========================================================

    try:

        latitude = float(
            latitude
        )

        longitude = float(
            longitude
        )

    except Exception:

        return {
            "success": False,
            "message": (
                "Invalid GPS coordinates."
            ),
        }

    # ========================================================
    # VALIDATE RANGE
    # ========================================================

    if not (
        -90 <= latitude <= 90
    ):

        return {
            "success": False,
            "message": (
                "Latitude must be between "
                "-90 and 90."
            ),
        }

    if not (
        -180 <= longitude <= 180
    ):

        return {
            "success": False,
            "message": (
                "Longitude must be between "
                "-180 and 180."
            ),
        }

    # ========================================================
    # FIND ACTUAL STATE BOUNDARY
    # ========================================================

    state = find_state_from_gps(
        latitude,
        longitude,
    )

    if state is None:

        return {
            "success": False,
            "latitude": latitude,
            "longitude": longitude,
            "message": (
                "The GPS coordinate could not "
                "be matched to an India state "
                "or union territory boundary."
            ),
        }

    # ========================================================
    # GET HISTORICAL NCRB RISK
    # ========================================================

    risk = get_state_risk(
        state
    )

    # ========================================================
    # BASE RESULT
    # ========================================================

    result = {
        "success": True,
        "latitude": latitude,
        "longitude": longitude,
        "state": state,
        "geographic_method": (
            "GeoJSON point-in-polygon"
        ),
        "risk_data_source": (
            "NCRB Crime in India 2023"
        ),
    }

    # ========================================================
    # ADD RISK INFORMATION
    # ========================================================

    if risk is not None:

        result.update(
            risk
        )

    else:

        result["risk_score"] = None

        result["risk_category"] = (
            "Unavailable"
        )

    return result