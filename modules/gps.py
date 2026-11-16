import streamlit as st
from streamlit_geolocation import streamlit_geolocation


def get_current_location():
    """
    Request the user's browser GPS location.

    Returns:
        dict with latitude, longitude, accuracy, or None
    """

    location = streamlit_geolocation()

    if not location:
        return None

    latitude = location.get("latitude")
    longitude = location.get("longitude")
    accuracy = location.get("accuracy")

    if latitude is None or longitude is None:
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "accuracy": accuracy,
    }


def display_gps():
    """
    Display the GPS interface in Streamlit.
    """

    st.subheader("📍 Live GPS Location")

    st.write(
        "Allow browser location access to display your current position."
    )

    location = get_current_location()

    if location is None:
        st.info(
            "Waiting for GPS location. "
            "Click the location button and allow location access in your browser."
        )
        return None

    latitude = location["latitude"]
    longitude = location["longitude"]
    accuracy = location["accuracy"]

    st.success("GPS location detected.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Latitude", f"{latitude:.6f}")

    with col2:
        st.metric("Longitude", f"{longitude:.6f}")

    with col3:
        if accuracy is not None:
            st.metric("Accuracy", f"{accuracy:.1f} m")
        else:
            st.metric("Accuracy", "Unavailable")

    return location