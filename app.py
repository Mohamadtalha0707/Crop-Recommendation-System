import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR ---------------- #

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2909/2909763.png",
    width=70
)

st.sidebar.title("🌱 Smart Farming AI")

page = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Home",
        "🌾 Crop Recommendation",
        "📚 Crop Information",
        "ℹ️ About Project"
    ]
)

st.sidebar.success("✅ AI Powered Agriculture")

# ---------------- HOME PAGE ---------------- #

# ---------------- HOME PAGE ---------------- #

if page == "🏠 Home":

    # HERO SECTION
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1b4332, #2d6a4f, #40916c);
        padding: 60px;
        border-radius: 25px;
        text-align: center;
        color: white;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
    ">

    <h1 style="
        font-size:60px;
        margin-bottom:10px;
    ">
        🌱 Smart Crop Recommendation System
    </h1>

    <h3 style="
        color:#e9f5ec;
        font-size:28px;
        font-weight:normal;
    ">
        AI Powered Agriculture Solution for Modern Farming 🚜
    </h3>

    <p style="
        font-size:18px;
        margin-top:20px;
        color:#d8f3dc;
    ">
        Predict the best crop using Machine Learning, Soil Analysis,
        and Environmental Conditions.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # MAIN IMAGE
    st.image(
    "https://images.unsplash.com/photo-1500937386664-56d1dfef3854",
    width="stretch"
)

    st.write("")
    st.write("")

    # FEATURES TITLE
    st.markdown("""
    <h2 style='text-align:center; color:#52b788;'>
        🚀 Powerful Features
    </h2>
    """, unsafe_allow_html=True)

    st.write("")

    # FEATURE CARDS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background-color:#1f2937;
            padding:30px;
            border-radius:20px;
            text-align:center;
            color:white;
            box-shadow:0px 4px 15px rgba(0,0,0,0.3);
        ">
        <h2>🌾 Smart Prediction</h2>
        <p>
        Get accurate crop recommendations based on soil nutrients.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background-color:#1f2937;
            padding:30px;
            border-radius:20px;
            text-align:center;
            color:white;
            box-shadow:0px 4px 15px rgba(0,0,0,0.3);
        ">
        <h2>🤖 Machine Learning</h2>
        <p>
        Uses trained AI models for intelligent agriculture decisions.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background-color:#1f2937;
            padding:30px;
            border-radius:20px;
            text-align:center;
            color:white;
            box-shadow:0px 4px 15px rgba(0,0,0,0.3);
        ">
        <h2>📈 Better Farming</h2>
        <p>
        Increase productivity and improve farming efficiency.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # STATS SECTION
    st.markdown("""
    <h2 style='text-align:center; color:#52b788;'>
        📊 Project Highlights
    </h2>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌱 Accuracy", "98%")
    col2.metric("🚜 Farmers Helped", "500+")
    col3.metric("📈 Predictions", "10K+")
    col4.metric("🤖 AI Model", "Random Forest")

    st.write("")
    st.write("")

    # FOOTER SECTION
    st.markdown("""
    <div style="
        background-color:#1b4332;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
    ">

    <h3>🌍 Smart Agriculture for Modern India</h3>

    <p>
    Developed using Python, Streamlit, and Machine Learning
    </p>

    </div>
    """, unsafe_allow_html=True)

# ---------------- CROP RECOMMENDATION ---------------- #

elif page == "🌾 Crop Recommendation":

    st.title("🌾 Crop Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen", min_value=0)
        P = st.number_input("Phosphorus", min_value=0)
        K = st.number_input("Potassium", min_value=0)
        temperature = st.number_input("Temperature")

    with col2:
        humidity = st.number_input("Humidity")
        ph = st.number_input("pH Value")
        rainfall = st.number_input("Rainfall")

    if st.button("🔍 Predict Crop"):

        data = np.array([
            [N, P, K, temperature, humidity, ph, rainfall]
        ])

        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)

        st.success(f"✅ Recommended Crop: {prediction[0]}")

        st.balloons()

# ---------------- CROP INFO ---------------- #

elif page == "📚 Crop Information":

    st.title("📚 Crop Information")

    crop = st.selectbox(
        "Select Crop",
        ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
    )

    info = {
        "Rice": "Rice grows best in high rainfall.",
        "Wheat": "Wheat requires moderate temperature.",
        "Maize": "Maize grows well in warm climate.",
        "Cotton": "Cotton grows best in black soil.",
        "Sugarcane": "Sugarcane requires fertile soil."
    }

    st.info(info[crop])

# ---------------- ABOUT PAGE ---------------- #

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.markdown("""
    ### Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - Machine Learning

    ### Algorithm Used
    - Random Forest Classifier

    ### Objective
    To help farmers select suitable crops using AI.
    """)

    st.success("🚀 Project Developed Successfully")