import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

category_map = {'Education': 1, 'Music': 2, 'Tech': 3, 'Entertainment': 4, 'Gaming': 5, 'Lifestyle': 6}
device_map = {'TV': 1, 'Mobile': 2, 'Desktop': 3, 'Tablet': 4}
country_map = {'CA': 1, 'DE': 2, 'IN': 3, 'AU': 4, 'UK': 5, 'US': 6}

st.set_page_config(
    page_title="YouTube Ad Revenue Dashboard",
    page_icon="🎬",
    layout="wide"
)


st.markdown("""
<style>
/* Background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f5f7fa, #c3e0ff);
}

/* Sidebar style */
[data-testid="stSidebar"] {
    background-color: #e6f0ff;
}

/* Card style */
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    transition: transform 0.2s;
    text-align: center;
}
.card:hover {
    transform: translateY(-5px);
}

/* Buttons style */
div.stButton > button:first-child {
    background-color: #0052cc;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #0066ff;
}

/* Headings */
h1, h2, h3, h4 {
    color: #003366;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align:center;'>🎬 YouTube Ad Revenue Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#333;'>Predict ad revenue based on video performance and engagement.</p>", unsafe_allow_html=True)
st.divider()


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    views = st.number_input("👁️ Views", min_value=0, max_value=2_000_000, value=100000)
    likes = st.number_input("❤️ Likes", min_value=0, max_value=1_000_000, value=50000)
    comments = st.number_input("💬 Comments", min_value=0, max_value=100_000, value=20000)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    watch_time_minutes = st.number_input("⏱️ Watch Time (minutes)", min_value=0.0, max_value=10000.0, value=500.0)
    video_length_minutes = st.number_input("🎞️ Video Length (minutes)", min_value=0.0, max_value=500.0, value=100.0)
    subscribers = st.number_input("👥 Subscribers", min_value=0, max_value=2_000_000, value=100000)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    category = st.selectbox("📚 Category", options=list(category_map.keys()))
    device = st.selectbox("📱 Device", options=list(device_map.keys()))
    country = st.selectbox("🌍 Country", options=list(country_map.keys()))
    st.markdown("</div>", unsafe_allow_html=True)


engagement_rate = (likes + comments) / views if views > 0 else 0
col_a, col_b, col_c = st.columns(3)
col_a.metric("Engagement Rate", f"{engagement_rate:.4f}")
col_b.metric("Views", f"{views:,}")
col_c.metric("Subscribers", f"{subscribers:,}")

st.divider()


st.markdown("<div class='card'>", unsafe_allow_html=True)
if st.button("💡 Predict Revenue"):
    input_data = np.array([[views, likes, comments, watch_time_minutes,
                            video_length_minutes, subscribers, engagement_rate,
                            category_map[category], device_map[device], country_map[country]]])
    
    # Scale numeric features
    num_cols = [0, 1, 2, 3, 4, 5, 6]
    input_data[:, num_cols] = scaler.transform(input_data[:, num_cols])
    
    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Ad Revenue: **${prediction:,.2f} USD**")
    st.balloons()
st.markdown("</div>", unsafe_allow_html=True)


st.divider()
st.markdown("<p style='text-align:center; color:#555;'>Developed with ❤️ using Streamlit | © 2025 YouTube Revenue Predictor</p>", unsafe_allow_html=True)
