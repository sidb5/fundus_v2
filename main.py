import streamlit as st
from PIL import Image

st.set_page_config(layout="wide",
    page_title="Fundus Disease Detection", page_icon="assets/logo_small.png"
)
# logo_path = "assets/logo.png"  # Path to your logo
# logo = Image.open(logo_path) # Resize the logo
_, col1,_ = st.columns([2.8,3,1])
with col1:
    st.image('assets/logo.png', width=200)
st.markdown("""
    <style>
    .app-spacing {
        margin-top: -60px;
        margin-bottom: -70px;
    }
    </style>
    """, unsafe_allow_html=True)

app_name = """
    <div class='app-spacing' style="padding:4px">
    <h1 style='text-align: center; color: #22686E; font-size: 30px;'>Fundus Disease Detection</h1>
    </div>
"""
st.markdown(app_name, unsafe_allow_html=True)

# Hiding the sidebar using custom CSS
hide_sidebar_style = """
    <style>
    /* Hide the Streamlit sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)


# --- PAGE SETUP ---
login = st.Page(page="pages/login.py", title=" ", default=True)
app = st.Page(page="pages/app.py", title=" ")


# --- NAVIGATION SETUP ---
pg = st.navigation(pages=[login, app])

pg.run()
