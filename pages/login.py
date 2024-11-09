import streamlit as st
from pages.functions import add_user_to_firestore, fetch_user_from_firestore, hash_password, check_password

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


button_style = """
    <style>
    .stButton > button {
        background-color: #4CAF50;  /* Green background */
        color: white;  /* White text */
        border: none;  /* Remove border */
        padding: 12px 24px;  /* Padding for the button */
        text-align: center;  /* Centered text */
        text-decoration: none;  /* Remove underline */
        display: inline-block;  /* Inline block */
        font-size: 16px;  /* Text size */
        margin: 4px 2px;  /* Space between buttons */
        transition-duration: 0.4s;  /* Animation */
        cursor: pointer;  /* Pointer cursor on hover */
        border-radius: 8px; /* Rounded corners */
    }

    .stButton > button:hover {
        background-color: white; 
        color: black; 
        border: 2px solid #4CAF50; /* Green border on hover */
    }
    </style>
"""

st.markdown(button_style, unsafe_allow_html=True)

def login_ui():

    _,col1,_ = st.columns([0.9,2,1])

    with col1:

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary", use_container_width=True):
            if username and password:
                user_data = fetch_user_from_firestore(username)
                if user_data:
                    # Check if the password matches the hashed password stored in Firestore
                    if check_password(user_data['password'], password):
                        st.success("Login successful!")
                        st.session_state.user = username  # Store the username in session state
                        st.session_state.user_data = user_data  # Store user data in session state
                        return True  # Indicate that login was successful
                    else:
                        st.error("Incorrect password. Please try again.")
                else:
                    # If the user doesn't exist, create a new user entry
                    new_user_data = {
                        "username": username,
                        "password": hash_password(password),  # Hash the password before storing
                    }
                    add_user_to_firestore(username, new_user_data)
                    st.success("New user created and logged in!")
                    st.session_state.user = username  # Store the username in session state
                    return True  # Indicate that login was successful
            else:
                st.error("Please enter both username and password")
        return False  # Indicate that login was not successful

if login_ui():
    st.switch_page("pages/app.py")  # Switch to the app page if login is successful

_,col1,_ = st.columns([0.9,2,1])

with col1:

    if st.button("Continue without login", type="secondary", use_container_width=True):
        st.session_state.user = "guest"  # Set session state for guest user
        st.switch_page("pages/app.py")