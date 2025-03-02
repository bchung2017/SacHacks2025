import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# Set page configuration
st.set_page_config(page_title="WEB DETECTIVE ", page_icon="🔎", layout="centered")

# Custom CSS to hide the default header and menu bar and add custom font
st.markdown(
    """
    <style>
        @import url('https://db.onlinewebfonts.com/c/18d21d4b60c0f073046832e87b3d9675?family=Angro+LT+W01+Light');
        
        body {
            font-family: 'Angro LT W01 Light', sans-serif;
            background-color: #B0E0E6; /* Powder Blue Background */
        }
        
        /* Hide Streamlit's default menu and header */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Additional styling for a clean look */
        .main-title {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            font-family: 'Angro LT W01 Light' !important;
        }
        .subtitle {
            font-size: 1.2rem;
            text-align: center;
            color: gray;
            font-family: 'Angro LT W01 Light', sans-serif;
        }
        .search-box {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 1rem;
        }
        .search-box:focus {
            outline: 2px solid brown !important; /* Change highlight to brown */
            border-color: brown !important;
            box-shadow: 0 0 5px brown !important;
        }
        .generate-button {
            background-color: #5a5df0;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            font-size: 1rem;
        }
        .generate-button:hover {
            background-color: #4a4dcf;
        }
        .info-box {
            background-color: #f8f9fc;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
        div[data-baseweb="input"] > div {
            border: 2px solid #8B4513 !important; /* Brown border */
            border-radius: 8px !important;
        }

        div[data-baseweb="input"]:focus-within {
            border: 2px solid brown !important;
        }

        button {
            transition: background-color 0.3s ease-in-out !important;
        }

        button:focus, button:hover {
            background-color: brown !important;
            border-color: brown !important;
            box-shadow: none !important; /* Remove any default highlight */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1 class='main-title'>Generate Search Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter your website URL below and we'll generate an AI-powered search engine for your content.</p>", unsafe_allow_html=True)

# Input field for URL
url = st.text_input("Website URL", placeholder="https://example.com")

with stylable_container(
    "brown-button",
    css_styles="""
    button {
        background-color: #5A3D2B; /* Initial brown color */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        transition: background-color 0.3s ease-in-out;
    }
    
    button:hover, button:focus {
        background-color: #8B4513 !important; /* Darker brown on hover/focus */
        box-shadow: none !important; /* Remove any orange glow */
    }
    """,
):
    button_clicked = st.button("Generate Search Engine", key="generate_button")
# Button to generate search engine
if button_clicked:
    if url:
        st.success(f"Generating search engine for {url}...")
    else:
        st.error("Please enter a valid URL.")

# Information box on what happens next
st.markdown(
    """
    <div class='info-box'>
        <h4>What happens next?</h4>
        <p>✅ We'll scan your website and analyze its content</p>
        <p>✅ Our AI will build a custom search index for your content</p>
        <p>✅ You'll get a ready-to-use search interface for your website</p>
    </div>
    """,
    unsafe_allow_html=True
)
