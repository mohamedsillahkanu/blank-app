import streamlit as st
import base64

# Page configuration
st.set_page_config(
    page_title="CHIRPS Rainfall Analysis Tool",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for blue theme
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        color: white;
    }
    
    /* Custom slide container */
    .slide-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 3rem;
        border-radius: 20px;
        margin: 2rem 0;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Title styling */
    .title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
        color: #ffffff;
        text-shadow: 2px 2px 20px rgba(0,0,0,0.5);
    }
    
    .subtitle {
        font-size: 1.8rem;
        text-align: center;
        color: #bbdefb;
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 15px rgba(0,0,0,0.4);
    }
    
    /* Impact cards */
    .impact-card {
        background: rgba(255, 255, 255, 0.15);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .impact-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Highlight text */
    .highlight {
        background: linear-gradient(45deg, #ffffff, #e8f4fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    
    /* Stats box */
    .stats-box {
        background: rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .big-number {
        font-size: 3rem;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    /* Workflow steps */
    .workflow-step {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .step-number {
        background: linear-gradient(135deg, #ffffff, #e3f2fd);
        color: #1565c0;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0 auto 1rem auto;
        box-shadow: 0 8px 20px rgba(255, 255, 255, 0.2);
    }
    
    /* Before/After comparison */
    .before-card {
        background: rgba(244, 67, 54, 0.15);
        border: 2px solid rgba(244, 67, 54, 0.4);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .after-card {
        background: rgba(76, 175, 80, 0.15);
        border: 2px solid rgba(76, 175, 80, 0.4);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    
    /* Quote box */
    .quote-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #ffffff;
        margin: 2rem 0;
        font-style: italic;
        font-size: 1.3rem;
        color: #e3f2fd;
    }
    
    /* Technical grid */
    .tech-card {
        background: rgba(255, 255, 255, 0.15);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 0.5rem;
    }
    
    /* Navigation */
    .nav-button {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        margin: 0.5rem;
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Override Streamlit button styling */
    .stButton > button {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(13, 71, 161, 0.9) !important;
    }
    
    /* Text styling */
    .main-text {
        font-size: 1.2rem;
        line-height: 1.6;
        color: #e3f2fd;
        text-align: center;
    }
    
    /* Hide streamlit elements */
    .css-18e3th9 {
        padding-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for slide navigation
if 'slide' not in st.session_state:
    st.session_state.slide = 0

# Define slides
slides = [
    "title",
    "problem", 
    "solution",
    "impact",
    "technical",
    "applications",
    "revolution",
    "contact"
]

def next_slide():
    if st.session_state.slide < len(slides) - 1:
        st.session_state.slide += 1

def prev_slide():
    if st.session_state.slide > 0:
        st.session_state.slide -= 1

def go_to_slide(slide_num):
    st.session_state.slide = slide_num

# Navigation sidebar
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    slide_names = [
        "🏠 Title",
        "❌ The Problem", 
        "✅ The Solution",
        "🚀 Real Impact",
        "⚙️ Technical Power",
        "🌍 Applications",
        "🔥 Revolution",
        "📞 Contact"
    ]
    
    for i, name in enumerate(slide_names):
        if st.button(name, key=f"nav_{i}"):
            go_to_slide(i)
    
    st.markdown("---")
    st.markdown(f"**Slide {st.session_state.slide + 1} of {len(slides)}**")

# Main content area
current_slide = slides[st.session_state.slide]

if current_slide == "title":
    st.markdown("""
    <div class="slide-container">
        <div class="title">🌧️ CHIRPS Rainfall Analysis Tool</div>
        <div class="subtitle">Democratizing Satellite Climate Data for Global Health</div>
        
        <div class="stats-box">
            <div class="big-number">From PhD-level coding to smartphone simplicity</div>
            <p class="main-text">Making satellite rainfall data accessible to everyone, everywhere</p>
        </div>
        
        <div style="text-align: center; margin-top: 3rem;">
            <h3 style="color: #ffffff; margin-bottom: 0.5rem;">Mohamed Sillah Kanu</h3>
            <p style="color: #bbdefb; font-size: 1.2rem;">Northwestern University Malaria Modeling Team</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif current_slide == "problem":
    st.markdown("""
    <div class="slide-container">
        <div class="section-header">The Barrier to Life-Saving Data</div>
        
        <div class="quote-box">
            "Critical rainfall data for malaria interventions, agriculture, and disaster management was locked behind technical barriers that only PhD-level programmers could overcome."
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="impact-card">
            <h3 style="color: #ef4444; margin-bottom: 1rem;">Traditional Approach</h3>
            <ul style="color: #e3f2fd; font-size: 1.1rem; line-height: 1.8;">
                <li>Required R/Python expertise</li>
                <li>Complex .tif file processing</li>
                <li>Hours of manual coding</li>
                <li>Expensive GIS software</li>
                <li>High-performance computers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="impact-card">
            <h3 style="color: #ef4444; margin-bottom: 1rem;">Real-World Impact</h3>
            <ul style="color: #e3f2fd; font-size: 1.1rem; line-height: 1.8;">
                <li>Health workers couldn't access data</li>
                <li>Delayed malaria interventions</li>
                <li>Limited disaster preparedness</li>
                <li>Inefficient resource allocation</li>
                <li>Lives at risk</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif current_slide == "solution":
    st.markdown("""
    <div class="slide-container">
        <div class="section-header">Revolutionary <span class="highlight">Democratization</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="before-card">
            <h3 style="color: #ef4444;">BEFORE</h3>
            <div style="font-size: 2rem; font-weight: 900; color: #ef4444;">3 days</div>
            <p>Expert programmers only</p>
            <p>Complex installation</p>
            <p>Expensive software</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 4rem; color: #ffffff;">→</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="after-card">
            <h3 style="color: #22c55e;">NOW</h3>
            <div style="font-size: 2rem; font-weight: 900; color: #22c55e;">3 minutes</div>
            <p>Anyone with internet</p>
            <p>Zero installation</p>
            <p>Free and accessible</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Workflow steps
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="workflow-step">
            <div class="step-number">1</div>
            <h4>Open Browser</h4>
            <p>Any device, anywhere</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="workflow-step">
            <div class="step-number">2</div>
            <h4>Select Region</h4>
            <p>Point and click</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="workflow-step">
            <div class="step-number">3</div>
            <h4>Get Data</h4>
            <p>Instant analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="workflow-step">
            <div class="step-number">4</div>
            <h4>Download</h4>
            <p>Ready-to-use CSV</p>
        </div>
        """, unsafe_allow_html=True)

elif current_slide == "impact":
    st.markdown("""
    <div class="slide-container">
        <div class="section-header">Real-World <span class="highlight">Game Changer</span></div>
        
        <div class="quote-box">
            <strong>Scenario:</strong> A health worker in rural Ghana needs rainfall data to plan malaria interventions. Previously impossible without a computer science degree.
        </div>
        
        <div style="background: rgba(255, 255, 255, 0.1); padding: 3rem; border-radius: 20px; text-align: center; margin: 2rem 0; font-size: 1.5rem; color: #bbdefb;">
            📱 Smartphone → 🗺️ Select Ghana → 📊 Instant Rainfall Maps → 📧 Share with Team
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="impact-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🌍</div>
            <h3>Global Access</h3>
            <p class="main-text">Works on smartphones in remote villages with basic internet connection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="impact-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">⚡</div>
            <h3>Instant Impact</h3>
            <p class="main-text">From days of programming to minutes of clicking - immediate actionable data</p>
        </div>
        """, unsafe_allow_html=True)

elif current_slide == "technical":
    st.markdown("""
    <div class="slide-container">
        <div class="section-header">Sophisticated Analysis, <span class="highlight">Simple Interface</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="impact-card">
            <h3>🛰️ Behind the Scenes</h3>
            <ul style="color: #e3f2fd; font-size: 1.1rem; line-height: 1.8;">
                <li><strong>CHIRPS v2.0:</strong> 5km resolution satellite data</li>
                <li><strong>GADM v4.1:</strong> Global administrative boundaries</li>
                <li><strong>Python + GeoPandas:</strong> Complex geospatial processing</li>
                <li><strong>Automated pipeline:</strong> Download → Process → Visualize</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="impact-card">
            <h3>📊 Advanced Visualizations</h3>
            <ul style="color: #e3f2fd; font-size: 1.1rem; line-height: 1.8;">
                <li><strong>Country-level analysis:</strong> Complete rainfall patterns</li>
                <li><strong>Monthly subplots:</strong> Seasonal variation tracking</li>
                <li><strong>Admin unit breakdown:</strong> District-level precision</li>
                <li><strong>Export ready:</strong> High-resolution maps & data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats-box">
        <p style="font-size: 1.4rem; margin: 0;">Complex satellite data processing that previously required a team of data scientists is now available to anyone with a web browser</p>
    </div>
    """, unsafe_allow_html=True)

elif current_slide == "applications":
    st.markdown("""
    <div class="slide-container">
        <div class="section-header">Transforming <span class="highlight">Multiple Sectors</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="impact-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🏥</div>
            <h3>Public Health</h3>
            <p class="main-text">Malaria intervention timing, disease outbreak prediction, resource planning</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="impact-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🚨</div>
            <h3>Disaster Management</h3>
            <p class="main-text">Flood early warning, emergency preparedness, climate risk assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="impact-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🌾</div>
            <h3>Agriculture</h3>
            <p class="main-text">Crop planning, irrigation scheduling, drought monitoring, yield forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="impact-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🔬</div>
            <h3>Research</h3>
            <p class="main-text">Climate studies, policy making, environmental monitoring, academic research</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats-box">
        <div class="big-number">Universal Impact</div>
        <p class="main-text">From field workers to researchers - satellite rainfall data is now accessible to everyone who needs it</p>
    </div>
    """, unsafe_allow_html=True)

elif current_slide == "revolution":
    st.markdown("""
    <div class="slide-container">
        <div class="section-header">This Changes <span class="highlight">Everything</span></div>
        
        <div class="quote-box">
            "We've transformed satellite rainfall analysis from a complex, expert-only process requiring advanced programming skills into a simple, accessible tool that anyone can use on any device, anywhere in the world."
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; font-weight: 900; color: #ef4444;">BEFORE</div>
            <p style="color: #ef4444; font-size: 1.2rem;">Days of coding</p>
            <p style="color: #ef4444; font-size: 1.2rem;">Expert programmers only</p>
            <p style="color: #ef4444; font-size: 1.2rem;">Expensive barriers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 4rem; color: #ffffff;">→</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; font-weight: 900; color: #22c55e;">NOW</div>
            <p style="color: #22c55e; font-size: 1.2rem;">Minutes of clicking</p>
            <p style="color: #22c55e; font-size: 1.2rem;">Anyone, anywhere</p>
            <p style="color: #22c55e; font-size: 1.2rem;">Free and accessible</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats-box">
        <div class="big-number">The Future is Here</div>
        <p class="main-text">Democratizing climate data for global health - one click at a time</p>
    </div>
    """, unsafe_allow_html=True)

elif current_slide == "contact":
    st.markdown("""
    <div class="slide-container">
        <div class="section-header">Empowering Global Health</div>
        <div class="subtitle">Through Accessible Climate Data</div>
        
        <div class="stats-box">
            <div class="big-number">Thank You</div>
            <p style="font-size: 1.3rem; margin: 0;">Questions & Discussion</p>
        </div>
        
        <div style="text-align: center; margin-top: 3rem;">
            <h3 style="color: #ffffff; margin-bottom: 0.5rem;">Mohamed Sillah Kanu</h3>
            <p style="color: #bbdefb; font-size: 1.2rem;">Northwestern University Malaria Modeling Team</p>
            <p style="color: #90caf9; font-size: 1.1rem; margin-top: 2rem;">Making satellite data accessible to save lives</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Navigation buttons
st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.session_state.slide > 0:
        if st.button("← Previous", key="prev"):
            prev_slide()

with col3:
    if st.session_state.slide < len(slides) - 1:
        if st.button("Next →", key="next"):
            next_slide()

with col2:
    st.markdown(f"<div style='text-align: center; color: #bbdefb; font-weight: 600;'>Slide {st.session_state.slide + 1} of {len(slides)}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #90caf9; font-style: italic;'>Built for malaria researchers and public health professionals</div>", unsafe_allow_html=True)
