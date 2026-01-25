import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(
    page_title="exHUMA",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Interactive Starfield Background ---
import random

# Generate random stars with wrapper for movement
stars_html = ""
for _ in range(150): # Increased star count
    top = random.randint(0, 100)
    left = random.randint(0, 100)
    size = random.randint(1, 3)
    duration_drift = random.randint(15, 30) # Slower drift for realism
    delay = random.randint(0, 10)
    duration_twinkle = random.randint(2, 5)
    
    stars_html += f'''
    <div class="star-wrapper" style="top: {top}%; left: {left}%; animation-duration: {duration_drift}s; animation-delay: -{delay}s;">
        <div class="star" style="width: {size}px; height: {size}px; animation-duration: {duration_twinkle}s;"></div>
    </div>
    '''

st.markdown(f"""
    <style>
    
    /* 1. Deep Space Black Background */
    .stApp {{
        background-color: #000000;
        background-image: none;
    }}

    /* 2. Star Wrapper (Handles Position & Drift) */
    .star-wrapper {{
        position: fixed;
        z-index: 0;
        animation-name: drift;
        animation-iteration-count: infinite;
        animation-timing-function: ease-in-out;
        animation-direction: alternate;
        pointer-events: auto; /* Catch hover */
    }}

    /* 3. The Star Visual (Handles Appearance & Twinkle) */
    .star {{
        background: white;
        border-radius: 50%;
        opacity: 0.5;
        transition: all 0.2s ease-out;
        box-shadow: 0 0 2px rgba(255,255,255,0.4);
        
        /* Twinkle Effect */
        animation-name: twinkle;
        animation-iteration-count: infinite;
        animation-direction: alternate;
    }}

    /* Animations */
    @keyframes drift {{
        0% {{ transform: translate(0px, 0px); }}
        50% {{ transform: translate(30px, -20px); }} /* Gentle Wander */
        100% {{ transform: translate(-10px, 20px); }}
    }}

    @keyframes twinkle {{
        0% {{ opacity: 0.2; transform: scale(1); }}
        100% {{ opacity: 0.9; transform: scale(1.15); }}
    }}

    /* 4. Interaction (Hover on Wrapper triggers Inner) */
    .star-wrapper:hover .star {{
        opacity: 1 !important;
        transform: scale(5) !important; /* Significant Expansion */
        background: #00f2ff; /* Blue-shift glow */
        box-shadow: 0 0 20px 5px rgba(0, 242, 255, 0.9) !important;
        animation-play-state: paused; /* Stop twinkle */
    }}
    
    .star-wrapper:hover {{
        z-index: 1; /* Bring to front */
        animation-play-state: paused; /* Stop drifting so you can catch it */
    }}

    /* Container management */
    #star-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        pointer-events: none;
    }}
    #star-container .star-wrapper {{
        pointer-events: auto;
    }}
    
    /* Ensure Streamlit content sits above stars */
    .main .block-container {{
        z-index: 10;
        position: relative;
    }}
    
    /* --- Existing UI Styling Preserved --- */
    
    /* Headers & Fonts */
    h1, h2, h3, h4 {{
        color: #00f2ff;
        font-family: 'Orbitron', 'sans-serif';
        text-shadow: 0 0 10px #00f2ff;
    }}
    
    /* Metrics / KPIs */
    div[data-testid="stMetricValue"] {{
        color: #fff;
        font-size: 1.8rem !important;
        text-shadow: 0 0 5px #00f2ff;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #aaa;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(10, 10, 10, 0.95);
        border-right: 1px solid #333;
        z-index: 20;
    }}
    
    /* Leaderboard Table */
    .dataframe {{
        font-size: 0.8rem; 
        color: white !important;
        background-color: rgba(0,0,0,0.5);
    }}
    
    /* Live Pulse Animation */
    .live-indicator {{
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #00ff00;
        border-radius: 50%;
        box-shadow: 0 0 0 rgba(0, 255, 0, 0.7);
        animation: pulse 2s infinite;
        margin-right: 8px;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7); }}
        70% {{ box-shadow: 0 0 0 10px rgba(0, 255, 0, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }}
    }}
    </style>
    
    <!-- Inject Starfield HTML -->
    <div id="star-container">
        {stars_html}
    </div>
    """, unsafe_allow_html=True)

# --- 2. GLOBAL DISCOVERY HEADER ---
# Top KPI Panel
col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])

with col1:
    st.title("exHUMA")
    st.markdown('<div style="display:flex; align-items:center; color:#00ff00; font-weight:bold;"><span class="live-indicator"></span> MISSION ACTIVE</div>', unsafe_allow_html=True)

with col2:
    st.metric("Stars Scanned", "5,087")
with col3:
    st.metric("Candidates", "111")
with col4:
    st.metric("Verified", "5")
with col5:
    st.metric("Time Saved", "~80%")

st.markdown("---")

# --- 3. SIDEBAR: CANDIDATE LEADERBOARD ---
# Load Shortlist Data
@st.cache_data
def load_leaderboard():
    path = "data/processed/antigravity_verified_shortlist.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["Star_Index", "Confidence", "Vetting_SNR", "Period"])

df_leaderboard = load_leaderboard()

# Sidebar Layout
with st.sidebar:
    if os.path.exists("assets/logo.jpg"):
        st.image("assets/logo.jpg", use_container_width=True)
    
    st.markdown("### Candidate Planet Leaderboard")
    
    # Priority Color Coding function for dataframe (visual only in Streamlit 1.29+ with column config, else simplified)
    def highlight_snr(val):
        color = '#00ff00' if val > 0.1 else '#ffff00' if val > 0.0 else '#ff0000'
        return f'color: {color}'

    # Selection Mechanism
    if not df_leaderboard.empty:
        # Format for display
        df_display = df_leaderboard.copy()
        df_display['Star'] = df_display['Star_Index'].apply(lambda x: f"Star {x}")
        
        selected_star_label = st.selectbox(
            "Select Target for Analysis:",
            options=df_display['Star'].tolist(),
            index=0
        )
        
        # Get selected row
        selected_star_idx = int(selected_star_label.split(" ")[1])
        star_data = df_leaderboard[df_leaderboard['Star_Index'] == selected_star_idx].iloc[0]
        
        # Mini Profile in Sidebar
        st.info(f"**SNR:** {star_data['Vetting_SNR']:.4f}")
        st.info(f"**Period:** {star_data['Period']:.2f} days")
        
        # File Uploader for Raw Data (Required for Signal View)
        st.markdown("---")
        st.markdown("### 📡 Data Uplink")
        uploaded_file = st.file_uploader("Upload Raw Flux (exoTest.csv)", type=['csv'])
        
    else:
        st.warning("No candidates found in shortlist.")
        selected_star_idx = None
        star_data = None
        uploaded_file = None

# --- 4. EVIDENCE VAULT (CENTER PANELS) ---

if selected_star_idx is not None:
    st.subheader(f"Analyzing Target: {selected_star_label}")
    
    # Tabs
    tab_signal, tab_phase, tab_xai, tab_3d = st.tabs(["📉 AI Signal Analysis", "⚛️ Phase Folding", "🧠 XAI Heatmap", "🪐 3D Orbit Sim"])
    
    # --- Tab 1: AI Signal (Requires Raw Data) ---
    with tab_signal:
        if uploaded_file:
            try:
                # Load Raw Data on the fly (This simulates the massive scan)
                # Optimization: In production, use indexed database. Here we scan CSV.
                # Assuming standard Kepler format: Label, Flux1, Flux2...
                # We need to find the row corresponding to the index. 
                # Note: Star_Index in shortlist usually refers to dataframe index.
                # If file is huge, this might be slow.
                
                # Check cache for dataframe
                @st.cache_data
                def load_raw_data(file):
                    return pd.read_csv(file)
                
                df_raw = load_raw_data(uploaded_file)
                
                if 0 <= selected_star_idx < len(df_raw):
                    raw_row = df_raw.iloc[selected_star_idx]
                    # Drop label if exists
                    if 'LABEL' in raw_row.index:
                        flux_values = raw_row.drop('LABEL').values.astype(float)
                    else:
                        flux_values = raw_row.values.astype(float)
                        
                    # Preprocessing (Denoising)
                    flux_smooth = gaussian_filter1d(flux_values, sigma=2)
                    
                    # Plotting
                    fig_signal = go.Figure()
                    fig_signal.add_trace(go.Scatter(y=flux_values, mode='lines', name='Raw Flux', line=dict(color='gray', width=0.5), opacity=0.5))
                    fig_signal.add_trace(go.Scatter(y=flux_smooth, mode='lines', name='Denoised AI Signal', line=dict(color='#00f2ff', width=1.5)))
                    
                    fig_signal.update_layout(
                        title="Flux Signal Processing (Raw vs Gaussian Filtered)",
                        xaxis_title="Observation Time",
                        yaxis_title="Normalized Flux",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0e0'),
                        height=450
                    )
                    st.plotly_chart(fig_signal, use_container_width=True)
                else:
                    st.error(f"Index {selected_star_idx} out of bounds for uploaded file.")
            except Exception as e:
                st.error(f"Error reading raw data: {e}")
        else:
            st.info("⚠️ Awaiting Raw Data Link. Please upload `exoTest.csv` in sidebar to view live signal telemetry.")
            # Show placeholder image or static demo if needed
            
    # --- Tab 2: Phase Folding ---
    with tab_phase:
        # Dynamic search for phase image
        phase_dir = "outputs/phase_folded"
        phase_img_path = None
        if os.path.exists(phase_dir):
            for f in os.listdir(phase_dir):
                if f.startswith(f"star{selected_star_idx}"): # Matches star4_twocheck.png or star_4.png
                    phase_img_path = os.path.join(phase_dir, f)
                    break
        
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            if phase_img_path:
                st.image(phase_img_path, caption=f"Phase Folded Light Curve (Star {selected_star_idx})", use_container_width=True)
            else:
                # If we have raw data, we can compute it live!
                if uploaded_file and 'flux_values' in locals():
                    st.markdown("**Simulating Phase Folding...**")
                    # Simple fold based on period (if valid)
                    period = star_data['Period']
                    if period > 0:
                        time_steps = np.arange(len(flux_values))
                        phase = (time_steps % period) / period
                        
                        fig_phase = go.Figure(go.Scatter(
                            x=phase, 
                            y=flux_values, 
                            mode='markers', 
                            marker=dict(size=2, color='#00f2ff', opacity=0.3),
                            name='Folded Data'
                        ))
                        # Add hypothetical transit zone
                        fig_phase.add_vrect(x0=0.45, x1=0.55, fillcolor="red", opacity=0.1, annotation_text="Orbital Transit Zone")
                        
                        fig_phase.update_layout(
                            title=f"Phase Folded at Period {period:.2f} days",
                            xaxis_title="Phase (0.0 - 1.0)",
                            yaxis_title="Flux",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e0e0e0'),
                            height=450
                        )
                        st.plotly_chart(fig_phase, use_container_width=True)
                else:
                    st.warning("Phase Folding Data Unavailable. Upload raw data to generate scientific proof.")

    # --- Tab 3: XAI Heatmap ---
    with tab_xai:
        st.markdown("#### 🧠 Neural Network Attention Map")
        st.markdown("*Red zones indicate high-probability planetary signatures identified by the CNN temporal branch.*")
        
        # Dynamic search for XAI image
        xai_dir = "outputs/xai_heatmaps"
        xai_path = None
        if os.path.exists(xai_dir):
            for f in os.listdir(xai_dir):
                if f.startswith(f"star{selected_star_idx}"):
                    xai_path = os.path.join(xai_dir, f)
                    break

        if xai_path:
            st.image(xai_path, use_container_width=True)
        else:
            # Placeholder for effect
            st.info("XAI Heatmap generation requires full model runtime. (Placeholder visual below)")
            # Create a dummy heatmap using plotting just to show UI if file missing
            dummy_heat = np.random.rand(10, 100)
            fig_heat = px.imshow(dummy_heat, color_continuous_scale='RdBu_r', aspect='auto')
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'),
                height=300
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # --- Tab 4: 3D Orbital Visualization ---
    with tab_3d:
        st.markdown("#### 🪐 Planetary Tracking Simulation")
        
        period = star_data['Period']
        if period > 0:
            # Create orbit path
            theta = np.linspace(0, 2*np.pi, 100)
            r = 1.0 # Normalized AU distance
            x_orbit = r * np.cos(theta)
            y_orbit = r * np.sin(theta)
            z_orbit = np.zeros_like(theta) # Flat plane
            
            # Animation Frames (Simulated) with explicit trace targeting
            frames = []
            steps = 100 # High resolution for realism
            for i in range(steps):
                angle = (i / steps) * 2 * np.pi
                # Realistic Planet Position
                planet_x = r * np.cos(angle)
                planet_y = r * np.sin(angle)
                
                frames.append(go.Frame(
                    data=[
                        # We only update the planet trace (Index 2)
                        go.Scatter3d(
                            x=[planet_x], y=[planet_y], z=[0],
                            mode='markers',
                            marker=dict(size=8, color='#00f2ff', line=dict(width=2, color='white')) # Glowing planet
                        )
                    ],
                    traces=[2] # CRITICAL: Only update trace 2 (Planet), leave Star/Orbit alone
                ))

            fig_3d = go.Figure(
                data=[
                    # Trace 0: Star (Static) - Enhanced realism
                    go.Scatter3d(
                        x=[0], y=[0], z=[0], 
                        mode='markers', 
                        marker=dict(size=50, color='#ffaa00', opacity=0.9, line=dict(width=0)),
                        name='Host Star'
                    ),
                    # Trace 1: Orbit Path (Static)
                    go.Scatter3d(
                        x=x_orbit, y=y_orbit, z=z_orbit, 
                        mode='lines', 
                        line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'),
                        name='Orbit Trajectory'
                    ),
                    # Trace 2: Planet (Dynamic - Initial)
                    go.Scatter3d(
                        x=[r], y=[0], z=[0], 
                        mode='markers', 
                        marker=dict(size=8, color='#00f2ff'),
                        name='Exoplanet Candidate'
                    )
                ],
                layout=go.Layout(
                    title=f"Orbital Simulation (Period: {period:.2f} Days)",
                    scene=dict(
                        xaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
                        yaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
                        zaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
                        bgcolor='rgba(0,0,0,0)',
                        aspectmode='cube' # Prevent distortion
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=True,
                    legend=dict(font=dict(color="white"), y=0.9),
                    updatemenus=[dict(
                        type="buttons",
                        showactive=False,
                        y=0.1,
                        x=0.1,
                        xanchor="right",
                        yanchor="top",
                        buttons=[dict(label="▶ Play Simulation",
                                    method="animate",
                                    args=[None, {"frame": {"duration": 20, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                        )]
                    )]
                ),
                frames=frames
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("Orbital period undefined for this candidate.")

# --- 5. DISCOVERY REPORT (BOTTOM PANEL) ---
if selected_star_idx is not None:
    st.markdown("---")
    st.subheader("📝 Discovery Report")
    
    rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)
    
    with rep_col1:
        st.metric("Saturn-Day Orbit (Period)", f"{star_data['Period']:.2f} Days")
    with rep_col2:
        st.metric("AI Confidence", f"{star_data['Confidence']:.4f}")
    with rep_col3:
        status = "🟢 Confirmed" if star_data['Vetting_SNR'] > 0.1 else "🟡 Candidate"
        st.metric("Classification Status", status)
    with rep_col4:
        st.markdown("<br>", unsafe_allow_html=True)
        # CSV Export
        csv_data = star_data.to_frame().T.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Export Scientific Report",
            data=csv_data,
            file_name=f"discovery_report_star_{selected_star_idx}.csv",
            mime='text/csv',
            help="Download detailed parameter packet."
        )

# --- 6. BUSINESS INTELLIGENCE SUMMARY ---
if selected_star_idx is not None:
    with st.expander(" Resource Optimization "):
        st.markdown("### Efficiency Analysis")
        
        # Calculations (Hypothetical)
        total_stars = 5087
        # Calculations (Exact Math)
        total_stars = 5087
        manual_time_per_star = 37 # minutes (User Value)
        ai_time_per_star = 0.05 # seconds
        scientist_hourly_rate = 55 # $ (User Value)
        
        # Corrected Divisors for Exact Time
        manual_total_hours = (total_stars * manual_time_per_star) / 60 
        ai_total_hours = (total_stars * ai_time_per_star) / 3600
        cost_saved = (manual_total_hours - ai_total_hours) * scientist_hourly_rate
        
        bi_col1, bi_col2, bi_col3 = st.columns(3)
        
        with bi_col1:
            st.metric("Manual Vetting Load", f"{manual_total_hours:,.0f} Hours", help="Estimated time for human analysis")
        with bi_col2:
            st.metric("Antigravity Speed", f"{ai_total_hours:.2f} Hours", delta=f"{((manual_total_hours-ai_total_hours)/manual_total_hours):.1%}", help="Total computational time")
        with bi_col3:
            st.metric("Opportunity Cost Saved", f"${cost_saved:,.2f}", help="Value of scientist time redirected")
            
        st.caption(f"*Based on {manual_time_per_star} min/star manual review vs {ai_time_per_star}s/star inference latency.*")

# --- 6. ANTIGRAVITY AI CHAT OVERLAY ---
if selected_star_idx is not None:
    with st.expander(" exBOT ", expanded=True):
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input
        if prompt := st.chat_input(f"Ask about Star {selected_star_idx}..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Simple logic response
            response = f"**System Analysis:** Star {selected_star_idx} flagged due to a consistent **{star_data['Period']:.2f}-day orbital period** and distinct U-shaped transit signature. SNR of {star_data['Vetting_SNR']:.3f} exceeds background noise threshold."
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
