import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Corporate Decarbonization Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-title">🔍 Corporate Decarbonization Detective</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Unmasking the Reality Behind Corporate Emission Claims</p>', unsafe_allow_html=True)

st.markdown("""
**Decompose observed emission changes into their constituent drivers using the framework from RF3:**

$$\\Delta_{observed} = \\Delta_{genuine} + \\Delta_{methodological} + \\Delta_{organizational} + \\Delta_{growth} + \\varepsilon$$

Where each component reveals different aspects of a company's climate journey.
""")

# Sidebar for inputs
st.sidebar.header("🏢 Company Configuration")

# Company details
company_name = st.sidebar.text_input("Company Name", value="GreenTech Foods Inc.")
industry = st.sidebar.selectbox(
    "Industry Sector",
    ["Food & Beverage", "Infrastructure", "Technology", "Manufacturing", "Retail", "Energy", "Chemicals"]
)

# Industry-specific benchmarks (based on paper's data)
industry_benchmarks = {
    "Food & Beverage": {"median_boundary_change": 213, "volatility": 0.15, "growth_rate": 0.04},
    "Infrastructure": {"median_boundary_change": 21, "volatility": 0.12, "growth_rate": 0.03},
    "Technology": {"median_boundary_change": 45, "volatility": 0.20, "growth_rate": 0.08},
    "Manufacturing": {"median_boundary_change": 89, "volatility": 0.18, "growth_rate": 0.05},
    "Retail": {"median_boundary_change": 34, "volatility": 0.14, "growth_rate": 0.06},
    "Energy": {"median_boundary_change": 156, "volatility": 0.25, "growth_rate": 0.02},
    "Chemicals": {"median_boundary_change": 124, "volatility": 0.22, "growth_rate": 0.03}
}

# Baseline configuration
st.sidebar.subheader("📊 Baseline Configuration")
baseline_year = st.sidebar.selectbox("Baseline Year", [2024], index=0)  # Fixed to 2024 for 2025-2030 analysis
baseline_emissions = st.sidebar.number_input("Baseline Emissions (MtCO₂e)", value=1000.0, min_value=1.0, step=10.0)

# Scenario selection
st.sidebar.subheader("🎭 Scenario Selection")
scenario = st.sidebar.selectbox(
    "Choose a Scenario",
    [
        "Custom Configuration",
        "The Green Hero (Genuine Leader)",
        "The Method Switcher (Gaming Suspect)",
        "The Growing Gamer (Hidden Growth)",
        "The Industry Laggard (Mixed Signals)"
    ]
)

# Years for analysis - Fixed to 2025-2030 for future-looking analysis
years = list(range(2025, 2031))  # Always 2025-2030

# Scenario presets
if scenario == "The Green Hero (Genuine Leader)":
    genuine_reductions = [-5, -8, -12, -15, -20]  # Strong genuine reductions
    methodological_changes = [1, -2, 1, 0, 1]  # Minor methodological adjustments
    organizational_changes = [0, 0, -3, 0, 0]  # One divestment
    business_growth = [3, 4, 5, 6, 7]  # Steady growth
    
elif scenario == "The Method Switcher (Gaming Suspect)":
    genuine_reductions = [-2, -3, -1, -2, -1]  # Minimal genuine effort
    methodological_changes = [-8, -12, -15, -5, -3]  # Large methodological "improvements"
    organizational_changes = [0, -5, 0, 0, 0]  # Strategic divestment
    business_growth = [6, 8, 7, 9, 8]  # High growth partially hidden
    
elif scenario == "The Growing Gamer (Hidden Growth)":
    genuine_reductions = [-4, -6, -5, -7, -6]  # Moderate genuine effort
    methodological_changes = [-3, -8, -6, -4, -2]  # Convenient method changes
    organizational_changes = [2, -2, 3, -1, 0]  # Mixed M&A activity
    business_growth = [12, 15, 18, 14, 16]  # Rapid growth mostly hidden
    
elif scenario == "The Industry Laggard (Mixed Signals)":
    genuine_reductions = [-1, 2, -3, 1, -2]  # Inconsistent effort
    methodological_changes = [-2, -4, 3, -6, -1]  # Opportunistic changes
    organizational_changes = [0, 0, 5, -2, 0]  # Acquisition then divestment
    business_growth = [4, 3, 6, 5, 4]  # Modest growth
    
else:  # Custom Configuration
    st.sidebar.subheader("🎚️ Custom Parameters")
    
    # Genuine reductions
    st.write("**Genuine Decarbonization (%/year)** - 2025 to 2030")
    genuine_reductions = []
    for i, year in enumerate(years):
        genuine_reductions.append(
            st.sidebar.slider(f"Year {year}", -25, 10, -5, key=f"genuine_{i}")
        )
    
    # Methodological changes
    st.sidebar.write("**Methodological Changes (%/year)** - 2025 to 2030")
    methodological_changes = []
    for i, year in enumerate(years):
        methodological_changes.append(
            st.sidebar.slider(f"Year {year}", -20, 15, 0, key=f"method_{i}")
        )
    
    # Organizational changes
    st.sidebar.write("**Organizational Changes (%/year)** - 2025 to 2030")
    organizational_changes = []
    for i, year in enumerate(years):
        organizational_changes.append(
            st.sidebar.slider(f"Year {year}", -15, 10, 0, key=f"org_{i}")
        )
    
    # Business growth
    st.sidebar.write("**Business Growth Impact (%/year)** - 2025 to 2030")
    business_growth = []
    for i, year in enumerate(years):
        business_growth.append(
            st.sidebar.slider(f"Year {year}", 0, 20, 4, key=f"growth_{i}")
        )

# Set random seed for consistency
np.random.seed(42)

# Calculate emissions trajectory - Simplified for stacked bars
def calculate_emissions_trajectory():
    trajectory_data = []
    current_emissions = baseline_emissions
    
    for i, year in enumerate(years):
        # Calculate each component as absolute values (MtCO₂e)
        genuine_change = current_emissions * (genuine_reductions[i] / 100)
        method_change = current_emissions * (methodological_changes[i] / 100)
        org_change = current_emissions * (organizational_changes[i] / 100)
        growth_change = current_emissions * (business_growth[i] / 100)
        
        # Add some random uncertainty (ε) - consistent seed for reproducibility
        uncertainty = np.random.normal(0, current_emissions * 0.015)
        
        # Update emissions for next year calculation
        total_change = genuine_change + method_change + org_change + growth_change + uncertainty
        current_emissions += total_change
        
        # Store decomposition data for visualization
        trajectory_data.append({
            'year': year,
            'current_emissions': current_emissions,
            'genuine': genuine_change,
            'methodological': method_change,
            'organizational': org_change,
            'growth': growth_change,
            'uncertainty': uncertainty,
            'net_change': total_change
        })
    
    return trajectory_data

# Generate data
decomposition_data = calculate_emissions_trajectory()

# Create DataFrame for plotting
df = pd.DataFrame(decomposition_data)

# Main dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 {company_name} - Emission Analysis (2025-2030)")
    
    # Create simplified stacked bar chart
    fig = go.Figure()
    
    # Add stacked bars for each component - this creates the classic stacked bar view
    fig.add_trace(go.Bar(
        name='Genuine Decarbonization',
        x=df['year'],
        y=df['genuine'],
        marker_color='#2E7D32',  # Green
        hovertemplate='<b>Genuine Decarbonization</b><br>Year: %{x}<br>Change: %{y:.1f} MtCO₂e<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Methodological Changes',
        x=df['year'],
        y=df['methodological'],
        marker_color='#FF9800',  # Orange
        hovertemplate='<b>Methodological Changes</b><br>Year: %{x}<br>Change: %{y:.1f} MtCO₂e<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Organizational Changes',
        x=df['year'],
        y=df['organizational'],
        marker_color='#2196F3',  # Blue
        hovertemplate='<b>Organizational Changes</b><br>Year: %{x}<br>Change: %{y:.1f} MtCO₂e<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Business Growth',
        x=df['year'],
        y=df['growth'],
        marker_color='#F44336',  # Red
        hovertemplate='<b>Business Growth</b><br>Year: %{x}<br>Change: %{y:.1f} MtCO₂e<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Uncertainty',
        x=df['year'],
        y=df['uncertainty'],
        marker_color='#9E9E9E',  # Grey
        hovertemplate='<b>Measurement Uncertainty</b><br>Year: %{x}<br>Change: %{y:.1f} MtCO₂e<extra></extra>'
    ))
    
    # Add horizontal line at zero for reference
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.7)
    
    # Add industry benchmark as scatter points for comparison (not bars to avoid confusion)
    benchmark_decline_rate = 0.042  # 4.2% annual decline
    benchmark_changes = []
    for i, year in enumerate(years):
        # Annual benchmark change (not cumulative)
        annual_benchmark = -baseline_emissions * benchmark_decline_rate
        benchmark_changes.append(annual_benchmark)
    
    fig.add_trace(go.Scatter(
        name='Industry Benchmark',
        x=years,
        y=benchmark_changes,
        mode='markers+lines',
        marker=dict(
            color='black',
            size=8,
            symbol='diamond'
        ),
        line=dict(color='black', width=2, dash='dash'),
        hovertemplate='<b>Industry Benchmark</b><br>Year: %{x}<br>Expected: %{y:.1f} MtCO₂e<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{company_name} - Annual Emission Changes (2025-2030)',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title='Year',
        yaxis_title='Annual Emission Changes (MtCO₂e)',
        barmode='relative',  # This creates the stacked effect
        hovermode='x unified',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(t=80, l=60, r=30, b=140),
        font=dict(size=12),
        showlegend=True,
        xaxis=dict(
            type='category',
            categoryorder='category ascending'
        ),
        # Add reference line annotation
        annotations=[
            dict(
                x=2027.5,
                y=0,
                text="Zero line",
                showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="white",
                borderwidth=1
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🚨 Credibility Assessment")
    
    # Calculate credibility metrics
    total_genuine_abs = sum([abs(x) for x in df['genuine']])
    total_methodological_abs = sum([abs(x) for x in df['methodological']])
    
    # Credibility score calculation
    if total_methodological_abs + total_genuine_abs > 0:
        credibility_ratio = total_genuine_abs / (total_methodological_abs + total_genuine_abs)
        credibility_score = min(100, credibility_ratio * 100)
    else:
        credibility_score = 0
    
    # Display credibility score
    if credibility_score >= 70:
        score_color = "#2E7D32"
        assessment = "High Credibility"
        card_class = "metric-card"
    elif credibility_score >= 40:
        score_color = "#FF9800"
        assessment = "Moderate Credibility"
        card_class = "warning-card"
    else:
        score_color = "#F44336"
        assessment = "Low Credibility"
        card_class = "danger-card"
    
    st.markdown(f"""
    <div class="{card_class}">
        <h3 style="margin: 0; color: {score_color};">Credibility Score: {credibility_score:.1f}/100</h3>
        <p style="margin: 0.5rem 0 0 0; font-weight: bold;">{assessment}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.subheader("📊 Key Metrics")
    
    # Calculate total changes over the period
    total_genuine = sum(df['genuine'])
    total_methodological = sum(df['methodological'])
    total_growth = sum(df['growth'])
    total_net_change = sum(df['net_change'])
    
    final_emissions = baseline_emissions + total_net_change
    total_change_percent = (total_net_change / baseline_emissions) * 100
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            label="Net Change",
            value=f"{total_change_percent:.1f}%",
            delta=f"{total_change_percent + 21:.1f}% vs -21% Target"  # SBTi 2025-2030 target
        )
    
    with col_b:
        st.metric(
            label="Final Emissions",
            value=f"{final_emissions:.0f} MtCO₂e",
            delta=f"{total_net_change:.0f} MtCO₂e"
        )
    
    # Red flags
    st.subheader("🚩 Red Flags")
    
    red_flags = []
    
    # Check for suspicious methodological changes
    max_method_change = max([abs(x) for x in methodological_changes])
    if max_method_change > 10:
        red_flags.append(f"Large methodological change: {max_method_change:.1f}%")
    
    # Check for timing of changes
    method_reductions = [x for x in methodological_changes if x < 0]
    if len(method_reductions) >= 3:
        red_flags.append("Frequent 'convenient' methodology updates")
    
    # Check ratio of methodological vs genuine
    if total_methodological_abs > total_genuine_abs and total_methodological_abs > 0:
        red_flags.append("Methodological changes exceed genuine efforts")
    
    # Check for organizational timing
    org_reductions = [x for x in organizational_changes if x < 0]
    if len(org_reductions) >= 2:
        red_flags.append("Multiple strategic divestments")
    
    if red_flags:
        for flag in red_flags:
            st.error(f"⚠️ {flag}")
    else:
        st.success("✅ No major red flags detected")
    
    # Industry comparison
    st.subheader("🏭 Industry Context")
    benchmark = industry_benchmarks[industry]
    
    st.write(f"**{industry} Sector Characteristics:**")
    st.write(f"• Median boundary sensitivity: {benchmark['median_boundary_change']}%")
    st.write(f"• Typical volatility: {benchmark['volatility']*100:.1f}%")
    st.write(f"• Average growth rate: {benchmark['growth_rate']*100:.1f}%")

# Detailed analysis section
st.subheader("🔬 Detailed Decomposition Analysis")

col1, col2 = st.columns(2)

with col1:
    # Show the equation breakdown
    st.write("**Mathematical Decomposition (Equation 1):**")
    equation_df = pd.DataFrame({
        'Component': ['Δ_genuine', 'Δ_methodological', 'Δ_organizational', 'Δ_growth', 'ε'],
        'Description': [
            'Actual decarbonization interventions',
            'Changes in emission factors/methodology',
            'M&A, divestments, restructuring',
            'Business expansion impacts',
            'Measurement uncertainty'
        ],
        'Total Impact (MtCO₂e)': [
            sum(df['genuine']),
            sum(df['methodological']),
            sum(df['organizational']),
            sum(df['growth']),
            sum(df['uncertainty'])
        ]
    })
    
    # Color code the dataframe
    def color_values(val):
        if val < 0:
            return 'color: green'
        elif val > 0:
            return 'color: red'
        else:
            return 'color: black'
    
    styled_df = equation_df.style.applymap(color_values, subset=['Total Impact (MtCO₂e)'])
    st.dataframe(styled_df, use_container_width=True)

with col2:
    # Pie chart of absolute contributions
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Genuine Decarbonization', 'Methodological Changes', 'Organizational Changes', 'Business Growth', 'Uncertainty'],
        values=[abs(sum(df['genuine'])), abs(sum(df['methodological'])), abs(sum(df['organizational'])), 
                abs(sum(df['growth'])), abs(sum(df['uncertainty']))],
        marker_colors=['#2E7D32', '#FF9800', '#2196F3', '#F44336', '#9E9E9E'],
        hovertemplate='<b>%{label}</b><br>Impact: %{value:.1f} MtCO₂e<br>Share: %{percent}<extra></extra>'
    )])
    
    fig_pie.update_layout(
        title="Absolute Impact Distribution",
        height=400,
        margin=dict(t=50, l=20, r=20, b=20),
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# Action recommendations
st.subheader("💡 Recommendations & Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**For Investors:**")
    if credibility_score >= 70:
        st.success("• High confidence in reported progress\n• Suitable for green investment criteria\n• Track continued performance")
    elif credibility_score >= 40:
        st.warning("• Request additional verification\n• Monitor methodology changes\n• Seek third-party validation")
    else:
        st.error("• High greenwashing risk\n• Avoid green investment classification\n• Demand fundamental transparency")

with col2:
    st.write("**For Regulators:**")
    if len(red_flags) > 2:
        st.error("• Requires detailed audit\n• Consider mandatory reporting\n• Investigate methodology shifts")
    elif len(red_flags) > 0:
        st.warning("• Enhanced monitoring needed\n• Request justification for changes\n• Compare against industry peers")
    else:
        st.success("• Compliance appears adequate\n• Use as industry benchmark\n• Monitor continued progress")

with col3:
    st.write("**For Company:**")
    total_genuine_sum = sum(df['genuine'])  # Note: negative values are good for genuine reductions
    total_methodological_sum = sum(df['methodological'])
    
    if abs(total_genuine_sum) > abs(total_methodological_sum):
        st.success("• Strong decarbonization efforts\n• Maintain transparency\n• Share best practices")
    else:
        st.warning("• Focus on genuine interventions\n• Reduce methodology changes\n• Improve verification processes")

# Data export
st.subheader("📥 Export Analysis")

if st.button("Generate Verification Report"):
    # Calculate summary metrics
    final_emissions_calc = baseline_emissions + sum(df['net_change'])
    total_change_calc = (sum(df['net_change']) / baseline_emissions) * 100
    
    report_data = {
        'Company': company_name,
        'Industry': industry,
        'Baseline_Year': 2024,
        'Analysis_Period': '2025-2030',
        'Baseline_Emissions_MtCO2e': baseline_emissions,
        'Final_Emissions_MtCO2e': final_emissions_calc,
        'Total_Change_Percent': total_change_calc,
        'Credibility_Score': credibility_score,
        'Assessment': assessment,
        'Red_Flags': len(red_flags),
        'Genuine_Reduction_MtCO2e': sum(df['genuine']),
        'Methodological_Changes_MtCO2e': sum(df['methodological']),
        'Organizational_Changes_MtCO2e': sum(df['organizational']),
        'Growth_Impact_MtCO2e': sum(df['growth']),
        'Uncertainty_MtCO2e': sum(df['uncertainty'])
    }
    
    report_df = pd.DataFrame([report_data])
    csv = report_df.to_csv(index=False)
    
    st.download_button(
        label="Download Verification Report (CSV)",
        data=csv,
        file_name=f"{company_name.replace(' ', '_')}_verification_report.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**About this tool:** Based on the research framework from "Operationalizing corporate climate action through five research frontiers" - RF3: Progress Tracking. 
This tool demonstrates how mathematical decomposition can distinguish genuine decarbonization from accounting manipulations, 
addressing the verification crisis in corporate climate action. **Timeline: 2025-2030 for future-looking analysis.**

**License:** This tool is licensed under CC BY-NC 4.0 for research and educational use only. Commercial use requires permission.

**Source:** [Research Paper] (Send request to ramana.gudipudi@gmail.com)  | 
**GitHub:** [Repository](https://github.com/RamanaGudipudi/decarbonization-detective) | 
**Connect:** [LinkedIn](https://linkedin.com/in/ramana-gudipudi)
""")
