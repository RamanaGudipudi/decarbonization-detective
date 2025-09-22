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
baseline_year = st.sidebar.selectbox("Baseline Year", [2024], index=0)
baseline_emissions = st.sidebar.number_input("Baseline Emissions (MtCO₂e)", value=1000.0, min_value=1.0, step=10.0)

# Scenario selection
st.sidebar.subheader("🎭 Scenario Selection")
scenario = st.sidebar.selectbox(
    "Choose a Scenario",
    [
        "The Green Hero (Genuine Leader)",
        "The Method Switcher (Gaming Suspect)", 
        "The Growing Gamer (Hidden Growth)",
        "The Industry Laggard (Mixed Signals)",
        "Custom Configuration"
    ]
)

# Years for analysis - Fixed to 2025-2030 (6 years of data)
years = list(range(2025, 2031))  # [2025, 2026, 2027, 2028, 2029, 2030]
num_years = len(years)  # 6 years

# Initialize all parameter arrays with default values to prevent errors
genuine_reductions = [-5.0] * num_years
methodological_changes = [0.0] * num_years  
organizational_changes = [0.0] * num_years
business_growth = [4.0] * num_years

# Scenario presets - all arrays must have exactly 6 elements
if scenario == "The Green Hero (Genuine Leader)":
    genuine_reductions = [-5, -8, -12, -15, -20, -25]
    methodological_changes = [1, -2, 1, 0, 1, 0]
    organizational_changes = [0, 0, -3, 0, 0, 0]
    business_growth = [3, 4, 5, 6, 7, 8]
    
elif scenario == "The Method Switcher (Gaming Suspect)":
    genuine_reductions = [-2, -3, -1, -2, -1, -2]
    methodological_changes = [-8, -12, -15, -5, -3, -6]
    organizational_changes = [0, -5, 0, 0, 0, -2]
    business_growth = [6, 8, 7, 9, 8, 10]
    
elif scenario == "The Growing Gamer (Hidden Growth)":
    genuine_reductions = [-4, -6, -5, -7, -6, -8]
    methodological_changes = [-3, -8, -6, -4, -2, -5]
    organizational_changes = [2, -2, 3, -1, 0, 1]
    business_growth = [12, 15, 18, 14, 16, 17]
    
elif scenario == "The Industry Laggard (Mixed Signals)":
    genuine_reductions = [-1, 2, -3, 1, -2, 0]
    methodological_changes = [-2, -4, 3, -6, -1, -3]
    organizational_changes = [0, 0, 5, -2, 0, 1]
    business_growth = [4, 3, 6, 5, 4, 5]
    
elif scenario == "Custom Configuration":
    st.sidebar.subheader("🎚️ Custom Parameters")
    
    # Reset arrays for custom configuration
    genuine_reductions = []
    methodological_changes = []
    organizational_changes = []
    business_growth = []
    
    # Genuine reductions
    st.sidebar.write("**Genuine Decarbonization (%/year)**")
    for i, year in enumerate(years):
        value = st.sidebar.slider(
            f"Genuine {year}", 
            min_value=-30, max_value=10, value=-5, 
            key=f"genuine_{year}_{i}"  # Unique key using both year and index
        )
        genuine_reductions.append(float(value))
    
    # Methodological changes
    st.sidebar.write("**Methodological Changes (%/year)**")
    for i, year in enumerate(years):
        value = st.sidebar.slider(
            f"Method {year}", 
            min_value=-25, max_value=20, value=0,
            key=f"method_{year}_{i}"
        )
        methodological_changes.append(float(value))
    
    # Organizational changes  
    st.sidebar.write("**Organizational Changes (%/year)**")
    for i, year in enumerate(years):
        value = st.sidebar.slider(
            f"Org {year}", 
            min_value=-20, max_value=15, value=0,
            key=f"org_{year}_{i}"
        )
        organizational_changes.append(float(value))
    
    # Business growth
    st.sidebar.write("**Business Growth Impact (%/year)**")
    for i, year in enumerate(years):
        value = st.sidebar.slider(
            f"Growth {year}", 
            min_value=0, max_value=25, value=4,
            key=f"growth_{year}_{i}"
        )
        business_growth.append(float(value))

# Verify all arrays have correct length (safety check)
if len(genuine_reductions) != num_years:
    st.error(f"Error: Genuine reductions array has {len(genuine_reductions)} elements, expected {num_years}")
if len(methodological_changes) != num_years:
    st.error(f"Error: Methodological changes array has {len(methodological_changes)} elements, expected {num_years}")
if len(organizational_changes) != num_years:
    st.error(f"Error: Organizational changes array has {len(organizational_changes)} elements, expected {num_years}")
if len(business_growth) != num_years:
    st.error(f"Error: Business growth array has {len(business_growth)} elements, expected {num_years}")

# Set random seed for reproducible uncertainty
np.random.seed(42)

# Calculate emissions trajectory
def calculate_emissions_trajectory():
    """Calculate the emission trajectory with proper decomposition"""
    if not all(len(arr) == num_years for arr in [genuine_reductions, methodological_changes, organizational_changes, business_growth]):
        st.error("Array length mismatch detected!")
        return []
        
    trajectory_data = []
    current_emissions = float(baseline_emissions)
    
    for i in range(num_years):
        year = years[i]
        
        # Calculate each component as absolute values (MtCO₂e)
        genuine_change = current_emissions * (genuine_reductions[i] / 100.0)
        method_change = current_emissions * (methodological_changes[i] / 100.0)
        org_change = current_emissions * (organizational_changes[i] / 100.0)
        growth_change = current_emissions * (business_growth[i] / 100.0)
        
        # Add uncertainty (ε) - small random component
        uncertainty = np.random.normal(0, current_emissions * 0.01)
        
        # Calculate total change
        total_change = genuine_change + method_change + org_change + growth_change + uncertainty
        
        # Update emissions for next year
        current_emissions = max(0, current_emissions + total_change)  # Prevent negative emissions
        
        # Store data
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

# Generate trajectory data
try:
    decomposition_data = calculate_emissions_trajectory()
    
    if not decomposition_data:
        st.error("Failed to calculate emissions trajectory")
        st.stop()
        
    # Create DataFrame
    df = pd.DataFrame(decomposition_data)
    
except Exception as e:
    st.error(f"Error in trajectory calculation: {str(e)}")
    st.stop()

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 {company_name} - Emission Analysis (2025-2030)")
    
    try:
        # Create stacked bar chart
        fig = go.Figure()
        
        # Add decomposition components
        components = [
            ('Genuine Decarbonization', df['genuine'], '#2E7D32'),
            ('Methodological Changes', df['methodological'], '#FF9800'),
            ('Organizational Changes', df['organizational'], '#2196F3'),
            ('Business Growth', df['growth'], '#F44336'),
            ('Uncertainty', df['uncertainty'], '#9E9E9E')
        ]
        
        for name, values, color in components:
            fig.add_trace(go.Bar(
                name=name,
                x=df['year'],
                y=values,
                marker_color=color,
                hovertemplate=f'<b>{name}</b><br>Year: %{{x}}<br>Change: %{{y:.1f}} MtCO₂e<extra></extra>',
                offsetgroup=1
            ))
        
        # Add zero reference line
        fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
        
        # Add industry benchmark
        benchmark_annual = -baseline_emissions * 0.042  # 4.2% annual decline
        fig.add_trace(go.Bar(
            name='SBTi Benchmark (-4.2%/year)',
            x=years,
            y=[benchmark_annual] * num_years,
            marker_color='rgba(0,0,0,0.3)',
            marker_line_color='black',
            marker_line_width=2,
            hovertemplate='<b>SBTi Annual Target</b><br>Year: %{x}<br>Target: %{y:.1f} MtCO₂e<extra></extra>',
            offsetgroup=2,
            width=0.3
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{company_name} - Annual Emission Changes Decomposition',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Year',
            yaxis_title='Annual Emission Changes (MtCO₂e)',
            barmode='group',
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(t=80, l=60, r=30, b=160),
            font=dict(size=12),
            showlegend=True,
            xaxis=dict(
                type='category',
                categoryorder='category ascending',
                tickangle=0
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

with col2:
    st.subheader("🚨 Credibility Assessment")
    
    try:
        # Calculate credibility metrics
        total_genuine_abs = sum([abs(x) for x in df['genuine']])
        total_methodological_abs = sum([abs(x) for x in df['methodological']])
        
        # Avoid division by zero
        if (total_methodological_abs + total_genuine_abs) > 0:
            credibility_ratio = total_genuine_abs / (total_methodological_abs + total_genuine_abs)
            credibility_score = min(100, credibility_ratio * 100)
        else:
            credibility_score = 50  # Neutral score if no significant changes
        
        # Determine assessment category
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
        
        # Display credibility score
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin: 0; color: {score_color};">Credibility Score: {credibility_score:.1f}/100</h3>
            <p style="margin: 0.5rem 0 0 0; font-weight: bold;">{assessment}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error calculating credibility: {str(e)}")
        credibility_score = 0
        assessment = "Error"
    
    # Key metrics
    st.subheader("📊 Key Metrics")
    
    try:
        # Calculate totals
        total_genuine = sum(df['genuine'])
        total_methodological = sum(df['methodological'])
        total_growth = sum(df['growth'])
        total_net_change = sum(df['net_change'])
        
        final_emissions = baseline_emissions + total_net_change
        total_change_percent = (total_net_change / baseline_emissions) * 100
        
        # SBTi target: 4.2% annual decline over 6 years ≈ 21% total
        sbti_target_percent = -4.2 * len(years)
        target_gap = total_change_percent - sbti_target_percent
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="Net Change",
                value=f"{total_change_percent:.1f}%",
                delta=f"{target_gap:.1f}% vs target"
            )
        
        with col_b:
            st.metric(
                label="Final Emissions",
                value=f"{final_emissions:.0f} MtCO₂e",
                delta=f"{total_net_change:+.0f} MtCO₂e"
            )
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
    
    # Red flags analysis
    st.subheader("🚩 Red Flags")
    
    try:
        red_flags = []
        
        # Check for large methodological changes
        max_method_change = max([abs(x) for x in methodological_changes])
        if max_method_change > 10:
            red_flags.append(f"Large methodological change: {max_method_change:.1f}%")
        
        # Check for frequent methodological reductions
        method_reductions = [x for x in methodological_changes if x < -2]
        if len(method_reductions) >= 3:
            red_flags.append("Frequent 'convenient' methodology updates")
        
        # Check dominance of methodological vs genuine changes
        if total_methodological_abs > total_genuine_abs * 1.5:
            red_flags.append("Methodological changes dominate genuine efforts")
        
        # Check for strategic organizational changes
        org_reductions = [x for x in organizational_changes if x < -2]
        if len(org_reductions) >= 2:
            red_flags.append("Multiple strategic divestments detected")
        
        # Check for emission increase despite claimed reductions
        if total_change_percent > 0 and sum([x for x in methodological_changes if x < 0]) < -10:
            red_flags.append("Net increase despite methodological 'improvements'")
        
        # Display red flags
        if red_flags:
            for flag in red_flags[:5]:  # Limit to 5 most important flags
                st.error(f"⚠️ {flag}")
        else:
            st.success("✅ No major red flags detected")
        
    except Exception as e:
        st.error(f"Error in red flag analysis: {str(e)}")
        red_flags = []
    
    # Industry comparison
    st.subheader("🏭 Industry Context")
    
    try:
        benchmark = industry_benchmarks[industry]
        
        st.write(f"**{industry} Sector:**")
        st.write(f"• Boundary sensitivity: {benchmark['median_boundary_change']}%")
        st.write(f"• Typical volatility: {benchmark['volatility']*100:.1f}%")
        st.write(f"• Avg. growth rate: {benchmark['growth_rate']*100:.1f}%")
        
    except Exception as e:
        st.error(f"Error displaying industry context: {str(e)}")

# Detailed Analysis Section
st.subheader("🔬 Detailed Decomposition Analysis")

try:
    col1, col2 = st.columns(2)

    with col1:
        # Mathematical decomposition table
        st.write("**Mathematical Decomposition (RF3 Framework):**")
        
        equation_data = {
            'Component': ['Δ_genuine', 'Δ_methodological', 'Δ_organizational', 'Δ_growth', 'ε'],
            'Description': [
                'Actual decarbonization efforts',
                'Changes in emission factors/methods',
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
            ],
            'Avg Annual (%)': [
                np.mean(genuine_reductions),
                np.mean(methodological_changes),
                np.mean(organizational_changes),
                np.mean(business_growth),
                sum(df['uncertainty']) / baseline_emissions * 100 / num_years
            ]
        }
        
        equation_df = pd.DataFrame(equation_data)
        
        # Format the dataframe with colors
        def color_negative_red_positive_green(val):
            if isinstance(val, (int, float)):
                color = 'color: green' if val < 0 else 'color: red' if val > 0 else 'color: black'
                return color
            return 'color: black'
        
        styled_df = equation_df.style.applymap(
            color_negative_red_positive_green, 
            subset=['Total Impact (MtCO₂e)', 'Avg Annual (%)']
        )
        
        st.dataframe(styled_df, use_container_width=True)

    with col2:
        # Impact distribution pie chart
        st.write("**Impact Distribution:**")
        
        absolute_values = [
            abs(sum(df['genuine'])),
            abs(sum(df['methodological'])),
            abs(sum(df['organizational'])),
            abs(sum(df['growth'])),
            abs(sum(df['uncertainty']))
        ]
        
        labels = ['Genuine', 'Methodological', 'Organizational', 'Growth', 'Uncertainty']
        colors = ['#2E7D32', '#FF9800', '#2196F3', '#F44336', '#9E9E9E']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=absolute_values,
            marker_colors=colors,
            hovertemplate='<b>%{label}</b><br>Impact: %{value:.1f} MtCO₂e<br>Share: %{percent}<extra></extra>',
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig_pie.update_layout(
            title="Absolute Impact Distribution",
            height=400,
            margin=dict(t=50, l=20, r=20, b=20),
            font=dict(size=11),
            showlegend=False
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

except Exception as e:
    st.error(f"Error in detailed analysis: {str(e)}")

# Recommendations Section
st.subheader("💡 Stakeholder Recommendations")

try:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**For Investors:**")
        if credibility_score >= 70:
            st.success("• High confidence in progress\n• Suitable for ESG criteria\n• Monitor continued performance")
        elif credibility_score >= 40:
            st.warning("• Request additional verification\n• Monitor methodology changes\n• Third-party validation needed")
        else:
            st.error("• High greenwashing risk\n• Avoid ESG classification\n• Demand transparency reforms")

    with col2:
        st.write("**For Regulators:**")
        if len(red_flags) > 2:
            st.error("• Requires immediate audit\n• Consider enforcement action\n• Investigate methodology shifts")
        elif len(red_flags) > 0:
            st.warning("• Enhanced monitoring required\n• Request change justifications\n• Compare with industry peers")
        else:
            st.success("• Compliance appears adequate\n• Suitable as benchmark\n• Continue routine monitoring")

    with col3:
        st.write("**For Company:**")
        if abs(total_genuine) > abs(total_methodological):
            st.success("• Strong decarbonization focus\n• Maintain transparency\n• Share best practices")
        else:
            st.warning("• Prioritize genuine interventions\n• Reduce methodology gaming\n• Improve verification systems")

except Exception as e:
    st.error(f"Error generating recommendations: {str(e)}")

# Export functionality
st.subheader("📥 Export Analysis")

try:
    if st.button("📊 Generate Verification Report"):
        
        report_data = {
            'Company': company_name,
            'Industry': industry,
            'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'Scenario': scenario,
            'Baseline_Year': baseline_year,
            'Baseline_Emissions_MtCO2e': baseline_emissions,
            'Final_Emissions_MtCO2e': final_emissions,
            'Total_Change_Percent': total_change_percent,
            'Credibility_Score': credibility_score,
            'Assessment': assessment,
            'Red_Flags_Count': len(red_flags),
            'Genuine_Total_MtCO2e': sum(df['genuine']),
            'Methodological_Total_MtCO2e': sum(df['methodological']),
            'Organizational_Total_MtCO2e': sum(df['organizational']),
            'Growth_Total_MtCO2e': sum(df['growth']),
            'Uncertainty_Total_MtCO2e': sum(df['uncertainty']),
            'SBTi_Target_Gap_Percent': target_gap
        }
        
        # Create detailed yearly breakdown
        yearly_data = []
        for i, year in enumerate(years):
            yearly_data.append({
                'Year': year,
                'Genuine_Pct': genuine_reductions[i],
                'Methodological_Pct': methodological_changes[i],
                'Organizational_Pct': organizational_changes[i],
                'Growth_Pct': business_growth[i],
                'Genuine_MtCO2e': df.iloc[i]['genuine'],
                'Methodological_MtCO2e': df.iloc[i]['methodological'], 
                'Organizational_MtCO2e': df.iloc[i]['organizational'],
                'Growth_MtCO2e': df.iloc[i]['growth'],
                'Uncertainty_MtCO2e': df.iloc[i]['uncertainty'],
                'Total_Emissions_MtCO2e': df.iloc[i]['current_emissions']
            })
        
        # Combine summary and yearly data
        summary_df = pd.DataFrame([report_data])
        yearly_df = pd.DataFrame(yearly_data)
        
        # Create Excel file with multiple sheets
        from io import BytesIO
        excel_buffer = BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            yearly_df.to_excel(writer, sheet_name='Yearly_Breakdown', index=False)
            
            # Add red flags as a separate sheet if any exist
            if red_flags:
                red_flags_df = pd.DataFrame({
                    'Red_Flag': red_flags,
                    'Severity': ['High'] * len(red_flags)
                })
                red_flags_df.to_excel(writer, sheet_name='Red_Flags', index=False)
        
        excel_buffer.seek(0)
        
        st.download_button(
            label="📊 Download Complete Verification Report (Excel)",
            data=excel_buffer.getvalue(),
            file_name=f"{company_name.replace(' ', '_')}_{scenario.replace(' ', '_')}_verification_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Also offer CSV for simple data
        csv_data = yearly_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Yearly Data (CSV)",
            data=csv_data,
            file_name=f"{company_name.replace(' ', '_')}_yearly_data.csv",
            mime="text/csv"
        )
        
        st.success("✅ Reports generated successfully!")

except Exception as e:
    st.error(f"Error generating reports: {str(e)}")

# Footer with methodology and sources
st.markdown("---")
st.markdown("""
### 📚 **Methodology & Sources**

**Framework:** Based on RF3 (Progress Tracking) from "Operationalizing corporate climate action through five research frontiers"

**Equation:** $\\Delta_{observed} = \\Delta_{genuine} + \\Delta_{methodological} + \\Delta_{organizational} + \\Delta_{growth} + \\varepsilon$

**Key Features:**
- ✅ Mathematical decomposition of emission changes
- ✅ Credibility scoring based on genuine vs. gaming indicators  
- ✅ Industry-specific benchmarking
- ✅ Red flag detection for greenwashing patterns
- ✅ Multi-scenario analysis capabilities

**Timeline:** 2025-2030 future projection analysis

---
**🔗 Links:** [Research Paper](https://docs.google.com/document/d/1NcvZDqKb9h1VeNMvsexI_OG29bEjrq3pwQoYx5G5YkU/edit) | 
[GitHub Repository](https://github.com/RamanaGudipudi/decarbonization-detective) | 
[Connect on LinkedIn](https://linkedin.com/in/ramana-gudipudi)

**⚠️ Disclaimer:** This tool is for analytical purposes only. Actual corporate climate verification should involve comprehensive third-party auditing and validation processes.
""")
