# Corporate Decarbonization Detective 🔍

An interactive Streamlit application that demonstrates **RF3: Progress Tracking** from the research paper "Operationalizing corporate climate action through five research frontiers". This tool mathematically decomposes corporate emission changes to distinguish genuine decarbonization from accounting manipulations.

## 🚀 Live Demo

**[🔗 Try the App](https://your-app-name.streamlit.app/)**

## 📋 Overview

The Corporate Decarbonization Detective addresses the verification crisis in corporate climate action by implementing the mathematical framework:

```
Δ_observed = Δ_genuine + Δ_methodological + Δ_organizational + Δ_growth + ε
```

This equation decomposes reported emission changes into:
- **Genuine decarbonization**: Actual emission reductions from climate interventions
- **Methodological changes**: Variations due to emission factor or boundary changes
- **Organizational changes**: M&A, divestments, and restructuring impacts
- **Business growth**: Emission increases from business expansion
- **Uncertainty**: Measurement errors and external factors

## ✨ Key Features

### 🔍 **Mathematical Decomposition Engine**
- Real-time visualization of emission change components
- Interactive parameter adjustment with immediate feedback
- Industry-specific benchmarking and peer comparison

### 🎭 **Scenario-Based Learning**
Four preset scenarios demonstrate different corporate behaviors:
- **The Green Hero**: Genuine climate leader with authentic reductions
- **The Method Switcher**: Gaming suspect relying on accounting tricks
- **The Growing Gamer**: Rapid growth hidden by methodological changes
- **The Industry Laggard**: Inconsistent performance with mixed signals

### 🚨 **Gaming Detection Dashboard**
- Automated red flag identification for suspicious patterns
- Credibility scoring (0-100) based on genuine vs. methodological changes
- Industry context with sector-specific volatility benchmarks

### 📊 **Interactive Visualizations**
- Stacked bar charts showing emission decomposition over time
- Dynamic industry benchmark trajectory
- Pie charts displaying absolute impact distribution
- Comprehensive verification reports

## 🛠️ Technical Implementation

### Built With
- **Streamlit** - Web application framework
- **Plotly** - Interactive data visualization
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Key Components
- **Emission Trajectory Calculator**: Implements the mathematical decomposition model
- **Industry Benchmark System**: Based on NZDPU data analysis from the research paper
- **Credibility Assessment Engine**: Automated scoring and red flag detection
- **Scenario Management**: Preset configurations for educational demonstration

## 📚 Research Foundation

This application is based on **Research Frontier 3 (RF3): Progress Tracking** from the paper:

> **"Operationalizing corporate climate action through five research frontiers"**
> 
> *Authors: Ramana Gudipudi, Luis Costa, Ponraj Arumugam, Matthew Agarwala, Jürgen P. Kropp, Felix Creutzig*

### Key Research Insights Implemented:
1. **Verification Crisis**: 85% of companies cite Scope 3 accounting as their primary obstacle
2. **Emissions Gaming**: Strategic selection can artificially show 30% reductions despite actual increases
3. **Boundary Sensitivity**: Companies changing boundaries experience fluctuations exceeding 30,000% in some sectors
4. **Mathematical Decomposition**: Framework to separate genuine climate action from accounting manipulation

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/decarbonization-detective.git
   cd decarbonization-detective
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 🎯 Usage Guide

### For Investors
- **High Credibility Score (70+)**: Suitable for green investment criteria
- **Moderate Score (40-69)**: Requires additional verification and monitoring
- **Low Score (<40)**: High greenwashing risk, avoid green classifications

### For Regulators
- **Multiple Red Flags**: Requires detailed audit and investigation
- **Some Red Flags**: Enhanced monitoring and peer comparison needed
- **No Red Flags**: Compliance adequate, suitable as industry benchmark

### For Corporations
- **Strong Genuine Component**: Maintain transparency and share best practices
- **Weak Genuine Component**: Focus on actual interventions vs. accounting changes

## 📊 Industry Benchmarks

The app includes sector-specific characteristics based on research analysis:

| Industry | Boundary Sensitivity | Volatility | Growth Rate |
|----------|---------------------|------------|-------------|
| Food & Beverage | 213% | 15% | 4% |
| Infrastructure | 21% | 12% | 3% |
| Technology | 45% | 20% | 8% |
| Manufacturing | 89% | 18% | 5% |
| Energy | 156% | 25% | 2% |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup
git clone https://github.com/your-username/decarbonization-detective.git
cd decarbonization-detective
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run locally
streamlit run streamlit_app.py
```

## 📄 License & Usage

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** - see the [LICENSE](LICENSE) file for details.

### ✅ **Permitted Uses:**
- **Academic research** and educational purposes
- **Non-profit climate initiatives** and NGO applications  
- **Personal learning** and skill development
- **Open source contributions** and improvements
- **Government and regulatory** applications

### ❌ **Prohibited Uses:**
- **Commercial products or services** without permission
- **For-profit consulting** using this methodology
- **Corporate internal tools** for commercial advantage
- **Revenue-generating applications** of any kind

### 🤝 **Commercial Licensing**
For commercial use, enterprise licensing, or integration into for-profit ventures, please contact **[your-email@domain.com]** for permission and licensing terms.

### 🎓 **Research Citation**
If you use this tool in academic research, please cite:
```
Gudipudi, R., et al. (2025). Corporate Decarbonization Detective: 
An Interactive Tool for RF3 Progress Tracking Framework. 
GitHub repository: https://github.com/RamanaGudipudi/decarbonization-detective
```

## 📞 Contact & Links

- **Related Apps**: 
  - [Emission Gaming Demo](https://emission-gaming-demo.streamlit.app/)
  - [RF2 Climate Simulator](https://rf2-climate-simulator-zgkwcpx5j5qkfvdmpbn9y6.streamlit.app/)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/your-profile)
- **GitHub**: [Your GitHub](https://github.com/your-username)

## 🎓 Educational Use Cases

### Corporate Training
- Demonstrate emission accounting complexities to sustainability teams
- Train auditors on red flag identification
- Educate investors on greenwashing detection

### Academic Research
- Validate RF3 framework with real corporate data
- Extend methodology to other planetary boundaries
- Compare across different regulatory frameworks

### Policy Development
- Inform enhanced disclosure requirements
- Design verification protocols for mandatory reporting
- Benchmark industry-specific standards

## 🔬 Technical Details

### Mathematical Framework
The core decomposition algorithm implements:
```python
def calculate_emissions_trajectory():
    for year in projection_period:
        genuine_change = emissions * (genuine_rate / 100)
        method_change = emissions * (method_rate / 100)
        org_change = emissions * (org_rate / 100)
        growth_change = emissions * (growth_rate / 100)
        uncertainty = normal_distribution(0, emissions * volatility)
        
        observed_change = sum([genuine_change, method_change, 
                              org_change, growth_change, uncertainty])
        
        emissions += observed_change
```

### Industry Benchmark Calculation
Dynamic benchmarks based on:
- **SBTi uniform approach**: 4.2% annual decline
- **Sector volatility**: Historical emission variance
- **Boundary sensitivity**: Median changes from NZDPU data

### Credibility Scoring Algorithm
```python
credibility_score = (genuine_effort / (genuine_effort + methodological_gaming)) * 100
```

## 📈 Data Sources

1. **Net-Zero Data Public Utility (NZDPU)**: Corporate disclosure analysis
2. **Science Based Targets initiative (SBTi)**: Target-setting methodologies
3. **IPCC AR6**: Sectoral decarbonization pathways
4. **Industry surveys**: Business Ambition for 1.5°C campaign data

## 🚀 Deployment

### Streamlit Community Cloud
1. Fork this repository
2. Connect to Streamlit Community Cloud
3. Deploy with one click
4. Share your custom URL

### Alternative Deployments
- **Heroku**: `git push heroku main`
- **Docker**: Use provided Dockerfile
- **Local**: `streamlit run streamlit_app.py`

## 🐛 Known Issues & Roadmap

### Current Limitations
- Static industry benchmarks (working on dynamic updates)
- Limited to 5-year projections
- Simplified uncertainty modeling

### Future Enhancements
- [ ] Real-time corporate data integration
- [ ] Multi-pathway scenario analysis
- [ ] Extended planetary boundary metrics
- [ ] API for programmatic access
- [ ] Machine learning red flag detection
- [ ] Peer comparison database

## 🏆 Recognition

This tool demonstrates practical implementation of academic climate research, bridging the gap between scientific frameworks and corporate decision-making tools.

## 📚 References

1. Gudipudi, R., et al. "Operationalizing corporate climate action through five research frontiers" (2025)
2. Aikman, D., et al. "Emissions Gaming? A Gap in the GHG Protocol" (2023)
3. Science Based Targets initiative. "Corporate Net-Zero Standard Version 2.0" (2025)
4. Net-Zero Data Public Utility. Corporate Climate Disclosure Database (2024)

---

**Built with 💚 for transparent corporate climate action**
