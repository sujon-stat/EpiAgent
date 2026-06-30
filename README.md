# 🦠 EpiAgent: Secure Multi-Agent Epidemic Surveillance & Predictive Modeling System

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://python.org)
[![Google ADK 2.3](https://img.shields.io/badge/Google%20ADK-2.3.0-4285F4.svg)](https://google.github.io/adk-docs/)
[![Tests](https://img.shields.io/badge/tests-70%2F70%20passing-brightgreen.svg)]()
[![HIPAA](https://img.shields.io/badge/HIPAA-Safe%20Harbor%20Compliant-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An autonomous public health intelligence pipeline that streams surveillance data, applies strict security guardrails, runs deterministic compartmental models to detect outbreak signals, and generates executive situation reports.**

Built for the **Google/Kaggle 5-Day AI Agents Intensive Course** — Track: *"Agents for Good" (Healthcare)*.

🎬 **[Watch the Complete 7-Minute Video Presentation on YouTube](https://youtu.be/yB7ILJe-SAc)**  
📖 **[Read the Competition Write-Up / Blog Post](blog_post.md)**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              EpiAgent Sequential Pipeline                     │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  1. Data  │→│2. Security│→│3. Validate│→│4. Analyze │     │
│  │  Agent    │  │  Agent   │  │  Agent   │  │  Agent    │     │
│  │          │  │ (HIPAA)  │  │(8 checks)│  │(SEIR, Rt) │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│                                               │               │
│                              ┌──────────┐  ┌──┴───────┐     │
│                              │6. SitRep │←│5. ML/SHAP│     │
│                              │  Agent   │  │  Agent   │     │
│                              └──────────┘  └──────────┘     │
└──────────────────────────────────────────────────────────────┘
                         ↓
              📊 Interactive Dashboard
              📝 Executive Situation Report
```

### Key Design Principle: **LLM Decides, Math Computes, LLM Interprets**

Every numerical result comes from a deterministic `FunctionTool` — never from LLM generation. The agents decide *when* to call tools and *how to interpret* results, but the math is always exact and reproducible.

---

## 📊 What EpiAgent Does

| Step | Agent | Methods | Output |
|------|-------|---------|--------|
| 1 | **Data Retrieval** | MCP Server, Synthetic SEIR, CDC FluView API | Surveillance time series |
| 2 | **Security Guardrail** | 18 HIPAA Safe Harbor patterns, SHA-256 hashing | PII-free data + audit trail |
| 3 | **Data Validation** | 8-point epidemiological plausibility checks | Quality score (0-100%) |
| 4 | **Epi Analysis** | SEIR (RK45), Cori Rt, Wilson CFR, BOCPD | Rt, metrics, changepoints |
| 5 | **ML Forecasting** | XGBoost ensemble, SHAP TreeExplainer | 14-day forecast + drivers |
| 6 | **SitRep Generator** | Template synthesis, alert classification | Executive report |

---

## 🧮 Statistical Methods

### 1. SEIR Compartmental Model
Solves the Susceptible-Exposed-Infectious-Recovered system of ODEs:

$$\frac{dS}{dt} = -\beta \frac{SI}{N}, \quad \frac{dE}{dt} = \beta \frac{SI}{N} - \sigma E, \quad \frac{dI}{dt} = \sigma E - \gamma I, \quad \frac{dR}{dt} = \gamma I$$

**Implementation:** `scipy.integrate.solve_ivp` with RK45 adaptive stepping. Parameter fitting via Nelder-Mead optimization of RMSE.

### 2. Bayesian Rt Estimation (Cori et al., 2013)
The WHO/CDC gold standard for real-time effective reproduction number:

$$R_t \mid \text{data} \sim \text{Gamma}(a + \sum I_t, \; (1/b + \sum \Lambda_t)^{-1})$$

where $\Lambda_t = \sum_{s=1}^{T} I_{t-s} \cdot w_s$ is the total infectiousness and $w_s$ is the discretized serial interval distribution.

**Reference:** Cori A, Ferguson NM, Fraser C, Cauchemez S. (2013) *American Journal of Epidemiology*, 178(9):1505-1512.

### 3. Wilson Score Confidence Intervals
For Case Fatality Rate and Attack Rate:

$$\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

**Why Wilson, not Wald?** The Wald interval ($\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$) can produce negative lower bounds and has poor coverage for small $n$ or extreme $p$.

### 4. Exact Poisson CI for Incidence Rates
Using the chi-squared inversion method:

$$\text{Lower} = \frac{\chi^2_{\alpha/2, 2k}}{2n}, \quad \text{Upper} = \frac{\chi^2_{1-\alpha/2, 2(k+1)}}{2n}$$

### 5. Bayesian Online Changepoint Detection (BOCPD)
Detects structural breaks in epidemic time series using the Adams & MacKay (2007) algorithm with a Student-t predictive distribution.

### 6. XGBoost + SHAP Ensemble Forecasting
- **Features:** 14 lag values, 7/14-day rolling statistics, day-over-day change, week growth ratio, calendar features
- **Ensemble:** Inverse-RMSE weighted average when multiple models available
- **Explainability:** SHAP TreeExplainer for exact Shapley values

---

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/epiagent.git
cd epiagent
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install numpy scipy pandas pydantic xgboost scikit-learn shap plotly requests
```

### 2. Run the Demo (No API Key Required)
```bash
python run_demo.py
```
This runs the complete 6-step pipeline in ~3 seconds and generates an interactive dashboard.

### 3. Run with ADK Agents (Requires Gemini API Key)
```bash
# Get free key at https://aistudio.google.com/apikey
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

pip install google-adk "mcp[cli]"
adk run epiagent
```

### 4. Run Tests
```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
epiagent/
├── epiagent/                    # Main package
│   ├── agent.py                 # ADK root_agent (SequentialAgent)
│   ├── agents/
│   │   ├── tools.py             # 10 FunctionTool wrappers
│   │   └── __init__.py
│   ├── engines/                 # Deterministic computation engines
│   │   ├── seir_model.py        # SEIR compartmental model
│   │   ├── rt_estimation.py     # Bayesian Rt (Cori et al.)
│   │   ├── epi_metrics.py       # CFR, incidence, doubling time
│   │   ├── changepoint_detector.py  # BOCPD
│   │   ├── ml_forecaster.py     # XGBoost + Prophet ensemble
│   │   └── explainability.py    # SHAP TreeExplainer
│   ├── guardrails/
│   │   └── security.py          # HIPAA PII detection/stripping
│   ├── validators/
│   │   └── epi_validator.py     # 8-point data quality checks
│   ├── schemas/
│   │   └── surveillance.py      # Pydantic v2 data models
│   └── dashboard/
│       └── generator.py         # 7-panel Plotly dashboard
├── mcp_server/                  # MCP Data Server
│   ├── server.py                # MCP tools (stdio transport)
│   └── data_sources/
│       ├── synthetic.py         # SEIR-based synthetic data
│       └── cdc_fluview.py       # CDC Delphi Epidata API
├── tests/                       # 55 unit tests
│   ├── test_seir_model.py
│   ├── test_rt_estimation.py
│   ├── test_epi_validator.py
│   └── test_security.py
├── run_demo.py                  # Full pipeline demo
├── pyproject.toml               # Dependencies
└── .env.example                 # API key template
```

---

## 🔒 Security

EpiAgent implements the **HIPAA Safe Harbor** de-identification method (45 CFR § 164.514(b)(2)):

- Scans for all **18 HIPAA identifier types** (names, SSN, email, phone, addresses, etc.)
- **SHA-256 data provenance hashing** for complete audit trails
- PII is stripped *before* data enters the analysis pipeline
- All operations logged for compliance documentation

---

## 🏆 Competition Differentiators

| Feature | Why It Matters |
|---------|---------------|
| **Deterministic math engines** | Zero hallucinated statistics — every number is verifiable |
| **Wilson score CIs** | Methodologically superior to Wald intervals (most implementations use Wald) |
| **Cori et al. Rt** | The actual WHO/CDC standard, not a simplified approximation |
| **HIPAA compliance** | Real-world deployment consideration, not just a toy project |
| **SHAP explainability** | Transparent AI — public health decisions need to be explainable |
| **55 passing tests** | Research-grade software engineering, not just a notebook |
| **3.1s full pipeline** | Production-ready performance |

---

## 📚 References

1. Cori A, Ferguson NM, Fraser C, Cauchemez S. (2013) "A New Framework and Software to Estimate Time-Varying Reproduction Numbers During Epidemics." *American Journal of Epidemiology*, 178(9):1505-1512.
2. Adams RP, MacKay DJC. (2007) "Bayesian Online Changepoint Detection." arXiv:0710.3742.
3. Wilson EB. (1927) "Probable Inference, the Law of Succession, and Statistical Inference." *JASA*, 22(158):209-212.
4. Lundberg SM, Lee SI. (2017) "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
5. Kermack WO, McKendrick AG. (1927) "A Contribution to the Mathematical Theory of Epidemics." *Proc. Royal Society A*, 115(772):700-721.
6. Chen T, Guestrin C. (2016) "XGBoost: A Scalable Tree Boosting System." *KDD*.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ for public health by a Statistics graduate aspiring to advance Biostatistics through AI.*
