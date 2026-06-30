# Building EpiAgent: Autonomous Multi-Agent Epidemic Surveillance & Real-Time Intelligence

*A comprehensive deep dive into combining Google's Agent Development Kit (ADK), deterministic Bayesian statistics, and explainable machine learning for public health — built for the Google & Kaggle 5-Day AI Agents Intensive Course ("Agents for Good" Track).*

---

**Author:** Sujon Mia  
**Affiliation:** Department of Statistics, Jagannath University, Dhaka, Bangladesh  
**Email:** [sujonsgc@gmail.com](mailto:sujonsgc@gmail.com)  
**GitHub Repository:** [https://github.com/sujon-stat/EpiAgent](https://github.com/sujon-stat/EpiAgent)  
**YouTube Video Presentation:** [https://youtu.be/yB7ILJe-SAc](https://youtu.be/yB7ILJe-SAc)  

---

## 1. The Problem: Can AI Help Us Detect & Respond to Outbreaks Faster?

During the COVID-19 pandemic, public health agencies worldwide faced a critical bottleneck: overwhelmed epidemiological surveillance systems. Raw data from clinical networks arrived in disjointed streams, manual cleaning and statistical modeling took days or weeks, and policy decision-makers lacked real-time, explainable forecasts. 

When outbreak signals are delayed, interventions like targeted lockdowns, hospital resource allocation, and vaccination distributions arrive too late. **In epidemiology, latency costs lives.**

While modern Large Language Models (LLMs) excel at natural language understanding and orchestration, asking a standalone LLM chatbot to analyze epidemiological data introduces a fatal flaw: **hallucination**. An LLM asked to compute a Case Fatality Rate (CFR) or a 95% confidence interval may generate plausible-sounding but mathematically incorrect numbers. In public health policy, a hallucinated confidence interval is unacceptable.

**The Research Question:** *How can we architect an autonomous AI agent system that leverages the reasoning capabilities of modern LLMs while guaranteeing 100% deterministic mathematical accuracy and strict patient privacy?*

---

## 2. The Solution: The EpiAgent Architecture

**EpiAgent** is a multi-agent public health surveillance system built on Google's **Agent Development Kit (ADK 2.3.0)**. Instead of relying on a single monolithic chatbot, EpiAgent deploys a sequential pipeline of six domain-specialized agents:

```
[Surveillance Data]
       │
       ▼
┌──────────────────┐
│  1. Data Agent   │ ── Ingests multi-pathogen surveillance feeds (synthetic & CDC ILINet)
└────────┬─────────┘
       │
       ▼
┌──────────────────┐
│ 2. Security Agent│ ── Audits & scrubs 18 HIPAA PII identifiers; attaches SHA-256 hash
└────────┬─────────┘
       │
       ▼
┌──────────────────┐
│3. Validator Agent│ ── Enforces epidemiological schema & data quality validation (>90% threshold)
└────────┬─────────┘
       │
       ▼
┌──────────────────┐
│4. Analysis Agent │ ── Computes Bayesian Rt (Cori method), SEIR ODEs, & Wilson Score intervals
└────────┬─────────┘
       │
       ▼
┌──────────────────┐
│   5. ML Agent    │ ── Fits XGBoost forecaster & extracts exact SHAP feature contributions
└────────┬─────────┘
       │
       ▼
┌──────────────────┐
│  6. SitRep Agent │ ── Synthesizes findings into interactive HTML dashboards & executive briefs
└──────────────────┘
```

---

## 3. Core Design Philosophy: "LLM Decides, Math Computes, LLM Interprets"

The fundamental architectural principle of EpiAgent is separation of concerns:

1. **LLM Orchestration & Reasoning:** The LLM manages pipeline execution, evaluates error states, decides which analytical tools to invoke, and translates complex statistical outputs into plain-language executive summaries.
2. **Deterministic Computational Engines:** Every calculation—whether solving differential equations, calculating statistical intervals, or computing gradient boosted decision trees—is executed inside isolated Python functions exposed to the agents via strict `FunctionTool` wrappers.

```python
# Example: FunctionTool wrapper guaranteeing exact statistical computation
def compute_cfr(deaths: int, cases: int, confidence: float = 0.95) -> MetricResult:
    """Computes Case Fatality Rate using exact Wilson score interval."""
    if cases == 0:
        return MetricResult(value=0.0, lower_ci=0.0, upper_ci=0.0, method="Wilson score interval")
    p_hat = deaths / cases
    z = 1.96  # 95% confidence level
    denominator = 1 + (z**2 / cases)
    center = p_hat + (z**2 / (2 * cases))
    margin = z * math.sqrt((p_hat * (1 - p_hat) / cases) + (z**2 / (4 * (cases**2))))
    return MetricResult(
        value=p_hat,
        lower_ci=(center - margin) / denominator,
        upper_ci=(center + margin) / denominator,
        method="Wilson score interval"
    )
```

---

## 4. Mathematical & Epidemiological Foundations

### A. Wilson Score Interval vs. Wald Approximation
Most basic analytics software uses the normal approximation (Wald interval: $p \pm z\sqrt{p(1-p)/n}$) for proportions. However, when sample sizes are small or proportions are near 0 or 1 (such as disease fatality rates where CFR $\approx 0.1\%$), the Wald interval frequently yields negative boundaries or severe under-coverage. EpiAgent strictly enforces the **Wilson score interval**, ensuring statistically valid confidence boundaries under all empirical conditions.

### B. Bayesian Real-Time Effective Reproduction Number ($R_t$)
To determine whether an outbreak is expanding or shrinking, EpiAgent implements the **Cori et al. (2013)** methodology—the standard utilized by the World Health Organization (WHO) and CDC.
* **Model:** Uses a Gamma prior conditioned on sliding incidence windows weighted by the pathogen's serial interval distribution:
  $$R_t \sim \text{Gamma}\left(a_{\text{prior}} + \sum_{s=1}^{\tau} I_{t-s+1}, \ \frac{1}{\frac{1}{b_{\text{prior}}} + \sum_{s=1}^{\tau} \Lambda_{t-s+1}}\right)$$
  where $\Lambda_t = \sum_{k=1}^t I_{t-k} w_k$ represents total infectiousness.
* **Phase Classification:** The system automatically categorizes outbreak states:
  * **Growing:** 95% Credible Interval lower bound $> 1.0$
  * **Declining:** 95% Credible Interval upper bound $< 1.0$
  * **Stable:** Credible Interval spans $1.0$

### C. SEIR Compartmental ODE Modeling
To model transmission dynamics, the system solves the nonlinear Susceptible-Exposed-Infectious-Recovered differential equations using `scipy.integrate.solve_ivp` (Runge-Kutta 45):
$$\frac{dS}{dt} = -\beta \frac{S \cdot I}{N}, \quad \frac{dE}{dt} = \beta \frac{S \cdot I}{N} - \sigma E, \quad \frac{dI}{dt} = \sigma E - \gamma I, \quad \frac{dR}{dt} = \gamma I$$
The engine checks strict mathematical conservation ($S + E + I + R = N$ within $10^{-6}$ tolerance) at every timestep.

### D. Transparent Machine Learning via XGBoost + SHAP
Machine learning predictions in healthcare must be explainable. EpiAgent fits an **XGBoost time-series regressor** on lag features, rolling averages, and day-of-week indicators. It then computes exact **SHAP (SHapley Additive exPlanations)** tree values. When the model forecasts a drop or spike in cases, it quantifies the exact contribution of factors such as weekend reporting delays or rolling momentum.

---

## 5. Enterprise-Grade Security & HIPAA Guardrails

Public health surveillance pipelines handle sensitive clinical data. EpiAgent integrates an automated **Security Agent** before any analytical modeling takes place:
* **Automated PII Scrubbing:** Scans and redacts all 18 HIPAA identifier types (names, emails, Social Security Numbers, telephone numbers, IP addresses, exact birth dates, URLs, and medical record numbers) using strict regex pattern matching.
* **Cryptographic Data Provenance:** Generates a unique SHA-256 data hash for all clean surveillance datasets. If a single record is modified downstream, the audit hash alerts administrators to tampering or corruption.

---

## 6. Benchmark Performance & Multi-Pathogen Evaluation

EpiAgent executes the entire 6-agent sequential pipeline locally in **under 4 seconds**, producing standalone, responsive HTML interactive dashboards (`Plotly`).

### Multi-Pathogen Comparative Summary

| Pathogen Profile | Basic $R_0$ | Total Simulated Cases | Case Fatality Rate | Current $R_t$ | Epidemic Phase | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 2.5 | 852,670 | 1.49% | 0.68 | Stable / Declining | ~3.3s |
| **Influenza** | 1.3 | 33,982 | 0.10% | 1.17 | Growing | ~1.1s |
| **Measles** | 12.0 | 1,003,871 | 0.20% | 4.99 | Stable (High Endemic) | ~1.2s |

The system perfectly adapts to varying pathogen dynamics: capturing the explosive transmission of measles ($R_0=12$), the moderate wave seasonality of influenza, and the post-peak reproductive slowdown of COVID-19.

---

## 7. Verification: 100% Automated Test Suite (70/70 Passing)

Software reliability is critical in healthcare. EpiAgent includes a complete suite of **70 automated pytest unit tests** covering all pipeline layers:
* **Mathematical Conservation:** Verifies exact population conservation across SEIR simulations.
* **Statistical Bounds:** Validates Wilson score interval boundaries and Bayesian Gamma credible bounds.
* **Security & Compliance:** Verifies detection and redacting of all 18 HIPAA PII categories.
* **Schema Integrity:** Confirms strict rejection of invalid surveillance records or missing values.

```bash
$ pytest tests/ -v
============================= 70 passed in 18.03s =============================
```

---

## 8. Interactive Dashboards & Video Presentation

EpiAgent automatically builds a standalone interactive HTML dashboard (`epiagent_dashboard.html`) featuring:
1. **Epidemic Curve & 14-Day XGBoost Forecast** (with 95% confidence bands).
2. **Real-Time Bayesian $R_t$ Trajectory** (with WHO threshold markings).
3. **SHAP Feature Contribution Chart** (explaining model prediction drivers).
4. **SEIR Compartmental Simulation Panel**.

Furthermore, a complete narrated video walkthrough (`EpiAgent_Final_Walkthrough.mp4`) has been produced and pushed to the repository, explaining the project architecture, statistical mechanics, and competition objectives.

---

## 9. Key Lessons Learned & Future Directions

1. **Specialization Beats Monoliths:** Dividing the workflow into six focused agents dramatically improved reliability compared to asking a single LLM prompt to execute analytics and report generation.
2. **Deterministic Tools are Essential:** Combining LLM reasoning with exact Python mathematical tools (`FunctionTool`) completely prevents computational hallucinations.
3. **Rigorous Guardrails Build Trust:** Integrating automated HIPAA scrubbing and SHA-256 provenance auditing turns an experimental AI demo into an enterprise-ready healthcare tool.

### Future Work
* **Real-Time API Integration:** Connecting directly to live CDC FluView and WHO Global Health Observatory APIs.
* **Deep Learning Hybridization:** Integrating Temporal Fusion Transformers (TFT) alongside XGBoost for long-horizon seasonal forecasting.
* **Cloud Deployment:** Packaging the ADK sequential orchestration engine as a serverless microservice on Google Cloud Run.

---

*Thank you for reviewing EpiAgent! Please visit the [GitHub Repository](https://github.com/sujon-stat/EpiAgent) or check out our Kaggle notebook submission.*
