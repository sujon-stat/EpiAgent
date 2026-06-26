# Building EpiAgent: How I Used Multi-Agent AI to Create a Real-Time Epidemic Surveillance System

*A deep dive into combining Google ADK, Bayesian statistics, and explainable AI for public health — built for the Kaggle "Agents for Good" competition.*

---

## The Problem: Can AI Help Us Detect Outbreaks Faster?

During COVID-19, we saw firsthand how delayed surveillance data and slow analysis can cost lives. Public health agencies need systems that can:
- Process surveillance data in real-time
- Detect outbreak signals automatically
- Forecast case trajectories with uncertainty quantification
- Explain predictions to non-technical decision-makers
- Do all of this without compromising patient privacy

I asked myself: **what if a team of AI agents could do this autonomously?**

## The Solution: EpiAgent

EpiAgent is a 6-agent pipeline built on Google's Agent Development Kit (ADK 2.3.0):

```
Data Agent → Security Agent → Validator Agent → Analysis Agent → ML Agent → SitRep Agent
```

Each agent has one job. Each uses deterministic mathematical tools. The LLM decides *when* to act and *how to communicate* — but it never makes up numbers.

### The Key Design Decision: "LLM Decides, Math Computes, LLM Interprets"

This might be the most important architectural pattern in the entire project. Here's why:

If you ask an LLM "what's the case fatality rate for 150 deaths out of 10,000 cases?", it might say "1.5%" — and it'd be right. But what's the confidence interval? The LLM might hallucinate one. In epidemiology, a hallucinated confidence interval could lead to wrong policy decisions.

Instead, EpiAgent uses `FunctionTool` wrappers that call exact mathematical implementations:

```python
# The LLM calls this FunctionTool — the math is always correct
def compute_cfr(deaths: int, cases: int) -> MetricResult:
    # Wilson score interval (not Wald!)
    p_hat = deaths / cases
    z = 1.96
    denominator = 1 + z**2 / cases
    center = p_hat + z**2 / (2 * cases)
    margin = z * sqrt(p_hat * (1 - p_hat) / cases + z**2 / (4 * cases**2))
    return MetricResult(
        value=p_hat,
        lower_ci=(center - margin) / denominator,
        upper_ci=(center + margin) / denominator,
        method="Wilson score interval"
    )
```

## Statistical Methods: Why These Matter

### Why Wilson Score, Not Wald?

Most introductory statistics courses teach the Wald interval: p̂ ± z√(p̂(1-p̂)/n). It's simple, but it's *wrong* for small samples and extreme proportions. The Wilson score interval has proper coverage even when p is near 0 or 1 — exactly the scenario we face with case fatality rates for diseases like influenza (CFR ≈ 0.1%).

### Bayesian Rt: The WHO/CDC Standard

The effective reproduction number R_t tells us: "on average, how many people does each infected person infect *right now*?" When Rt > 1, the epidemic is growing. When Rt < 1, it's declining.

I implemented the Cori et al. (2013) method — the same algorithm used by WHO and CDC. It uses a Gamma conjugate prior:

```
Posterior: Rt ~ Gamma(a + Σ cases, 1/(1/b + Σ infectiousness))
```

This gives us not just a point estimate, but a full posterior distribution with credible intervals. If the 95% credible interval is entirely above 1, we're *confident* the epidemic is growing.

### SHAP: Making ML Forecasts Transparent

When our XGBoost model predicts "cases will increase next week," public health officials need to know *why*. SHAP (SHapley Additive exPlanations) decomposes each prediction into feature contributions:

> "The model predicts 500 cases tomorrow primarily because: (1) yesterday saw 450 cases (+200 impact), (2) the 7-day average is rising (+150 impact), and (3) it's Monday, historically a catch-up day for weekend reporting (+50 impact)."

That last point — weekend reporting effects — is a real epidemiological phenomenon. The fact that SHAP independently identifies it validates our feature engineering.

## Results

### Full Pipeline: 3.1 Seconds

| Step | Result |
|------|--------|
| Data Retrieval | 181 records, 857,217 cases |
| Security Audit | HIPAA Clean ✅ |
| Data Quality | 99.6% score |
| Rt Estimation | 0.692 (stable/declining) |
| ML Forecast | RMSE = 50.69 |
| Dashboard | 78 KB interactive HTML |

### Multi-Pathogen Generalization

| Pathogen | R₀ | Cases | CFR | Current Rt |
|----------|-----|-------|-----|------------|
| COVID-19 | 2.5 | 857K | 1.48% | 0.69 |
| Influenza | 1.3 | 34K | 0.10% | 1.18 |
| Measles | 12.0 | 1M | 0.20% | 4.99 |

The system correctly captures the epidemiological characteristics of each pathogen: measles spreads explosively (Rt near 5), influenza is more moderate, and COVID-19 in our scenario has entered the declining phase.

### Test Coverage

55 unit tests covering:
- Conservation laws (S+E+I+R = N at every timestep)
- Subcritical R₀ produces no epidemic
- Wilson CI contains the true proportion
- PII detection catches all 18 HIPAA identifier types
- SHAP values sum to model output

## What I Learned

1. **Agent specialization > monolithic agents.** A single agent trying to do data retrieval + validation + analysis + reporting performs poorly. Six specialized agents, each with clear tools, work much better.

2. **Deterministic tools are non-negotiable for healthcare AI.** LLMs are great at orchestration and communication. They're terrible at math. Use `FunctionTool`.

3. **Wilson > Wald, always.** If you're computing confidence intervals for proportions, use Wilson score. It's one extra line of code and vastly better coverage properties.

4. **HIPAA compliance is a feature, not a burden.** Adding PII detection took one module (~200 lines) and dramatically increased the system's deployment readiness. Judges and reviewers notice this.

5. **Tests are your friend.** Having 55 passing tests gave me confidence to refactor aggressively. The conservation law test (S+E+I+R=N) caught three bugs during development.

## Try It Yourself

The complete code is in the Kaggle notebook. You can run the full pipeline without any API keys — all the deterministic engines work standalone.

To run with LLM agents, get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey) and use the ADK CLI:
```bash
adk run epiagent
```

## What's Next

- **LSTM neural forecaster** for longer-horizon predictions
- **Real CDC data integration** for live surveillance
- **Deployment to Cloud Run** for production use

---

*If you found this useful, please upvote the notebook! And if you're working on healthcare AI, let's connect — I'm a Statistics graduate applying to MS/PhD programs in Biostatistics and would love to collaborate.*

## References

1. Cori A, et al. (2013) Am J Epidemiol, 178(9):1505-1512.
2. Adams RP, MacKay DJC. (2007) arXiv:0710.3742.
3. Wilson EB. (1927) JASA, 22(158):209-212.
4. Lundberg SM, Lee SI. (2017) NeurIPS.
5. Kermack WO, McKendrick AG. (1927) Proc Royal Soc A, 115(772):700-721.
6. Chen T, Guestrin C. (2016) KDD.
