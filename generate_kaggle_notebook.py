"""EpiAgent Kaggle Notebook Generator.

Generates a self-contained Jupyter notebook (.ipynb) for Kaggle submission
by consolidating all EpiAgent modules into a single executable notebook.

The notebook includes:
1. All engine code inlined
2. Full pipeline execution
3. Interactive visualizations
4. Methodology documentation
5. SHAP analysis plots
"""

import json
from pathlib import Path


def create_cell(cell_type: str, source: str, metadata: dict = None) -> dict:
    """Create a Jupyter notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source.split("\n"),
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def build_notebook() -> dict:
    """Build the complete Kaggle notebook."""

    cells = []

    # ===================================================================
    # Title & Introduction
    # ===================================================================
    cells.append(create_cell("markdown", """# 🦠 EpiAgent: Secure Multi-Agent Epidemic Surveillance & Predictive Modeling System

**Google/Kaggle 5-Day AI Agents Intensive Course — "Agents for Good" (Healthcare)**

🎬 **[Watch the Complete 7-Minute Video Presentation on YouTube](https://youtu.be/yB7ILJe-SAc)**  
📦 **[GitHub Repository (All 70 Unit Tests & Code)](https://github.com/sujon-stat/EpiAgent)**  
✍️ **Author:** Sujon Mia (Department of Statistics, Jagannath University, Bangladesh)

---

## Executive Summary

EpiAgent is an autonomous public health intelligence pipeline that:
1. **Streams** surveillance data from synthetic SEIR models or CDC APIs
2. **Secures** data with HIPAA-compliant PII detection (18 identifier types)
3. **Validates** data quality with 8-point epidemiological plausibility checks
4. **Analyzes** epidemics using SEIR models, Bayesian Rt estimation, and BOCPD changepoint detection
5. **Forecasts** case counts using XGBoost ensemble with SHAP explainability
6. **Reports** executive situation reports with alert level classification

### Key Innovation: "LLM Decides, Math Computes, LLM Interprets"
Every numerical result comes from a deterministic `FunctionTool` — never from LLM generation. The agents decide *when* to call tools and *how to interpret* results, but the math is always exact and reproducible.

### Architecture
```
Data Agent → Security Agent → Validator Agent → Analysis Agent → ML Agent → SitRep Agent
                                                                                    ↓
                                                                        Interactive Dashboard
                                                                        Executive SitRep
```
"""))

    # ===================================================================
    # Install Dependencies
    # ===================================================================
    cells.append(create_cell("code", """# Install required packages
!pip install -q numpy scipy pandas pydantic xgboost scikit-learn shap plotly requests google-adk "mcp[cli]"

import warnings
warnings.filterwarnings('ignore')

import json
import logging
import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.stats import gamma as gamma_dist

print("✅ All packages imported successfully")
print(f"NumPy: {np.__version__}, Pandas: {pd.__version__}, SciPy: {stats.scipy.__version__}")
"""))

    # ===================================================================
    # Methodology Section
    # ===================================================================
    cells.append(create_cell("markdown", """## 📐 Statistical Methods

### 1. SEIR Compartmental Model
The SEIR model divides the population into four compartments:

$$\\frac{dS}{dt} = -\\beta \\frac{SI}{N}, \\quad \\frac{dE}{dt} = \\beta \\frac{SI}{N} - \\sigma E, \\quad \\frac{dI}{dt} = \\sigma E - \\gamma I, \\quad \\frac{dR}{dt} = \\gamma I$$

Where: $\\beta = R_0 \\cdot \\gamma$, $\\sigma = 1/\\text{latent period}$, $\\gamma = 1/\\text{infectious period}$.

### 2. Bayesian Rt Estimation (Cori et al., 2013)
The instantaneous reproduction number using Gamma conjugate prior:

$$R_t \\mid \\text{data} \\sim \\text{Gamma}\\left(a + \\sum_{s \\in \\tau} I_s, \\; \\left(\\frac{1}{b} + \\sum_{s \\in \\tau} \\Lambda_s\\right)^{-1}\\right)$$

### 3. Wilson Score Confidence Interval
For CFR and attack rate (superior to Wald interval):

$$\\text{CI} = \\frac{\\hat{p} + \\frac{z^2}{2n} \\pm z\\sqrt{\\frac{\\hat{p}(1-\\hat{p})}{n} + \\frac{z^2}{4n^2}}}{1 + \\frac{z^2}{n}}$$

### 4. BOCPD Changepoint Detection (Adams & MacKay, 2007)
Bayesian Online Changepoint Detection with Student-t predictive distribution.

### 5. XGBoost + SHAP Ensemble Forecasting
Feature-engineered gradient boosted trees with SHAP TreeExplainer for post-hoc explainability.
"""))

    # ===================================================================
    # Read and inline the engine code files
    # ===================================================================
    engine_files = {
        "SEIR Model": "epiagent/engines/seir_model.py",
        "Rt Estimation": "epiagent/engines/rt_estimation.py",
        "Epi Metrics": "epiagent/engines/epi_metrics.py",
        "Changepoint Detector": "epiagent/engines/changepoint_detector.py",
        "ML Forecaster": "epiagent/engines/ml_forecaster.py",
        "SHAP Explainability": "epiagent/engines/explainability.py",
    }

    support_files = {
        "Data Validator": "epiagent/validators/epi_validator.py",
        "Security Guardrails": "epiagent/guardrails/security.py",
        "Data Schemas": "epiagent/schemas/surveillance.py",
        "Synthetic Data Generator": "mcp_server/data_sources/synthetic.py",
        "CDC FluView Client": "mcp_server/data_sources/cdc_fluview.py",
        "FunctionTool Wrappers": "epiagent/agents/tools.py",
        "Dashboard Generator": "epiagent/dashboard/generator.py",
    }

    cells.append(create_cell("markdown", "## 🔧 Core Engine Code\nThe following cells contain the deterministic computation engines."))

    for name, filepath in {**engine_files, **support_files}.items():
        fp = Path(filepath)
        if fp.exists():
            code = fp.read_text(encoding="utf-8")
            cells.append(create_cell("markdown", f"### {name}\n`{filepath}`"))
            cells.append(create_cell("code", code))

    # ===================================================================
    # Agent Definition
    # ===================================================================
    agent_file = Path("epiagent/agent.py")
    if agent_file.exists():
        cells.append(create_cell("markdown", "## 🤖 Multi-Agent System (ADK 2.3.0)\n6 specialized LlmAgents orchestrated by a SequentialAgent pipeline."))
        cells.append(create_cell("code", agent_file.read_text(encoding="utf-8")))

    # ===================================================================
    # Pipeline Execution
    # ===================================================================
    cells.append(create_cell("markdown", """## 🚀 Full Pipeline Execution

Running the complete 6-step pipeline:
1. Data Retrieval (synthetic COVID-19)
2. Security Audit (HIPAA)
3. Data Quality Validation
4. Epidemiological Analysis (SEIR, Rt, Metrics, Changepoints)
5. ML Forecasting (XGBoost + SHAP)
6. Dashboard Generation
"""))

    demo_file = Path("run_demo.py")
    if demo_file.exists():
        demo_code = demo_file.read_text(encoding="utf-8")
        # Remove the sys.path manipulation for Kaggle
        demo_code = demo_code.replace(
            'sys.path.insert(0, str(project_root))',
            '# sys.path already set in Kaggle'
        )
        cells.append(create_cell("code", demo_code))

    # ===================================================================
    # Multi-Pathogen Comparison
    # ===================================================================
    cells.append(create_cell("markdown", """## 📊 Multi-Pathogen Scenario Comparison

Demonstrating generalizability across three pathogen profiles:
- **COVID-19**: R₀=2.5, serial interval 4.7 days, CFR≈1.5%
- **Influenza**: R₀=1.3, serial interval 2.6 days, CFR≈0.1%
- **Measles**: R₀=12.0, serial interval 11.5 days, CFR≈0.2%
"""))

    multi_file = Path("run_multi_pathogen.py")
    if multi_file.exists():
        multi_code = multi_file.read_text(encoding="utf-8")
        multi_code = multi_code.replace(
            'sys.path.insert(0, str(project_root))',
            '# sys.path already set in Kaggle'
        )
        cells.append(create_cell("code", multi_code))

    # ===================================================================
    # Conclusion
    # ===================================================================
    cells.append(create_cell("markdown", """## 🏁 Conclusion

### What EpiAgent Demonstrates

1. **Responsible AI for Healthcare**: HIPAA-compliant PII scanning, deterministic computation, and full audit trails ensure AI systems can be trusted in public health contexts.

2. **Multi-Agent Orchestration**: Google ADK 2.3.0's SequentialAgent pattern provides a clean, maintainable pipeline architecture where each agent has a single responsibility.

3. **Rigorous Statistical Methods**: Using Wilson score intervals (not Wald), exact Poisson CIs, and the Cori et al. (2013) Bayesian Rt method — the actual WHO/CDC standard.

4. **Explainable AI**: SHAP TreeExplainer provides post-hoc explanations for every forecast, critical for public health decision-making.

5. **Production-Ready Engineering**: 55 passing unit tests, 3.1-second pipeline execution, and modular architecture ready for deployment.

### References

1. Cori A, Ferguson NM, Fraser C, Cauchemez S. (2013) "A New Framework and Software to Estimate Time-Varying Reproduction Numbers During Epidemics." *Am J Epidemiol*, 178(9):1505-1512.
2. Adams RP, MacKay DJC. (2007) "Bayesian Online Changepoint Detection." arXiv:0710.3742.
3. Wilson EB. (1927) "Probable Inference, the Law of Succession, and Statistical Inference." *JASA*, 22(158):209-212.
4. Lundberg SM, Lee SI. (2017) "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
5. Kermack WO, McKendrick AG. (1927) "A Contribution to the Mathematical Theory of Epidemics." *Proc Royal Soc A*, 115(772):700-721.
6. Chen T, Guestrin C. (2016) "XGBoost: A Scalable Tree Boosting System." *KDD*.

---

*Built with ❤️ for public health.*
"""))

    # Build notebook JSON
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.0",
            },
        },
        "cells": cells,
    }

    return notebook


def main():
    notebook = build_notebook()
    output_path = Path("epiagent_kaggle_notebook.ipynb")
    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Kaggle notebook generated: {output_path.resolve()}")
    print(f"File size: {output_path.stat().st_size / 1024:.0f} KB")
    print(f"Cells: {len(notebook['cells'])}")


if __name__ == "__main__":
    main()
