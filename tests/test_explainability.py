import numpy as np
import pytest
from epiagent.engines.explainability import explain_xgboost_forecast

class TestExplainability:
    def test_explain_xgboost_forecast(self):
        # Generate dummy data
        t = np.arange(200)
        cases = 100 + 50 * np.sin(t * 2 * np.pi / 30) + np.random.normal(0, 5, 200)
        
        explanation, metrics = explain_xgboost_forecast(cases, top_k=5)
        
        assert metrics["model"] == "XGBoost"
        assert len(explanation.top_drivers) == 5
        assert "feature" in explanation.top_drivers[0]
        assert "importance" in explanation.top_drivers[0]
        assert explanation.base_value is not None
        assert isinstance(explanation.narrative(), str)
        assert len(explanation.narrative()) > 0
        assert "global_importance" in explanation.to_dict()
