"""Tests for demand sensing module."""

import pandas as pd
import pytest

from app.sensing.signals import DemandSignal, SignalProcessor


@pytest.fixture
def processor():
    return SignalProcessor(config={
        "pos_weight": 0.6,
        "social_weight": 0.25,
        "weather_weight": 0.15,
        "decay_half_life_days": 7,
        "spike_threshold_sigma": 2.0,
        "sensing_horizon_days": 14,
    })


@pytest.fixture
def sample_signals():
    return [
        DemandSignal(source="pos", timestamp="day_0", product_id="SKU-001", signal_value=120.0, confidence=0.9),
        DemandSignal(source="social", timestamp="day_0", product_id="SKU-001", signal_value=130.0, confidence=0.8),
        DemandSignal(source="pos", timestamp="day_1", product_id="SKU-001", signal_value=110.0, confidence=0.9),
        DemandSignal(source="social", timestamp="day_1", product_id="SKU-001", signal_value=125.0, confidence=0.85),
        DemandSignal(source="pos", timestamp="day_2", product_id="SKU-001", signal_value=115.0, confidence=0.9),
        DemandSignal(source="pos", timestamp="day_1", product_id="SKU-002", signal_value=80.0, confidence=0.95),
        DemandSignal(source="weather", timestamp="day_3", product_id="SKU-001", signal_value=500.0, confidence=0.7),  # spike
    ]


class TestSignalProcessor:
    def test_generate_synthetic_signals(self, processor):
        signals = processor.generate_synthetic_signals(["SKU-001", "SKU-002"], n_days=7)
        assert len(signals) > 0
        assert all(isinstance(s, DemandSignal) for s in signals)

    def test_compute_signal_weight(self, processor, sample_signals):
        weight = processor.compute_signal_weight(sample_signals[0], horizon_days=7)
        assert 0 < weight <= 1.0

    def test_compute_signal_weight_varies_by_source(self, processor):
        pos = DemandSignal(source="pos", timestamp="", product_id="", signal_value=100, confidence=1.0)
        social = DemandSignal(source="social", timestamp="", product_id="", signal_value=100, confidence=1.0)
        w_pos = processor.compute_signal_weight(pos, 7)
        w_social = processor.compute_signal_weight(social, 7)
        assert w_pos > w_social  # POS has higher weight

    def test_compute_sensing_adjustment(self, processor, sample_signals):
        base = pd.DataFrame({
            "product_id": ["SKU-001", "SKU-002"],
            "period": [1, 1],
            "forecast": [100.0, 90.0],
        })
        result = processor.compute_sensing_adjustment(base, sample_signals)
        assert "adjusted_forecast" in result.columns
        assert "adjustment_pct" in result.columns
        assert len(result) == 2

    def test_adjustment_no_signals(self, processor):
        base = pd.DataFrame({"product_id": ["SKU-001"], "period": [1], "forecast": [100.0]})
        result = processor.compute_sensing_adjustment(base, [])
        assert result["adjusted_forecast"].iloc[0] == 100.0
        assert result["adjustment_pct"].iloc[0] == 0.0

    def test_detect_spikes(self, processor, sample_signals):
        spikes = processor.detect_spikes(sample_signals)
        # The 500.0 signal should be detected as a spike
        assert len(spikes) >= 1
        spike_values = [s.magnitude for s in spikes]
        assert any(v >= 400 for v in spike_values)

    def test_detect_spikes_empty(self, processor):
        spikes = processor.detect_spikes([])
        assert spikes == []

    def test_synthetic_signals_deterministic(self, processor):
        s1 = processor.generate_synthetic_signals(["A", "B"], seed=42)
        s2 = processor.generate_synthetic_signals(["A", "B"], seed=42)
        assert len(s1) == len(s2)
        assert all(a.signal_value == b.signal_value for a, b in zip(s1, s2))
