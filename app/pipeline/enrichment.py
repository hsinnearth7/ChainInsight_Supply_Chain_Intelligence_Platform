"""Shared enrichment functions for pipeline analyzers."""

import numpy as np
import pandas as pd

from app.config import DSI_SENTINEL


def enrich_base(df: pd.DataFrame) -> pd.DataFrame:
    """Shared enrichment: DSI, Stock_Coverage_Ratio, Demand_Intensity.

    All analyzers call this then add their own specific columns.
    """
    df = df.copy()
    df["DSI"] = np.where(
        df["Daily_Demand_Est"] > 0,
        df["Current_Stock"] / df["Daily_Demand_Est"],
        DSI_SENTINEL,
    )
    df["Stock_Coverage_Ratio"] = np.where(
        df["Safety_Stock_Target"] > 0,
        df["Current_Stock"] / df["Safety_Stock_Target"],
        0,
    )
    df["Demand_Intensity"] = df["Daily_Demand_Est"] * df["Unit_Cost"]
    return df
