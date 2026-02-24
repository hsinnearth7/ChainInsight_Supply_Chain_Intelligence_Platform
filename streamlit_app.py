"""ChainInsight Live Dashboard — Streamlit interactive frontend."""

import sys
import time
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Ensure app module is importable
sys.path.insert(0, str(Path(__file__).parent))

from app.config import RAW_DIR, CLEAN_DIR, CHARTS_DIR, BASE_DIR
from app.db.models import init_db, SessionLocal, PipelineRun, AnalysisResult, InventorySnapshot
from app.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────
st.set_page_config(
    page_title="ChainInsight Live",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stMetric { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .pipeline-stage { padding: 8px 16px; border-radius: 6px; margin: 4px 0; }
    .stage-completed { background: #d4edda; color: #155724; }
    .stage-running { background: #fff3cd; color: #856404; }
    .stage-pending { background: #e2e3e5; color: #383d41; }
</style>
""", unsafe_allow_html=True)

# Initialize DB
init_db()


# ────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────
with st.sidebar:
    st.title("ChainInsight Live")
    st.caption("Real-time Supply Chain Analytics")
    st.divider()

    page = st.radio("Navigation", [
        "Dashboard",
        "Upload & Run Pipeline",
        "Statistical Analysis",
        "Supply Chain Optimization",
        "ML / AI Analysis",
        "RL Optimization",
        "Pipeline History",
    ], label_visibility="collapsed")

    st.divider()

    # Show latest run info
    db = SessionLocal()
    latest_run = db.query(PipelineRun).filter(
        PipelineRun.status == "completed"
    ).order_by(PipelineRun.completed_at.desc()).first()

    if latest_run:
        st.success(f"Latest: {latest_run.batch_id[:20]}...")
        st.caption(f"Completed: {latest_run.completed_at.strftime('%Y-%m-%d %H:%M') if latest_run.completed_at else 'N/A'}")
    else:
        st.info("No completed runs yet.\nUpload a CSV to get started!")
    db.close()


# ────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────
def get_latest_batch_id():
    db = SessionLocal()
    run = db.query(PipelineRun).filter(
        PipelineRun.status == "completed"
    ).order_by(PipelineRun.completed_at.desc()).first()
    db.close()
    return run.batch_id if run else None


def get_analysis_result(batch_id: str, analysis_type: str):
    db = SessionLocal()
    result = db.query(AnalysisResult).filter(
        AnalysisResult.batch_id == batch_id,
        AnalysisResult.analysis_type == analysis_type,
    ).first()
    db.close()
    return result


def get_clean_df(batch_id: str) -> pd.DataFrame:
    """Load cleaned data from DB snapshots."""
    db = SessionLocal()
    snapshots = db.query(InventorySnapshot).filter(
        InventorySnapshot.batch_id == batch_id
    ).all()
    db.close()
    if not snapshots:
        return pd.DataFrame()
    records = [{
        "Product_ID": s.product_id,
        "Category": s.category,
        "Unit_Cost": s.unit_cost,
        "Current_Stock": s.current_stock,
        "Daily_Demand_Est": s.daily_demand_est,
        "Safety_Stock_Target": s.safety_stock_target,
        "Vendor_Name": s.vendor_name,
        "Lead_Time_Days": s.lead_time_days,
        "Reorder_Point": s.reorder_point,
        "Stock_Status": s.stock_status,
        "Inventory_Value": s.inventory_value,
    } for s in snapshots]
    return pd.DataFrame(records)


def show_chart_images(batch_id: str, prefix: str = ""):
    """Display chart PNG images from the charts directory."""
    charts_dir = CHARTS_DIR / batch_id
    if not charts_dir.exists():
        st.warning("No charts found for this batch.")
        return
    charts = sorted(charts_dir.glob(f"{prefix}*.png"))
    if not charts:
        st.warning(f"No charts matching '{prefix}*' found.")
        return
    for chart in charts:
        st.image(str(chart), use_container_width=True, caption=chart.stem)


# ────────────────────────────────────────────
# PAGE: Dashboard (KPI Overview)
# ────────────────────────────────────────────
if page == "Dashboard":
    st.header("Dashboard — Key Performance Indicators")

    batch_id = get_latest_batch_id()
    if not batch_id:
        st.info("No data yet. Upload a CSV file to start the analysis pipeline.")
        st.stop()

    result = get_analysis_result(batch_id, "stats")
    if not result or not result.result_json:
        st.warning("No KPI data available.")
        st.stop()

    kpis = result.result_json
    df = get_clean_df(batch_id)

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Inventory Turnover", f'{kpis.get("inventory_turnover", 0)}x')
    col2.metric("Avg DSI", f'{kpis.get("avg_dsi", 0):.0f} days')
    col3.metric("OOS Rate", f'{kpis.get("oos_rate", 0):.1f}%')
    col4.metric("Slow-Moving Value", f'${kpis.get("slow_moving_value", 0):,.0f}')
    col5.metric("Total Inv. Value", f'${kpis.get("total_inventory_value", 0):,.0f}')

    st.divider()

    if not df.empty:
        # Interactive charts with Plotly
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Stock Status Distribution")
            status_counts = df["Stock_Status"].value_counts()
            fig_status = px.pie(
                values=status_counts.values, names=status_counts.index,
                color=status_counts.index,
                color_discrete_map={"Normal Stock": "#27AE60", "Low Stock": "#F39C12", "Out of Stock": "#E74C3C"},
                hole=0.4,
            )
            fig_status.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_status, use_container_width=True)

        with col_right:
            st.subheader("Inventory Value by Category")
            cat_value = df.groupby("Category")["Inventory_Value"].sum().sort_values(ascending=True)
            fig_cat = px.bar(
                x=cat_value.values, y=cat_value.index,
                orientation="h", labels={"x": "Inventory Value ($)", "y": "Category"},
                color=cat_value.values, color_continuous_scale="Viridis",
            )
            fig_cat.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_cat, use_container_width=True)

        # Vendor Performance
        st.subheader("Vendor Performance Comparison")
        vendor_stats = df.groupby("Vendor_Name").agg(
            total_value=("Inventory_Value", "sum"),
            avg_cost=("Unit_Cost", "mean"),
            avg_lead_time=("Lead_Time_Days", "mean"),
            oos_rate=("Stock_Status", lambda x: (x == "Out of Stock").mean() * 100),
            sku_count=("Product_ID", "count"),
        ).round(2)
        st.dataframe(vendor_stats, use_container_width=True)

        # Stockout Alert
        st.subheader("Stockout Alert — At-Risk SKUs")
        df_dsi = df.copy()
        df_dsi["DSI"] = np.where(df_dsi["Daily_Demand_Est"] > 0,
                                  df_dsi["Current_Stock"] / df_dsi["Daily_Demand_Est"], 999)
        at_risk = df_dsi[
            (df_dsi["DSI"] < df_dsi["Lead_Time_Days"]) | (df_dsi["Stock_Status"] == "Out of Stock")
        ].sort_values("DSI").head(20)
        if not at_risk.empty:
            st.dataframe(
                at_risk[["Product_ID", "Category", "Current_Stock", "Safety_Stock_Target",
                         "DSI", "Lead_Time_Days", "Stock_Status", "Vendor_Name"]],
                use_container_width=True,
            )
        else:
            st.success("No at-risk SKUs detected!")


# ────────────────────────────────────────────
# PAGE: Upload & Run Pipeline
# ────────────────────────────────────────────
elif page == "Upload & Run Pipeline":
    st.header("Upload Data & Run Analysis Pipeline")

    st.markdown("""
    Upload a raw inventory CSV file to automatically trigger the full analysis pipeline:
    1. **ETL** — 8-step data cleaning
    2. **Statistical Analysis** — Correlation, distribution, hypothesis testing (9 charts)
    3. **Supply Chain Optimization** — EOQ, Monte Carlo, Vendor Radar (6 charts)
    4. **ML / AI Analysis** — 30 algorithms, GA optimization (8 charts)
    5. **RL Optimization** — Q-Learning, SARSA, DQN, PPO, A2C, GA-RL Hybrid (6 charts)
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Or use existing dirty data
    existing_dirty = BASE_DIR / "Supply_Chain_Inventory_Dirty_10k.csv"
    if existing_dirty.exists():
        st.divider()
        use_existing = st.button("Or use existing dirty data (Supply_Chain_Inventory_Dirty_10k.csv)")
    else:
        use_existing = False

    input_path = None
    if uploaded_file:
        raw_path = RAW_DIR / uploaded_file.name
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_path = str(raw_path)
        st.success(f"File uploaded: {uploaded_file.name}")
    elif use_existing:
        input_path = str(existing_dirty)

    if input_path:
        st.divider()
        st.subheader("Pipeline Execution")

        # Progress tracking
        progress_bar = st.progress(0)
        status_area = st.empty()
        stage_info = st.container()

        stages_display = {
            "etl": {"label": "ETL (Data Cleaning)", "icon": "🔧"},
            "stats": {"label": "Statistical Analysis", "icon": "📊"},
            "supply_chain": {"label": "Supply Chain Optimization", "icon": "🔗"},
            "ml": {"label": "ML / AI Analysis", "icon": "🤖"},
            "rl": {"label": "RL Optimization", "icon": "🎮"},
        }
        stage_progress = {s: "pending" for s in stages_display}

        def on_progress(stage, status, data):
            """Callback to update Streamlit UI."""
            if stage in stage_progress:
                stage_progress[stage] = status

        # Run pipeline
        orchestrator = PipelineOrchestrator(on_progress=on_progress)

        status_area.info("Pipeline running... This may take a few minutes.")

        # Show stages
        with stage_info:
            stage_cols = st.columns(len(stages_display))
            for i, (key, info) in enumerate(stages_display.items()):
                stage_cols[i].markdown(f"**{info['icon']} {info['label']}**")
                stage_cols[i].caption("Pending...")

        result = orchestrator.run(input_path)
        progress_bar.progress(100)

        if result.get("status") == "completed":
            status_area.success(f"Pipeline completed! Batch ID: {result['batch_id']}")

            # Show ETL summary
            etl_stats = result.get("stages", {}).get("etl", {})
            if etl_stats:
                st.subheader("ETL Summary")
                etl_cols = st.columns(4)
                etl_cols[0].metric("Raw Rows", f'{etl_stats.get("raw_rows", "?")}')
                etl_cols[1].metric("Invalid Costs", f'{etl_stats.get("invalid_costs", 0)}')
                etl_cols[2].metric("Negative Stocks Fixed", f'{etl_stats.get("negative_stocks", 0)}')
                etl_cols[3].metric("Total Inv. Value", f'${etl_stats.get("total_inventory_value", 0):,.0f}')

            st.info("Navigate to Dashboard, Statistical Analysis, Supply Chain, or ML pages to explore results.")
        else:
            status_area.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")


# ────────────────────────────────────────────
# PAGE: Statistical Analysis
# ────────────────────────────────────────────
elif page == "Statistical Analysis":
    st.header("Statistical Analysis — Charts 01-08")

    batch_id = get_latest_batch_id()
    if not batch_id:
        st.info("No data yet. Upload a CSV first.")
        st.stop()

    df = get_clean_df(batch_id)
    if df.empty:
        st.warning("No data found.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Interactive Charts", "Generated Charts (PNG)", "Raw Data"])

    with tab1:
        # Interactive correlation heatmap
        st.subheader("Correlation Matrix (Interactive)")
        numeric_cols = ["Unit_Cost", "Current_Stock", "Daily_Demand_Est",
                        "Safety_Stock_Target", "Lead_Time_Days", "Reorder_Point", "Inventory_Value"]
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, labels=dict(color="Correlation"),
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Distribution explorer
        st.subheader("Distribution Explorer")
        col_select = st.selectbox("Select variable:", numeric_cols, index=0)
        fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["Histogram + KDE", "Box Plot"])
        data = df[col_select].dropna()
        fig_dist.add_trace(go.Histogram(x=data, nbinsx=50, name="Histogram", opacity=0.7,
                                         marker_color="#2E86C1"), row=1, col=1)
        fig_dist.add_trace(go.Box(y=data, name="Box", marker_color="#27AE60"), row=1, col=2)
        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Vendor comparison
        st.subheader("Vendor Performance (Interactive)")
        metric_select = st.selectbox("Select metric:", numeric_cols, index=0, key="vendor_metric")
        fig_vendor = px.box(df, x="Vendor_Name", y=metric_select, color="Vendor_Name",
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_vendor.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_vendor, use_container_width=True)

    with tab2:
        show_chart_images(batch_id, "chart_0")
        show_chart_images(batch_id, "chart_01")
        show_chart_images(batch_id, "chart_02")
        show_chart_images(batch_id, "chart_03")
        show_chart_images(batch_id, "chart_04")
        show_chart_images(batch_id, "chart_05")
        show_chart_images(batch_id, "chart_06")
        show_chart_images(batch_id, "chart_07")
        show_chart_images(batch_id, "chart_08")

    with tab3:
        st.subheader("Cleaned Data")
        st.dataframe(df, use_container_width=True, height=600)
        csv_data = df.to_csv(index=False)
        st.download_button("Download CSV", csv_data, "clean_data.csv", "text/csv")


# ────────────────────────────────────────────
# PAGE: Supply Chain Optimization
# ────────────────────────────────────────────
elif page == "Supply Chain Optimization":
    st.header("Supply Chain Optimization — Charts 09-14")

    batch_id = get_latest_batch_id()
    if not batch_id:
        st.info("No data yet.")
        st.stop()

    df = get_clean_df(batch_id)
    if df.empty:
        st.warning("No data found.")
        st.stop()

    tab1, tab2 = st.tabs(["Interactive Charts", "Generated Charts (PNG)"])

    with tab1:
        # EOQ Calculator
        st.subheader("EOQ Calculator (Interactive)")
        col1, col2, col3 = st.columns(3)
        annual_demand = col1.number_input("Annual Demand (units)", value=10000, step=100)
        ordering_cost = col2.number_input("Ordering Cost ($)", value=50, step=5)
        holding_rate_pct = col3.number_input("Holding Rate (%)", value=25, step=1)

        if annual_demand > 0 and holding_rate_pct > 0:
            unit_cost_input = st.slider("Unit Cost ($)", 5, 500, 100)
            holding_cost = unit_cost_input * holding_rate_pct / 100
            eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)

            st.metric("Optimal Order Quantity (EOQ)", f"{eoq:,.0f} units")

            Q_range = np.linspace(max(10, eoq * 0.1), eoq * 3, 200)
            holding_costs = (Q_range / 2) * holding_cost
            ordering_costs = (annual_demand / Q_range) * ordering_cost
            total_costs = holding_costs + ordering_costs

            fig_eoq = go.Figure()
            fig_eoq.add_trace(go.Scatter(x=Q_range, y=holding_costs, name="Holding Cost",
                                          line=dict(dash="dash", color="#3498DB")))
            fig_eoq.add_trace(go.Scatter(x=Q_range, y=ordering_costs, name="Ordering Cost",
                                          line=dict(dash="dash", color="#E74C3C")))
            fig_eoq.add_trace(go.Scatter(x=Q_range, y=total_costs, name="Total Cost",
                                          line=dict(color="#1B2A4A", width=3)))
            fig_eoq.add_vline(x=eoq, line_dash="dot", line_color="#27AE60",
                              annotation_text=f"EOQ = {eoq:,.0f}")
            fig_eoq.update_layout(height=400, xaxis_title="Order Quantity",
                                  yaxis_title="Annual Cost ($)")
            st.plotly_chart(fig_eoq, use_container_width=True)

        # Monte Carlo controls
        st.subheader("Monte Carlo Simulation (Interactive)")
        mc_category = st.selectbox("Select category:", df["Category"].unique())
        mc_sims = st.slider("Number of simulations:", 1000, 20000, 5000, 1000)
        if st.button("Run Simulation"):
            cat_df = df[(df["Category"] == mc_category) & (df["Daily_Demand_Est"] > 0)]
            if not cat_df.empty:
                np.random.seed(42)
                d_mean = cat_df["Daily_Demand_Est"].mean()
                d_std = cat_df["Daily_Demand_Est"].std()
                lt_mean = cat_df["Lead_Time_Days"].mean()
                lt_std = cat_df["Lead_Time_Days"].std()
                avg_stock = cat_df["Current_Stock"].mean()
                sim_lt = np.random.normal(lt_mean, max(lt_std, 1), mc_sims).clip(1, 60).astype(int)
                sim_demand = np.random.normal(d_mean, max(d_std, 1), mc_sims).clip(0)
                sim_total = sim_lt * sim_demand
                stockout_pct = (sim_total > avg_stock).mean() * 100

                col_mc1, col_mc2 = st.columns([2, 1])
                with col_mc1:
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Histogram(x=sim_total[sim_total <= avg_stock],
                                                   name="Fulfilled", marker_color="#27AE60", opacity=0.6))
                    fig_mc.add_trace(go.Histogram(x=sim_total[sim_total > avg_stock],
                                                   name="Stockout", marker_color="#E74C3C", opacity=0.6))
                    fig_mc.add_vline(x=avg_stock, line_dash="dash", line_color="#2E86C1",
                                     annotation_text=f"Avg Stock: {avg_stock:,.0f}")
                    fig_mc.update_layout(barmode="overlay", height=400,
                                         xaxis_title="Total Demand During Lead Time",
                                         yaxis_title="Count")
                    st.plotly_chart(fig_mc, use_container_width=True)
                with col_mc2:
                    st.metric("Stockout Probability", f"{stockout_pct:.1f}%")
                    st.metric("Avg Demand/Day", f"{d_mean:.0f}")
                    st.metric("Avg Lead Time", f"{lt_mean:.0f} days")
                    st.metric("Avg Stock", f"{avg_stock:,.0f}")
                    st.metric("Simulations", f"{mc_sims:,}")

    with tab2:
        show_chart_images(batch_id, "chart_09")
        show_chart_images(batch_id, "chart_10")
        show_chart_images(batch_id, "chart_11")
        show_chart_images(batch_id, "chart_12")
        show_chart_images(batch_id, "chart_13")
        show_chart_images(batch_id, "chart_14")


# ────────────────────────────────────────────
# PAGE: ML / AI Analysis
# ────────────────────────────────────────────
elif page == "ML / AI Analysis":
    st.header("ML / AI Analysis — Charts 15-22")

    batch_id = get_latest_batch_id()
    if not batch_id:
        st.info("No data yet.")
        st.stop()

    tab1, tab2 = st.tabs(["Generated Charts (PNG)", "Algorithm Summary"])

    with tab1:
        show_chart_images(batch_id, "chart_15")
        show_chart_images(batch_id, "chart_16")
        show_chart_images(batch_id, "chart_17")
        show_chart_images(batch_id, "chart_18")
        show_chart_images(batch_id, "chart_19")
        show_chart_images(batch_id, "chart_20")
        show_chart_images(batch_id, "chart_21")
        show_chart_images(batch_id, "chart_22")

    with tab2:
        st.subheader("30 AI/ML Algorithm Summary")
        algo_data = [
            {"#": 1, "Algorithm": "Linear Regression", "Category": "Supervised (Regression)", "Status": "Applied", "Use Case": "Inventory Value prediction"},
            {"#": 2, "Algorithm": "Logistic Regression", "Category": "Supervised (Classification)", "Status": "Applied", "Use Case": "Stock Status prediction"},
            {"#": 3, "Algorithm": "Decision Tree", "Category": "Supervised (Classification)", "Status": "Applied", "Use Case": "Stock Status prediction"},
            {"#": 4, "Algorithm": "Random Forest", "Category": "Supervised (Both)", "Status": "Applied", "Use Case": "Classification + Regression"},
            {"#": 5, "Algorithm": "SVM", "Category": "Supervised (Classification)", "Status": "Applied", "Use Case": "Stock Status prediction"},
            {"#": 6, "Algorithm": "k-NN", "Category": "Supervised (Classification)", "Status": "Applied", "Use Case": "Stock Status prediction"},
            {"#": 7, "Algorithm": "Naive Bayes", "Category": "Supervised (Classification)", "Status": "Applied", "Use Case": "Stock Status prediction"},
            {"#": 8, "Algorithm": "Gradient Boosting", "Category": "Supervised (Both)", "Status": "Applied", "Use Case": "Classification + Regression"},
            {"#": 9, "Algorithm": "AdaBoost", "Category": "Supervised (Classification)", "Status": "Applied", "Use Case": "Stock Status prediction"},
            {"#": 10, "Algorithm": "XGBoost", "Category": "Supervised (Classification)", "Status": "Optional", "Use Case": "Stock Status prediction"},
            {"#": 11, "Algorithm": "K-Means", "Category": "Unsupervised (Clustering)", "Status": "Applied", "Use Case": "Inventory segmentation"},
            {"#": 12, "Algorithm": "Hierarchical", "Category": "Unsupervised (Clustering)", "Status": "Applied", "Use Case": "Product grouping"},
            {"#": 13, "Algorithm": "DBSCAN", "Category": "Unsupervised (Clustering)", "Status": "Applied", "Use Case": "Noise/outlier detection"},
            {"#": 14, "Algorithm": "PCA", "Category": "Dimensionality Reduction", "Status": "Applied", "Use Case": "Feature visualization"},
            {"#": 15, "Algorithm": "t-SNE", "Category": "Dimensionality Reduction", "Status": "Applied", "Use Case": "Cluster visualization"},
            {"#": 16, "Algorithm": "Q-Learning", "Category": "Reinforcement Learning", "Status": "Applied", "Use Case": "Dynamic reorder decisions"},
            {"#": 17, "Algorithm": "SARSA", "Category": "Reinforcement Learning", "Status": "Applied", "Use Case": "On-policy reorder optimization"},
            {"#": 18, "Algorithm": "DQN", "Category": "Reinforcement Learning", "Status": "Applied", "Use Case": "Neural network reorder agent"},
            {"#": 19, "Algorithm": "PPO", "Category": "Reinforcement Learning", "Status": "Applied", "Use Case": "Policy gradient optimization"},
            {"#": 20, "Algorithm": "A2C (Actor-Critic)", "Category": "Reinforcement Learning", "Status": "Applied", "Use Case": "Advantage actor-critic training"},
            {"#": 21, "Algorithm": "ANN / MLP", "Category": "Deep Learning", "Status": "Applied", "Use Case": "Stock Status prediction"},
            {"#": 22, "Algorithm": "CNN", "Category": "Deep Learning", "Status": "N/A", "Use Case": "Image-based QC (needs images)"},
            {"#": 23, "Algorithm": "RNN", "Category": "Deep Learning", "Status": "Phase 4", "Use Case": "Demand time-series"},
            {"#": 24, "Algorithm": "LSTM", "Category": "Deep Learning", "Status": "Phase 4", "Use Case": "Demand forecasting"},
            {"#": 25, "Algorithm": "Transformer", "Category": "Deep Learning", "Status": "Phase 4", "Use Case": "Multi-category attention"},
            {"#": 26, "Algorithm": "K-Means++", "Category": "Unsupervised (Clustering)", "Status": "Applied", "Use Case": "Better centroid init"},
            {"#": 27, "Algorithm": "Autoencoder", "Category": "Anomaly Detection", "Status": "Applied", "Use Case": "Reconstruction error outliers"},
            {"#": 28, "Algorithm": "Isolation Forest", "Category": "Anomaly Detection", "Status": "Applied", "Use Case": "Tree-based anomaly scoring"},
            {"#": 29, "Algorithm": "GA-RL Hybrid", "Category": "RL + Optimization", "Status": "Applied", "Use Case": "GA warm-start + Q-Learning fine-tune"},
            {"#": 30, "Algorithm": "Genetic Algorithm", "Category": "Optimization", "Status": "Applied", "Use Case": "Safety stock optimization"},
        ]
        algo_df = pd.DataFrame(algo_data)

        def color_status(val):
            colors = {"Applied": "background-color: #d4edda", "Optional": "background-color: #fff3cd",
                      "Phase 2": "background-color: #cce5ff", "Phase 4": "background-color: #e2e3e5",
                      "N/A": "background-color: #f8d7da"}
            return colors.get(val, "")

        styled = algo_df.style.map(color_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, height=800)


# ────────────────────────────────────────────
# PAGE: RL Optimization
# ────────────────────────────────────────────
elif page == "RL Optimization":
    st.header("Reinforcement Learning — Inventory Optimization")

    batch_id = get_latest_batch_id()
    if not batch_id:
        st.info("No data yet. Upload a CSV first.")
        st.stop()

    rl_result = get_analysis_result(batch_id, "rl")

    tab1, tab2, tab3 = st.tabs(["RL Agent Comparison", "Generated Charts (PNG)", "Environment Details"])

    with tab1:
        if rl_result and rl_result.result_json:
            rl_kpis = rl_result.result_json

            # Best agent highlight
            best_agent = rl_kpis.get("rl_best_agent", "N/A")
            best_reward = rl_kpis.get("rl_best_reward", 0)
            best_svc = rl_kpis.get("rl_best_service_level", 0)

            st.subheader("Best Performing Agent")
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Agent", best_agent)
            col2.metric("Final Avg Reward", f"{best_reward:.1f}")
            col3.metric("Service Level", f"{best_svc*100:.1f}%")

            st.divider()

            # Per-agent metrics
            st.subheader("Agent Performance Summary")
            agent_names = []
            for key in rl_kpis:
                if key.startswith("rl_") and key.endswith("_mean_reward"):
                    name = key.replace("rl_", "").replace("_mean_reward", "")
                    agent_names.append(name)

            if agent_names:
                summary_data = []
                for name in agent_names:
                    prefix = f"rl_{name}_"
                    summary_data.append({
                        "Agent": name.replace("_", " ").title(),
                        "Mean Reward": f'{rl_kpis.get(prefix + "mean_reward", 0):.1f}',
                        "Best Reward": f'{rl_kpis.get(prefix + "best_reward", 0):.1f}',
                        "Service Level": f'{rl_kpis.get(prefix + "service_level", 0)*100:.1f}%',
                        "Convergence Ep": rl_kpis.get(prefix + "convergence", "N/A"),
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

            # Interactive reward comparison
            st.subheader("Interactive Agent Comparison")
            st.markdown("""
            The RL agents are trained on a **Gymnasium InventoryEnv** that simulates 90 days of
            inventory management. Each agent learns to decide **when and how much to reorder** to
            minimize total cost (holding + stockout + ordering) while maximizing service level.

            **Agents trained:**
            - **Q-Learning** — Tabular, off-policy TD control
            - **SARSA** — Tabular, on-policy TD control
            - **DQN** — Deep Q-Network with experience replay + target network
            - **PPO** — Proximal Policy Optimization (clipped surrogate)
            - **A2C** — Advantage Actor-Critic
            - **GA-RL Hybrid** — Genetic Algorithm warm-start + Q-Learning fine-tune
            """)
        else:
            st.info("No RL results available. Run the pipeline with the latest version to generate RL analysis.")

    with tab2:
        show_chart_images(batch_id, "chart_23")
        show_chart_images(batch_id, "chart_24")
        show_chart_images(batch_id, "chart_25")
        show_chart_images(batch_id, "chart_26")
        show_chart_images(batch_id, "chart_27")
        show_chart_images(batch_id, "chart_28")

    with tab3:
        st.subheader("Gymnasium InventoryEnv Specification")
        st.markdown("""
        | Component | Details |
        |-----------|---------|
        | **State Space** | `Box(5,)` — [current_stock, pending_orders, days_since_order, demand_trend, stockout_days] (all normalized [0,1]) |
        | **Action Space** | `Discrete(5)` — 0=no order, 1=0.5xEOQ, 2=1.0xEOQ, 3=1.5xEOQ, 4=2.0xEOQ |
        | **Reward** | `-(holding_cost + stockout_cost + ordering_cost)` |
        | **Episode Length** | 90 days |
        | **Holding Cost** | `(holding_rate / 365) * unit_cost * current_stock` per day |
        | **Stockout Cost** | `stockout_penalty * unmet_demand` per day |
        | **Ordering Cost** | Fixed cost per order placed |
        | **Demand** | Stochastic: `N(mean, std)` clipped to non-negative |
        | **Lead Time** | Fixed delay between order placement and arrival |
        """)

        st.subheader("Training Configuration")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Episodes | 300 |
        | Episode Length | 90 days |
        | Q-Learning/SARSA bins | 10 per dimension |
        | DQN replay buffer | 10,000 transitions |
        | DQN target update | Every 100 steps |
        | PPO clip epsilon | 0.2 |
        | PPO epochs per update | 4 |
        | GA population | 50, 40 generations |
        | GA-RL initial epsilon | 0.3 (warm-started) |
        """)


# ────────────────────────────────────────────
# PAGE: Pipeline History
# ────────────────────────────────────────────
elif page == "Pipeline History":
    st.header("Pipeline Run History")

    db = SessionLocal()
    runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(50).all()
    db.close()

    if not runs:
        st.info("No pipeline runs yet.")
        st.stop()

    runs_data = [{
        "Batch ID": r.batch_id,
        "Status": r.status,
        "Source": r.source_file,
        "Started": r.started_at.strftime("%Y-%m-%d %H:%M:%S") if r.started_at else "",
        "Completed": r.completed_at.strftime("%Y-%m-%d %H:%M:%S") if r.completed_at else "",
    } for r in runs]

    runs_df = pd.DataFrame(runs_data)

    def color_run_status(val):
        colors = {"completed": "background-color: #d4edda", "running": "background-color: #fff3cd",
                  "failed": "background-color: #f8d7da", "pending": "background-color: #e2e3e5"}
        return colors.get(val, "")

    styled = runs_df.style.map(color_run_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True)

    # KPI trend across runs
    if len(runs) > 1:
        st.subheader("KPI Trend Across Runs")
        db = SessionLocal()
        kpi_history = []
        for r in runs:
            if r.status == "completed":
                result = db.query(AnalysisResult).filter(
                    AnalysisResult.batch_id == r.batch_id,
                    AnalysisResult.analysis_type == "stats",
                ).first()
                if result and result.result_json:
                    entry = {"batch_id": r.batch_id[:12],
                             "date": r.completed_at.strftime("%Y-%m-%d %H:%M") if r.completed_at else ""}
                    entry.update(result.result_json)
                    kpi_history.append(entry)
        db.close()

        if len(kpi_history) > 1:
            kpi_df = pd.DataFrame(kpi_history)
            metric_choice = st.selectbox("Select KPI:", [
                "inventory_turnover", "avg_dsi", "oos_rate",
                "slow_moving_value", "total_inventory_value",
            ])
            if metric_choice in kpi_df.columns:
                fig_trend = px.line(kpi_df, x="date", y=metric_choice,
                                    markers=True, title=f"{metric_choice} Over Time")
                fig_trend.update_layout(height=350)
                st.plotly_chart(fig_trend, use_container_width=True)
