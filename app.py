import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.preprocessing import LabelEncoder

# --- TabPFN Initialization ---
# NOTE: Adjust this import based on the exact TabPFN-3.5 hackathon SDK instructions.
# If it uses the sklearn API:
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    st.error("TabPFN not found. Please install the TabPFN-3.5 package.")

st.set_page_config(page_title="LabelWise: TabPFN ROI Calculator", layout="wide")

# --- Helper Functions ---
def run_scaling_analysis(df, target_col, task_type, ratios, cost_per_label, labels_per_hour, n_splits=3):
    """Runs TabPFN across different data ratios and calculates ROI metrics."""
    results = []
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target if classification
    if task_type == "Classification" and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    progress_bar = st.progress(0)
    total_steps = len(ratios) * n_splits
    current_step = 0

    for ratio in ratios:
        for split_idx in range(n_splits):
            # 1. Split Data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=ratio, random_state=split_idx, stratify=y if task_type == "Classification" else None
            )
            
            # 2. Run TabPFN (Zero-shot / No heavy training loop)
            if task_type == "Classification":
                model = TabPFNClassifier(device='cpu') # Change to 'cuda' if available
                # NOTE: If TabPFN-3.5 uses the original API, change to: model.predict(X_val, X_train, y_train)
                model.fit(X_train, y_train) 
                preds = model.predict(X_val)
                metric = accuracy_score(y_val, preds)
                metric_name = "Accuracy"
            else:
                model = TabPFNRegressor(device='cpu')
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                metric = r2_score(y_val, preds)
                metric_name = "R² Score"

            # 3. Calculate Business Metrics
            n_train = len(X_train)
            cost = n_train * cost_per_label
            time_hours = n_train / labels_per_hour

            results.append({
                "Data Ratio": ratio,
                "Train Size": n_train,
                metric_name: metric,
                "Cost ($)": cost,
                "Time (Hours)": time_hours,
                "Split": split_idx
            })
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)

    progress_bar.empty()
    
    # Aggregate results (average across splits)
    df_results = pd.DataFrame(results)
    agg_results = df_results.groupby("Data Ratio").agg({
        "Train Size": "first",
        metric_name: "mean",
        "Cost ($)": "mean",
        "Time (Hours)": "mean"
    }).reset_index()
    
    # Calculate Marginal ROI
    agg_results["Metric Gain"] = agg_results[metric_name].diff()
    agg_results["Cost Increment ($)"] = agg_results["Cost ($)"].diff()
    agg_results["Marginal ROI (Gain/$)"] = agg_results["Metric Gain"] / agg_results["Cost Increment ($)"]
    agg_results["Marginal ROI (Gain/$)"] = agg_results["Marginal ROI (Gain/$)"].fillna(0)

    return agg_results, metric_name

def find_optimal_stopping_point(df_results, metric_name):
    """Finds the point where performance reaches 95% of its maximum observed value."""
    max_metric = df_results[metric_name].max()
    threshold = max_metric * 0.95
    
    # Find first index where metric >= threshold
    optimal_rows = df_results[df_results[metric_name] >= threshold]
    if not optimal_rows.empty:
        return optimal_rows.iloc[0]
    return df_results.iloc[-1] # Fallback to max data if never reaches 95%

# --- Streamlit UI ---
st.title("🏷️ LabelWise: TabPFN ROI Calculator")
st.markdown("Optimize your data labeling budget. See exactly where diminishing returns hit using **TabPFN-3.5's** instant zero-shot predictions.")

# Sidebar for Inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Tabular Data (CSV)", type=["csv"])
    
    task_type = st.radio("Task Type", ["Classification", "Regression"])
    
    st.markdown("---")
    st.subheader("💰 Labeling Economics")
    cost_per_label = st.number_input("Cost per label ($)", min_value=0.01, value=1.50, step=0.10)
    labels_per_hour = st.number_input("Labeling speed (labels/hour)", min_value=1, value=30, step=5)
    
    st.markdown("---")
    st.subheader("📊 Analysis Settings")
    min_ratio = st.slider("Min Data Ratio", 0.05, 0.5, 0.1)
    max_ratio = st.slider("Max Data Ratio", 0.6, 1.0, 0.9)
    steps = st.slider("Number of steps", 3, 10, 5)
    
    run_button = st.button("🚀 Run Scaling Analysis", type="primary", use_container_width=True)

# Main Area
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    target_col = st.selectbox("Select Target Column", df.columns)
    
    if run_button and TABPFN_AVAILABLE:
        if target_col not in df.columns:
            st.error("Please select a target column.")
        else:
            st.info(f"Running TabPFN-3.5 across {steps} data splits... (No retraining required!)")
            
            ratios = np.linspace(min_ratio, max_ratio, steps)
            
            with st.spinner("Evaluating scaling curves..."):
                results_df, metric_name = run_scaling_analysis(
                    df, target_col, task_type, ratios, cost_per_label, labels_per_hour
                )
            
            optimal_point = find_optimal_stopping_point(results_df, metric_name)
            
            # --- Visualizations ---
            tab1, tab2, tab3 = st.tabs(["📈 Scaling Curves", "💵 ROI & Cost", "💡 Business Insights"])
            
            with tab1:
                st.subheader(f"Performance vs. Data Size ({metric_name})")
                fig1 = px.line(
                    results_df, x="Train Size", y=metric_name, 
                    markers=True, title="How performance scales with more data"
                )
                fig1.add_vline(x=optimal_point["Train Size"], line_dash="dash", line_color="red", 
                               annotation_text=f"Optimal: {int(optimal_point['Train Size'])} samples")
                st.plotly_chart(fig1, use_container_width=True)

            with tab2:
                st.subheader("Performance vs. Total Cost ($)")
                fig2 = px.line(
                    results_df, x="Cost ($)", y=metric_name, 
                    markers=True, title="Metric gain per dollar spent"
                )
                fig2.add_vline(x=optimal_point["Cost ($)"], line_dash="dash", line_color="red",
                               annotation_text=f"Optimal Spend: ${optimal_point['Cost ($)']:,.2f}")
                st.plotly_chart(fig2, use_container_width=True)
                
                st.subheader("Marginal ROI (Gain per Dollar)")
                fig3 = px.bar(
                    results_df.dropna(), x="Data Ratio", y="Marginal ROI (Gain/$)",
                    title="Diminishing Returns: Where does the next dollar stop helping?"
                )
                st.plotly_chart(fig3, use_container_width=True)

            with tab3:
                st.subheader("💡 Actionable Recommendations")
                col1, col2, col3 = st.columns(3)
                col1.metric("Optimal Samples", f"{int(optimal_point['Train Size']):,}")
                col2.metric("Optimal Budget", f"${optimal_point['Cost ($)']:,.2f}")
                col3.metric("Time Required", f"{optimal_point['Time (Hours)']:.1f} Hours")
                
                st.markdown(f"""
                **Recommendation:** 
                To achieve ~95% of your maximum possible model performance ({optimal_point[metric_name]:.2%}), 
                you should purchase **{int(optimal_point['Train Size']):,} labels**. 
                
                This will cost **${optimal_point['Cost ($)']:,.2f}** and take approximately **{optimal_point['Time (Hours)']:.1f} hours** 
                of human labor. Spending more on additional labels will yield diminishing returns.
                """)
                
                st.caption("*Powered by TabPFN-3.5. Because you don't need to retrain to know your scaling curve.*")

else:
    st.info("👈 Upload a CSV file in the sidebar to get started!")
    st.markdown("""
    ### Why use LabelWise?
    Traditional ML models require hours of retraining to generate a single point on a scaling curve. 
    **TabPFN-3.5** evaluates the entire curve in seconds because it requires **zero training**. 
    This allows data science teams to instantly calculate the exact ROI of their data labeling budgets.
    """)
