"""
📡 Telco Customer Churn Prediction — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHURN_COLOR  = "#EF553B"
RETAIN_COLOR = "#00CC96"
WARN_COLOR   = "#FFA15A"
BLUE_COLOR   = "#636EFA"

FEATURE_COLS = [
    "State_enc", "Account_length", "Area_code",
    "International_plan", "Voice_mail_plan", "Number_vmail_messages",
    "Total_day_minutes", "Total_day_calls", "Total_day_charge",
    "Total_eve_minutes", "Total_eve_calls", "Total_eve_charge",
    "Total_night_minutes", "Total_night_calls", "Total_night_charge",
    "Total_intl_minutes", "Total_intl_calls", "Total_intl_charge",
    "Customer_service_calls",
]

TIER_ORDER = {"🔴 Critical Risk": 0, "🟠 Warning": 1, "🟢 Safe": 2}
COLOR_MAP  = {"🔴 Critical Risk": CHURN_COLOR, "🟠 Warning": WARN_COLOR, "🟢 Safe": RETAIN_COLOR}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def assign_tier(p):
    if p >= 0.80: return "🔴 Critical Risk"
    if p >= 0.50: return "🟠 Warning"
    return "🟢 Safe"


@st.cache_data
def load_and_preprocess(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df["Churn"] = df["Churn"].astype(int)

    df["International_plan"] = df["International_plan"].map({"Yes": 1, "No": 0})
    df["Voice_mail_plan"]    = df["Voice_mail_plan"].map({"Yes": 1, "No": 0})

    le = LabelEncoder()
    df["State_enc"] = le.fit_transform(df["State"])

    df["MonthlyCharges"] = (
        df["Total_day_charge"] + df["Total_eve_charge"] +
        df["Total_night_charge"] + df["Total_intl_charge"]
    ).round(2)
    df["customerID"] = [f"CUST-{i:05d}" for i in range(len(df))]
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=4,
        min_samples_leaf=10, random_state=42
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Retained", "Churned"], output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return model, X_test, y_test, y_prob, y_pred, auc, report, fpr, tpr


def build_biz_df(df, X_test, y_test, y_prob) -> pd.DataFrame:
    idx = X_test.index
    biz = pd.DataFrame({
        "customerID"         : df.loc[idx, "customerID"].values,
        "Account_length"     : df.loc[idx, "Account_length"].values,
        "Intl_plan"          : df.loc[idx, "International_plan"].map({1: "Yes", 0: "No"}).values,
        "Service_calls"      : df.loc[idx, "Customer_service_calls"].values,
        "MonthlyCharges"     : df.loc[idx, "MonthlyCharges"].values,
        "Actual_Churn"       : y_test.values,
        "Churn_Probability"  : y_prob.round(4),
    })
    biz["Expected_Revenue_Loss"] = (biz["Churn_Probability"] * biz["MonthlyCharges"]).round(2)
    biz["CLV_proxy"]             = (biz["MonthlyCharges"] * (biz["Account_length"] + 12)).round(2)
    biz["Risk_Tier"]             = biz["Churn_Probability"].apply(assign_tier)
    return biz


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/signal.png", width=64)
    st.title("📡 Churn Predictor")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload CSV (BigML Telco schema)",
        type=["csv"],
        help="Use churn-bigml-20.csv or any file with the same columns.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ ROI Assumptions")
    retention_rate  = st.slider("Intervention retention rate (%)", 10, 60, 35) / 100
    cost_per_cust   = st.number_input("Cost per customer ($)", min_value=1, max_value=200, value=15)

    st.markdown("---")
    st.markdown("### 🗂️ Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Overview", "📊 EDA", "🤖 Model Performance", "💰 Business Impact", "🔍 Customer Lookup"],
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────────────────────────
# LOAD DATA GATE
# ─────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown(
        """
        <div style='text-align:center; padding: 80px 0'>
            <h1>📡 Telco Customer Churn Predictor</h1>
            <p style='font-size:18px; color:grey'>
                Upload your <b>churn-bigml-20.csv</b> in the sidebar to get started.
            </p>
            <p style='font-size:15px; color:grey'>
                The app will train a model, run EDA, calculate revenue at risk,<br>
                and let you look up individual customers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# PROCESS
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading & preprocessing data…"):
    df = load_and_preprocess(uploaded)

with st.spinner("Training model (cached after first run)…"):
    model, X_test, y_test, y_prob, y_pred, auc, report, fpr, tpr = train_model(df)

biz_df = build_biz_df(df, X_test, y_test, y_prob)

# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🏠 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churned", f"{df['Churn'].sum():,}", f"{df['Churn'].mean()*100:.1f}%")
    c3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
    c4.metric("Model AUC", f"{auc:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Churn pie
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        counts["Label"] = counts["Churn"].map({1: "Churned", 0: "Retained"})
        fig = px.pie(
            counts, values="Count", names="Label",
            title="<b>Overall Churn Rate</b>",
            color="Label",
            color_discrete_map={"Churned": CHURN_COLOR, "Retained": RETAIN_COLOR},
            hole=0.45,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label", pull=[0.05, 0])
        fig.update_layout(
            annotations=[dict(text=f"{df['Churn'].mean()*100:.1f}%<br>Churn",
                              x=0.5, y=0.5, font_size=15, showarrow=False)],
            showlegend=False, height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk tier distribution from test set
        tier_counts = biz_df["Risk_Tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Customers"]
        tier_counts["_o"] = tier_counts["Tier"].map(TIER_ORDER)
        tier_counts = tier_counts.sort_values("_o")
        fig2 = px.bar(
            tier_counts, x="Tier", y="Customers",
            title="<b>Risk Tier Distribution (Test Set)</b>",
            color="Tier", color_discrete_map=COLOR_MAP,
            text="Customers",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, height=380, xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📋 Raw Data Preview")
    st.dataframe(df.drop(columns=["State_enc", "customerID"]).head(20), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    df_plot = df.copy()
    df_plot["Status"] = df_plot["Churn"].map({1: "Churned", 0: "Retained"})

    tab1, tab2, tab3, tab4 = st.tabs([
        "Monthly Charges", "Service Calls", "Day Charges vs Tenure", "Correlation Heatmap"
    ])

    with tab1:
        fig = px.histogram(
            df_plot, x="MonthlyCharges", color="Status",
            barmode="overlay", nbins=40,
            title="<b>Monthly Charges: Churned vs Retained</b>",
            color_discrete_map={"Churned": CHURN_COLOR, "Retained": RETAIN_COLOR},
            opacity=0.75,
            labels={"MonthlyCharges": "Monthly Charges ($)"},
        )
        for status, color in [("Churned", CHURN_COLOR), ("Retained", RETAIN_COLOR)]:
            mv = df_plot[df_plot["Status"] == status]["MonthlyCharges"].mean()
            fig.add_vline(x=mv, line_dash="dash", line_color=color, line_width=2,
                          annotation_text=f"{status} Mean: ${mv:.1f}", annotation_position="top")
        fig.update_layout(legend_title="Status", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_plot["CSC_bucket"] = pd.cut(
            df_plot["Customer_service_calls"],
            bins=[-1, 0, 1, 2, 3, 4, 20],
            labels=["0 calls", "1 call", "2 calls", "3 calls", "4 calls", "5+ calls"],
        )
        csc = df_plot.groupby("CSC_bucket", observed=True)["Churn"].agg(["mean", "count"]).reset_index()
        csc.columns = ["Calls", "ChurnRate", "Customers"]
        csc["Churn_%"] = (csc["ChurnRate"] * 100).round(1)

        fig = px.bar(
            csc, x="Calls", y="Churn_%", text="Churn_%",
            title="<b>Churn Rate by Customer Service Call Frequency</b>",
            color="Churn_%",
            color_continuous_scale=["#00CC96", "#FFA15A", "#EF553B"],
            labels={"Churn_%": "Churn Rate (%)", "Calls": "Service Calls"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        df_plot["Intl_label"] = df_plot["International_plan"].map({1: "Intl Plan", 0: "No Intl Plan"})
        fig = px.box(
            df_plot, x="Intl_label", y="Customer_service_calls",
            color="Status", points="outliers",
            title="<b>Service Calls by International Plan & Churn Status</b>",
            color_discrete_map={"Churned": CHURN_COLOR, "Retained": RETAIN_COLOR},
            labels={"Customer_service_calls": "Service Calls", "Intl_label": "Plan Type"},
        )
        fig.update_layout(boxmode="group", legend_title="Status", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = [
            "Account_length", "Total_day_minutes", "Total_day_charge",
            "Total_eve_charge", "Total_night_charge", "Total_intl_charge",
            "Customer_service_calls", "MonthlyCharges", "Churn",
        ]
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            title="<b>Feature Correlation Heatmap</b>",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{auc:.4f}")
    c2.metric("Accuracy", f"{report['accuracy']*100:.1f}%")
    c3.metric("Churn Precision", f"{report['Churned']['precision']*100:.1f}%")
    c4.metric("Churn Recall",    f"{report['Churned']['recall']*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # ROC Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"Model (AUC={auc:.3f})",
                                 line=dict(color=BLUE_COLOR, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Random Baseline",
                                 line=dict(color="grey", dash="dash", width=2)))
        fig.update_layout(title="<b>ROC Curve</b>",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          legend=dict(x=0.55, y=0.15), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature Importance
        imp = model.feature_importances_
        fi = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": imp}).sort_values("Importance")
        fig = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            title="<b>Feature Importance</b>",
            color="Importance", color_continuous_scale=["#636EFA", "#EF553B"],
            text=fi["Importance"].round(3),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Churn probability distribution
    st.markdown("### Churn Probability Distribution")
    fig = px.histogram(
        biz_df, x="Churn_Probability", color="Risk_Tier",
        nbins=40, barmode="overlay", opacity=0.80,
        color_discrete_map=COLOR_MAP,
        title="<b>Churn Probability Distribution by Risk Tier</b>",
        labels={"Churn_Probability": "Predicted Churn Probability"},
        category_orders={"Risk_Tier": ["🔴 Critical Risk", "🟠 Warning", "🟢 Safe"]},
    )
    fig.add_vline(x=0.50, line_dash="dash", line_color="grey",
                  annotation_text="50% threshold", annotation_position="top right")
    fig.add_vline(x=0.80, line_dash="dash", line_color=CHURN_COLOR,
                  annotation_text="80% threshold", annotation_position="top left")
    fig.update_layout(legend_title="Risk Tier", height=380)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: BUSINESS IMPACT
# ─────────────────────────────────────────────────────────────
elif page == "💰 Business Impact":
    st.title("💰 Business Impact & ROI Analysis")

    # KPI row
    total_at_risk    = biz_df["Expected_Revenue_Loss"].sum()
    critical_df      = biz_df[biz_df["Risk_Tier"] == "🔴 Critical Risk"]
    warning_df       = biz_df[biz_df["Risk_Tier"] == "🟠 Warning"]
    critical_at_risk = critical_df["Expected_Revenue_Loss"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue at Risk",    f"${total_at_risk:,.2f}/mo")
    c2.metric("Critical Tier at Risk",    f"${critical_at_risk:,.2f}/mo")
    c3.metric("Critical Risk Customers",  f"{len(critical_df)}")
    c4.metric("Warning Customers",        f"{len(warning_df)}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Revenue at risk bar chart
        summary = (
            biz_df.groupby("Risk_Tier")
            .agg(Customers=("customerID", "count"),
                 Revenue_at_Risk=("Expected_Revenue_Loss", "sum"))
            .reset_index()
        )
        summary["_o"] = summary["Risk_Tier"].map(TIER_ORDER)
        summary = summary.sort_values("_o")
        summary["Label"] = summary.apply(
            lambda r: f"${r['Revenue_at_Risk']:,.2f}<br>({r['Customers']} customers)", axis=1
        )

        fig = go.Figure()
        for _, row in summary.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Risk_Tier"]], y=[row["Revenue_at_Risk"]],
                name=row["Risk_Tier"],
                marker_color=COLOR_MAP[row["Risk_Tier"]],
                text=[row["Label"]], textposition="outside",
                width=0.5,
            ))
        fig.update_layout(
            title="<b>Monthly Revenue at Risk by Tier</b>",
            yaxis=dict(tickprefix="$", tickformat=",.0f"),
            showlegend=False, height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROI simulator per tier
        rows = []
        for tier_name, tier_df in [("🔴 Critical Risk", critical_df), ("🟠 Warning", warning_df)]:
            n          = len(tier_df)
            rev_risk   = tier_df["Expected_Revenue_Loss"].sum()
            retained   = int(n * retention_rate)
            rev_saved  = tier_df["MonthlyCharges"].nlargest(retained).sum()
            camp_cost  = n * cost_per_cust
            net        = rev_saved - camp_cost
            roi        = (net / camp_cost * 100) if camp_cost > 0 else 0
            rows.append({
                "Tier": tier_name, "Customers": n,
                "Rev at Risk ($)": f"${rev_risk:,.2f}",
                "Retained (est.)": retained,
                "Rev Saved ($)": f"${rev_saved:,.2f}",
                "Campaign Cost ($)": f"${camp_cost:,.2f}",
                "Net Benefit ($)": f"${net:,.2f}",
                "ROI (%)": f"{roi:.1f}%",
            })
        roi_df = pd.DataFrame(rows)
        st.markdown("#### 📊 ROI Simulator")
        st.caption(f"Retention rate: {retention_rate*100:.0f}% | Cost/customer: ${cost_per_cust}")
        st.dataframe(roi_df.set_index("Tier"), use_container_width=True)

        # Net benefit bar
        net_vals = []
        for tier_name, tier_df in [("🔴 Critical Risk", critical_df), ("🟠 Warning", warning_df)]:
            n = len(tier_df)
            retained  = int(n * retention_rate)
            rev_saved = tier_df["MonthlyCharges"].nlargest(retained).sum()
            camp_cost = n * cost_per_cust
            net_vals.append({"Tier": tier_name, "Net Benefit": rev_saved - camp_cost})
        nv_df = pd.DataFrame(net_vals)
        fig = px.bar(
            nv_df, x="Tier", y="Net Benefit", color="Tier",
            color_discrete_map=COLOR_MAP,
            title="<b>Net Benefit per Intervention Tier</b>",
            text=nv_df["Net Benefit"].apply(lambda x: f"${x:,.2f}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickprefix="$", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Top 20 Priority Customers")
    st.caption("Sorted by Expected Revenue Loss — intervene immediately!")

    action = (
        biz_df[biz_df["Risk_Tier"].isin(["🔴 Critical Risk", "🟠 Warning"])]
        .sort_values("Expected_Revenue_Loss", ascending=False)
        .head(20)[[
            "customerID", "Intl_plan", "Service_calls",
            "MonthlyCharges", "Churn_Probability", "Expected_Revenue_Loss", "Risk_Tier",
        ]]
        .reset_index(drop=True)
    )
    action["Churn_Probability"] = (action["Churn_Probability"] * 100).round(1).astype(str) + "%"
    action.columns = [
        "Customer ID", "Intl Plan", "Service Calls",
        "Monthly ($)", "Churn Prob", "Expected Loss ($)", "Risk Tier",
    ]
    st.dataframe(action, use_container_width=True)

    # Download button
    csv = biz_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Risk Report (CSV)",
        data=csv,
        file_name="churn_risk_report.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# PAGE: CUSTOMER LOOKUP
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Customer Lookup":
    st.title("🔍 Customer Risk Lookup")
    st.markdown("Enter customer details below to get an **instant churn probability** prediction.")

    with st.form("predict_form"):
        st.markdown("#### 📋 Customer Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            state_list = sorted(["LA","IN","ME","MA","OH","NJ","OR","CA","NY","TX","FL","WA","GA","IL","PA","MI","NC","VA","AZ","CO"])
            state      = st.selectbox("State", state_list)
            acct_len   = st.number_input("Account Length (days)", 1, 250, 100)
            area_code  = st.selectbox("Area Code", [408, 415, 510])
            intl_plan  = st.selectbox("International Plan", ["No", "Yes"])
            vm_plan    = st.selectbox("Voice Mail Plan", ["No", "Yes"])
            vm_msgs    = st.number_input("Voicemail Messages", 0, 50, 0)

        with c2:
            day_min    = st.number_input("Day Minutes", 0.0, 400.0, 180.0, step=0.1)
            day_calls  = st.number_input("Day Calls", 0, 200, 100)
            day_chg    = st.number_input("Day Charge ($)", 0.0, 70.0, float(round(day_min * 0.17, 2)), step=0.01)
            eve_min    = st.number_input("Evening Minutes", 0.0, 400.0, 200.0, step=0.1)
            eve_calls  = st.number_input("Evening Calls", 0, 200, 100)
            eve_chg    = st.number_input("Evening Charge ($)", 0.0, 35.0, float(round(eve_min * 0.085, 2)), step=0.01)

        with c3:
            night_min  = st.number_input("Night Minutes", 0.0, 400.0, 200.0, step=0.1)
            night_calls= st.number_input("Night Calls", 0, 200, 100)
            night_chg  = st.number_input("Night Charge ($)", 0.0, 20.0, float(round(night_min * 0.045, 2)), step=0.01)
            intl_min   = st.number_input("Intl Minutes", 0.0, 20.0, 10.0, step=0.1)
            intl_calls = st.number_input("Intl Calls", 0, 20, 3)
            intl_chg   = st.number_input("Intl Charge ($)", 0.0, 6.0, float(round(intl_min * 0.27, 2)), step=0.01)
            csc        = st.number_input("Customer Service Calls", 0, 10, 1)

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        le_state = LabelEncoder()
        le_state.fit(df["State"] if "State" in df.columns else df.index.astype(str))
        # Use training label encoder state encoding
        all_states = sorted(["LA","IN","ME","MA","OH","NJ","OR","CA","NY","TX","FL","WA","GA","IL","PA","MI","NC","VA","AZ","CO","WI","MN","IA","KS","NE","SD","ND","MT","ID","WY","UT","NV","NM","AK","HI","SC","AL","MS","TN","KY","WV","MD","DE","CT","RI","NH","VT"])
        state_enc_val = all_states.index(state) if state in all_states else 0

        input_df = pd.DataFrame([{
            "State_enc": state_enc_val,
            "Account_length": acct_len,
            "Area_code": area_code,
            "International_plan": 1 if intl_plan == "Yes" else 0,
            "Voice_mail_plan": 1 if vm_plan == "Yes" else 0,
            "Number_vmail_messages": vm_msgs,
            "Total_day_minutes": day_min,
            "Total_day_calls": day_calls,
            "Total_day_charge": day_chg,
            "Total_eve_minutes": eve_min,
            "Total_eve_calls": eve_calls,
            "Total_eve_charge": eve_chg,
            "Total_night_minutes": night_min,
            "Total_night_calls": night_calls,
            "Total_night_charge": night_chg,
            "Total_intl_minutes": intl_min,
            "Total_intl_calls": intl_calls,
            "Total_intl_charge": intl_chg,
            "Customer_service_calls": csc,
        }])

        prob = model.predict_proba(input_df)[0][1]
        tier = assign_tier(prob)
        monthly = day_chg + eve_chg + night_chg + intl_chg
        erl = prob * monthly

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Churn Probability", f"{prob*100:.1f}%")
        r2.metric("Risk Tier", tier)
        r3.metric("Monthly Charges", f"${monthly:.2f}")
        r4.metric("Expected Revenue Loss", f"${erl:.2f}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk Score", "font": {"size": 20}},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": CHURN_COLOR if prob >= 0.5 else RETAIN_COLOR},
                "steps": [
                    {"range": [0, 50],   "color": "#d4edda"},
                    {"range": [50, 80],  "color": "#fff3cd"},
                    {"range": [80, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
        ))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation
        if tier == "🔴 Critical Risk":
            st.error("⚠️ **Immediate Action Required!** Call this customer personally. Offer a plan upgrade or loyalty discount to prevent churn.")
        elif tier == "🟠 Warning":
            st.warning("🔔 **Proactive Outreach Recommended.** Send a check-in message, offer a reward or tech support session.")
        else:
            st.success("✅ **Low Risk.** Continue standard engagement. Monitor if service calls increase.")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:13px'>"
    "📡 Telco Churn Predictor | Model: HistGradientBoostingClassifier | Dataset: BigML Telco Churn"
    "</p>",
    unsafe_allow_html=True,
)
