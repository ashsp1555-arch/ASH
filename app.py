import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p  { margin: 0.5rem 0 0; font-size: 1rem; opacity: 0.85; }

    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
    }
    .metric-box .label { font-size: 0.85rem; color: #666; font-weight: 500; }
    .metric-box .value { font-size: 1.8rem; font-weight: 700; color: #1e3c72; }

    .result-churn {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-no-churn {
        background: linear-gradient(135deg, #00c851, #007e33);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load or Train Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model.pkl if available, otherwise train a demo model."""
    try:
        model = joblib.load("model.pkl")
        return model, "Loaded from model.pkl"
    except FileNotFoundError:
        # Train a lightweight demo model on synthetic data
        np.random.seed(42)
        X_demo = np.random.rand(100, 6)
        y_demo = np.random.randint(0, 2, 100)

        demo_model = RandomForestClassifier(n_estimators=10, random_state=42)
        demo_model.fit(X_demo, y_demo)

        return demo_model, "Demo model (model.pkl not found)"

# Load feature names
@st.cache_resource
def load_feature_names():
    """Load feature names from feature_names.pkl if available."""
    try:
        return joblib.load("feature_names.pkl")
    except FileNotFoundError:
        return ['Age', 'ServicesOpted', 'AccountSyncedToSocialMedia',
                'FrequentFlyer', 'AnnualIncomeClass', 'BookedHotelOrNot']

# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
def main():
    model, model_status = load_model()
    feature_names = load_feature_names()

    # Header
    st.markdown('<div class="main-header"><h1>✈️ Customer Churn Predictor</h1><p>Predict customer churn using Random Forest Machine Learning</p></div>', unsafe_allow_html=True)

    # Model Status
    if "Demo" in model_status:
        st.warning("⚠️ Using demo model. Upload model.pkl for production use.")
    else:
        st.success(f"✅ {model_status}")

    # Sidebar for inputs
    st.sidebar.header("📊 Customer Information")

    # Input fields
    age = st.sidebar.slider("Age", 18, 80, 30)
    services_opted = st.sidebar.slider("Services Opted", 1, 5, 2)
    account_synced = st.sidebar.selectbox("Account Synced to Social Media", ["Yes", "No"])
    frequent_flyer = st.sidebar.selectbox("Frequent Flyer", ["Yes", "No"])
    income_class = st.sidebar.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])
    hotel_booked = st.sidebar.selectbox("Booked Hotel or Not", ["Yes", "No"])

    # Encode categorical variables
    account_synced_encoded = 1 if account_synced == "Yes" else 0
    frequent_flyer_encoded = 1 if frequent_flyer == "Yes" else 0
    income_mapping = {"Low Income": 0, "Middle Income": 1, "High Income": 2}
    income_encoded = income_mapping[income_class]
    hotel_booked_encoded = 1 if hotel_booked == "Yes" else 0

    # Create input array
    input_data = np.array([[age, services_opted, account_synced_encoded,
                           frequent_flyer_encoded, income_encoded, hotel_booked_encoded]])

    # Prediction
    if st.sidebar.button("🔍 Predict Churn Risk"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.header("🎯 Prediction Results")

        # Result display
        if prediction == 1:
            st.markdown('<div class="result-churn">⚠️ HIGH CHURN RISK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-no-churn">✅ LOW CHURN RISK</div>', unsafe_allow_html=True)

        # Probability
        st.subheader("Churn Probability")
        st.progress(probability)
        st.write(f"**{probability:.1%}** chance of churn")

        # Feature importance visualization
        if hasattr(model, 'feature_importances_'):
            st.subheader("📈 Key Factors Influencing Prediction")

            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(range(len(indices)), importances[indices], color='#2a5298', alpha=0.7)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            plt.tight_layout()

            st.pyplot(fig)

    # Information section
    st.header("ℹ️ About This Model")
    st.markdown("""
    <div class="info-card">
    <strong>Algorithm:</strong> Random Forest Classifier<br>
    <strong>Features:</strong> Age, Services Opted, Social Media Sync, Frequent Flyer Status, Income Class, Hotel Booking<br>
    <strong>Performance:</strong> ~67% accuracy on test data<br>
    <strong>Use Case:</strong> Travel industry customer retention
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit & scikit-learn*")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p  { margin: 0.5rem 0 0; font-size: 1rem; opacity: 0.85; }

    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
    }
    .metric-box .label { font-size: 0.85rem; color: #666; font-weight: 500; }
    .metric-box .value { font-size: 1.8rem; font-weight: 700; color: #1e3c72; }

    .result-churn {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-no-churn {
        background: linear-gradient(135deg, #00c851, #007e33);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load or Train Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model.pkl if available, otherwise train a demo model."""
    try:
        model = joblib.load("model.pkl")
        return model, "Loaded from model.pkl"
    except FileNotFoundError:
        # Train a lightweight demo model on synthetic data
        from sklearn.preprocessing import LabelEncoder
        np.random.seed(42)
        n = 900
        age               = np.random.randint(27, 39, n)
        frequent_flyer    = np.random.randint(0, 2, n)
        income_class      = np.random.randint(0, 3, n)
        services_opted    = np.random.randint(1, 7, n)
        social_sync       = np.random.randint(0, 2, n)
        hotel_booked      = np.random.randint(0, 2, n)

        target = (
            (services_opted <= 2).astype(int) * 0.4 +
            (frequent_flyer == 0).astype(int) * 0.3 +
            (hotel_booked == 0).astype(int) * 0.2 +
            np.random.rand(n) * 0.3
        )
        y = (target > 0.55).astype(int)

        X = np.column_stack([age, frequent_flyer, income_class,
                             services_opted, social_sync, hotel_booked])
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, "Demo model (upload model.pkl to use your trained model)"

model, model_status = load_model()

FEATURE_NAMES = [
    "Age", "FrequentFlyer", "AnnualIncomeClass",
    "ServicesOpted", "AccountSyncedToSocialMedia", "BookedHotelOrNot"
]

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>✈️ Customer Churn Prediction</h1>
    <p>Random Forest Classifier | B.Tech Gen AI – 2nd Semester | Streamlit Deployment</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ About This App")
    st.markdown(f"""
    <div class="info-card">
        <b>Model Status:</b><br>{model_status}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Info")
    st.markdown(f"""
    - **Algorithm:** Random Forest
    - **Trees:** {model.n_estimators}
    - **Features:** {len(FEATURE_NAMES)}
    - **Task:** Binary Classification
    """)

    st.markdown("### 🎯 Target")
    st.markdown("""
    - `0` → Customer will **NOT** churn
    - `1` → Customer **WILL** churn
    """)

    st.markdown("---")
    st.markdown("### 📁 Deployment Files")
    st.markdown("""
    - `app.py`
    - `model.pkl`
    - `requirements.txt`
    """)

    st.markdown("---")
    st.markdown("*Final Project · Customer Churn Prediction*")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Single Customer", "📋 Batch Prediction (CSV)", "📊 Model Insights"])

# ══════════════════════════════════════════════
# TAB 1 – Single Prediction
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Customer Details")
    st.markdown("Fill in the fields below and click **Predict** to check if the customer will churn.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        age = st.slider("🎂 Age", min_value=18, max_value=70, value=30, step=1)

        frequent_flyer = st.radio(
            "✈️ Frequent Flyer?",
            options=["No", "Yes"],
            horizontal=True
        )
        frequent_flyer_enc = 1 if frequent_flyer == "Yes" else 0

        income_class = st.selectbox(
            "💰 Annual Income Class",
            options=["Low Income", "Middle Income", "High Income"]
        )
        income_map = {"Low Income": 0, "Middle Income": 1, "High Income": 2}
        income_enc = income_map[income_class]

    with col2:
        services_opted = st.slider(
            "🛎️ Number of Services Opted", min_value=1, max_value=6, value=3, step=1,
            help="Total number of travel services the customer has opted for (1–6)"
        )

        account_synced = st.radio(
            "📱 Account Synced to Social Media?",
            options=["No", "Yes"],
            horizontal=True
        )
        account_synced_enc = 1 if account_synced == "Yes" else 0

        hotel_booked = st.radio(
            "🏨 Booked Hotel or Not?",
            options=["No", "Yes"],
            horizontal=True
        )
        hotel_booked_enc = 1 if hotel_booked == "Yes" else 0

    st.markdown("---")

    if st.button("🚀 Predict Churn"):
        input_data = np.array([[age, frequent_flyer_enc, income_enc,
                                 services_opted, account_synced_enc, hotel_booked_enc]])

        prediction    = model.predict(input_data)[0]
        probability   = model.predict_proba(input_data)[0]
        churn_prob    = probability[1] * 100
        no_churn_prob = probability[0] * 100

        # Result Banner
        if prediction == 1:
            st.markdown(f"""
            <div class="result-churn">
                ⚠️ HIGH CHURN RISK — This customer is likely to churn!<br>
                <span style="font-size:1rem; font-weight:400;">
                    Churn Probability: {churn_prob:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-no-churn">
                ✅ LOW CHURN RISK — This customer is likely to stay!<br>
                <span style="font-size:1rem; font-weight:400;">
                    Retention Probability: {no_churn_prob:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Probability Chart
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(
                ['No Churn', 'Churn'],
                [no_churn_prob, churn_prob],
                color=['#00c851', '#ff4444'],
                edgecolor='black', linewidth=0.6,
                width=0.5
            )
            for bar, val in zip(bars, [no_churn_prob, churn_prob]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 115)
            ax.set_ylabel('Probability (%)', fontsize=12)
            ax.set_title('Prediction Probability', fontsize=14, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Input Summary
        st.markdown("#### 📋 Input Summary")
        input_df = pd.DataFrame({
            "Feature": ["Age", "Frequent Flyer", "Annual Income Class",
                        "Services Opted", "Account Synced", "Hotel Booked"],
            "Value":   [age, frequent_flyer, income_class,
                        services_opted, account_synced, hotel_booked]
        })
        st.dataframe(input_df.set_index("Feature"), use_container_width=True)

        # Recommendation
        st.markdown("#### 💡 Business Recommendation")
        if prediction == 1:
            st.warning("""
            **Action Required:** This customer is at high churn risk.
            - 🎁 Offer a personalized loyalty discount or exclusive travel package.
            - 📞 Schedule a retention call or send a targeted email campaign.
            - 🏨 Encourage hotel bundling to increase engagement.
            - ✈️ Enroll in frequent flyer program with bonus miles.
            """)
        else:
            st.success("""
            **Customer is Retained:** Continue engagement strategies.
            - 🌟 Upsell premium services (lounge access, priority boarding).
            - 📧 Send satisfaction surveys and collect feedback.
            - 🎯 Offer referral bonuses to attract new customers.
            """)

# ══════════════════════════════════════════════
# TAB 2 – Batch Prediction
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Prediction from CSV")
    st.markdown("Upload a CSV file with customer data to predict churn for multiple customers at once.")

    st.markdown("""
    **Expected CSV Columns:**
    `Age`, `FrequentFlyer`, `AnnualIncomeClass`, `ServicesOpted`,
    `AccountSyncedToSocialMedia`, `BookedHotelOrNot`
    """)

    # Sample CSV download
    sample_data = pd.DataFrame({
        'Age':                       [30, 34, 27],
        'FrequentFlyer':             ['No', 'Yes', 'No'],
        'AnnualIncomeClass':         ['Middle Income', 'Low Income', 'High Income'],
        'ServicesOpted':             [3, 1, 5],
        'AccountSyncedToSocialMedia':['Yes', 'No', 'No'],
        'BookedHotelOrNot':          ['No', 'Yes', 'No']
    })
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=sample_data.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Uploaded:** {batch_df.shape[0]} rows × {batch_df.shape[1]} columns")
            st.dataframe(batch_df.head(), use_container_width=True)

            # Preprocess
            df_proc = batch_df.copy()

            # Encode FrequentFlyer
            if df_proc['FrequentFlyer'].dtype == object:
                df_proc['FrequentFlyer'] = df_proc['FrequentFlyer'].map({'Yes': 1, 'No': 0})

            # Encode AnnualIncomeClass
            if df_proc['AnnualIncomeClass'].dtype == object:
                inc_map = {'Low Income': 0, 'Middle Income': 1, 'High Income': 2}
                df_proc['AnnualIncomeClass'] = df_proc['AnnualIncomeClass'].map(inc_map)

            # Encode binary cols
            for col in ['AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
                if df_proc[col].dtype == object:
                    df_proc[col] = df_proc[col].map({'Yes': 1, 'No': 0})

            X_batch  = df_proc[FEATURE_NAMES]
            preds    = model.predict(X_batch)
            probas   = model.predict_proba(X_batch)[:, 1] * 100

            batch_df['Churn_Prediction'] = preds
            batch_df['Churn_Probability_%'] = probas.round(2)
            batch_df['Risk_Level'] = pd.cut(
                probas, bins=[0, 30, 60, 100],
                labels=['🟢 Low', '🟡 Medium', '🔴 High']
            )

            st.markdown("#### 📊 Prediction Results")
            st.dataframe(batch_df, use_container_width=True)

            # Summary stats
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Customers", len(batch_df))
            with c2:
                st.metric("Likely to Churn", int(preds.sum()),
                          delta=f"{preds.mean()*100:.1f}%", delta_color="inverse")
            with c3:
                st.metric("Likely to Stay", int((preds == 0).sum()))

            st.download_button(
                label="⬇️ Download Results CSV",
                data=batch_df.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV matches the expected column names.")

# ══════════════════════════════════════════════
# TAB 3 – Model Insights
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Insights & Feature Analysis")

    col1, col2 = st.columns(2)

    # Feature Importance
    with col1:
        st.markdown("#### Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_df)))
        bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
                       color=colors, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, feat_df['Importance']):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title('Feature Importance – Random Forest', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature Importance Table
    with col2:
        st.markdown("#### Feature Importance Scores")
        fi_table = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance': [f"{v:.4f}" for v in importances],
            'Rank': pd.Series(importances).rank(ascending=False).astype(int).values
        }).sort_values('Rank')
        st.dataframe(fi_table.set_index('Rank'), use_container_width=True)

        st.markdown("#### 🔑 Key Insights")
        top_feat = fi_table.iloc[0]['Feature']
        st.markdown(f"""
        - **Most important feature:** `{top_feat}`
        - ServicesOpted is a strong churn predictor
        - Frequent Flyer status significantly impacts retention
        - Hotel booking correlates with lower churn risk
        - Social media sync indicates higher engagement
        """)

    # Model Configuration
    st.markdown("---")
    st.markdown("#### ⚙️ Model Configuration")
    config_col1, config_col2, config_col3, config_col4 = st.columns(4)
    with config_col1:
        st.markdown(f"""<div class="metric-box">
            <div class="label">Algorithm</div>
            <div class="value" style="font-size:1.1rem;">Random Forest</div>
        </div>""", unsafe_allow_html=True)
    with config_col2:
        st.markdown(f"""<div class="metric-box">
            <div class="label">No. of Trees</div>
            <div class="value">{model.n_estimators}</div>
        </div>""", unsafe_allow_html=True)
    with config_col3:
        st.markdown(f"""<div class="metric-box">
            <div class="label">Max Depth</div>
            <div class="value">{model.max_depth or 'None'}</div>
        </div>""", unsafe_allow_html=True)
    with config_col4:
        st.markdown(f"""<div class="metric-box">
            <div class="label">Features</div>
            <div class="value">{len(FEATURE_NAMES)}</div>
        </div>""", unsafe_allow_html=True)

    # How to Deploy section
    st.markdown("---")
    st.markdown("#### 🚀 Deployment Steps")
    st.markdown("""
    1. **Generate model.pkl** – Run the Jupyter Notebook to train and save `model.pkl`
    2. **Create GitHub Repo** – Upload `app.py`, `model.pkl`, `requirements.txt`
    3. **Streamlit Cloud** – Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
    4. **Deploy App** – Click *New App* → select repo → set main file as `app.py`
    5. **Share Link** – Streamlit generates a public URL like `https://your-app.streamlit.app`
    """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding: 0.5rem;">
    ✈️ Customer Churn Prediction | B.Tech Gen AI – 2nd Semester | 
    Built with Random Forest + Streamlit
</div>
""", unsafe_allow_html=True)
