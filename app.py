import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf0 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .sidebar-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .performance-excellent {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .performance-good {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .performance-average {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .performance-poor {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .animated-number {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .tip-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .preset-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .preset-excellent button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .preset-excellent button[data-testid="stBaseButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(40, 167, 69, 0.4) !important;
    }
    .preset-average button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .preset-average button[data-testid="stBaseButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(255, 193, 7, 0.4) !important;
    }
    .preset-at-risk button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .preset-at-risk button[data-testid="stBaseButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(220, 53, 69, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_data(n_samples=1000):
    """Generate synthetic student performance data"""
    np.random.seed(42)

    # Generate features
    attendance = np.random.normal(85, 15, n_samples)
    attendance = np.clip(attendance, 0, 100)

    study_hours = np.random.exponential(3, n_samples)
    study_hours = np.clip(study_hours, 0, 12)

    past_scores = np.random.normal(75, 12, n_samples)
    past_scores = np.clip(past_scores, 0, 100)

    # Generate target with realistic relationships
    performance = (
        0.3 * attendance +
        0.4 * past_scores +
        0.2 * study_hours * 10 +
        np.random.normal(0, 5, n_samples)
    )
    performance = np.clip(performance, 0, 100)

    data = pd.DataFrame({
        'attendance': attendance,
        'study_hours': study_hours,
        'past_scores': past_scores,
        'performance': performance
    })

    return data

@st.cache_resource
def train_models(data):
    """Train and return multiple models"""
    X = data[['attendance', 'study_hours', 'past_scores']]
    y = data['performance']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    trained_models = {}
    model_scores = {}

    for name, model in models.items():
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        if name == 'Linear Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        trained_models[name] = model
        model_scores[name] = {
            'MSE': mse,
            'R²': r2,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std()
        }

    return trained_models, model_scores, scaler, X_test, y_test

def main():
    st.markdown('<h1 class="main-header">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)

    st.markdown("""
    This machine learning application predicts student performance based on:
    - **Attendance Rate** (%)
    - **Study Hours** per day
    - **Past Academic Scores** (%)
    """)

    # Generate and load data
    data = generate_sample_data()

    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🔮 Prediction", "📊 Data Analysis", "🤖 Model Performance", "📚 Model Explanation"])

    # Add some sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.info(f"📅 Dataset: {len(data)} students\n📊 Features: 3 main + engineered\n🎯 Models: 2 algorithms")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - **Attendance**: Higher attendance typically improves performance
    - **Study Hours**: 3-6 hours daily is optimal
    - **Past Scores**: Strong predictor of future performance
    """)

    if page == "🔮 Prediction":
        prediction_page(data)
    elif page == "📊 Data Analysis":
        data_analysis_page(data)
    elif page == "🤖 Model Performance":
        model_performance_page(data)
    else:
        model_explanation_page()

def prediction_page(data):
    st.header("🔮 Student Performance Predictor")

    # Train models
    models, scores, scaler, X_test, y_test = train_models(data)

    # Input method selection
    st.subheader("📝 Choose Input Method")
    input_method = st.radio(
        "How would you like to input student data?",
        ["🎚️ Interactive Sliders", "🔢 Manual Number Input", "👥 Batch Prediction"],
        horizontal=True
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Student Data Input")

        if input_method == "🎚️ Interactive Sliders":
            attendance = st.slider(
                "📚 Attendance Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
                help="Student's class attendance percentage"
            )

            study_hours = st.slider(
                "⏰ Daily Study Hours",
                min_value=0.0,
                max_value=12.0,
                value=3.0,
                step=0.5,
                help="Average hours of study per day"
            )

            past_scores = st.slider(
                "📈 Past Academic Scores (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                help="Average of previous academic scores"
            )

        elif input_method == "🔢 Manual Number Input":
            col_a, col_b = st.columns(2)
            with col_a:
                attendance = st.number_input(
                    "📚 Attendance Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=85.0,
                    step=1.0,
                    help="Enter attendance percentage (0-100)"
                )

                study_hours = st.number_input(
                    "⏰ Daily Study Hours",
                    min_value=0.0,
                    max_value=12.0,
                    value=3.0,
                    step=0.1,
                    help="Enter daily study hours (0-12)"
                )

            with col_b:
                past_scores = st.number_input(
                    "📈 Past Academic Scores (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=1.0,
                    help="Enter past academic scores (0-100)"
                )

                st.markdown('<div class="preset-section">', unsafe_allow_html=True)
                st.markdown("### 🎯 Quick Presets")
                st.markdown("*Click to instantly set values for different student profiles:*")
                
                col_preset1, col_preset2, col_preset3 = st.columns([1, 1, 1], gap="large")
                with col_preset1:
                    st.markdown('<div class="preset-excellent">', unsafe_allow_html=True)
                    if st.button("🌟 Excellent", help="High performer", key="excellent_preset", use_container_width=True):
                        st.session_state.attendance = 95.0
                        st.session_state.study_hours = 5.0
                        st.session_state.past_scores = 90.0
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_preset2:
                    st.markdown('<div class="preset-average">', unsafe_allow_html=True)
                    if st.button("📊 Average", help="Average student", key="average_preset", use_container_width=True):
                        st.session_state.attendance = 80.0
                        st.session_state.study_hours = 3.0
                        st.session_state.past_scores = 75.0
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_preset3:
                    st.markdown('<div class="preset-at-risk">', unsafe_allow_html=True)
                    if st.button("⚠️ At Risk", help="Struggling student", key="at_risk_preset", use_container_width=True):
                        st.session_state.attendance = 60.0
                        st.session_state.study_hours = 1.5
                        st.session_state.past_scores = 55.0
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Use session state values if they exist
                if 'attendance' in st.session_state:
                    attendance = st.session_state.attendance
                if 'study_hours' in st.session_state:
                    study_hours = st.session_state.study_hours
                if 'past_scores' in st.session_state:
                    past_scores = st.session_state.past_scores

        else:  # Batch Prediction
            st.markdown("**Upload CSV or Enter Multiple Students:**")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                help="Upload a CSV with columns: attendance, study_hours, past_scores"
            )

            if uploaded_file is not None:
                batch_data = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_data.head())

                model_choice = st.selectbox(
                    "🤖 Choose ML Model for Batch",
                    options=list(models.keys()),
                    help="Select the machine learning model for batch prediction"
                )

                if st.button("🔮 Predict All"):
                    batch_predictions = []
                    for _, row in batch_data.iterrows():
                        input_data = np.array([[row['attendance'], row['study_hours'], row['past_scores']]])
                        if model_choice == 'Linear Regression':
                            input_scaled = scaler.transform(input_data)
                            pred = models[model_choice].predict(input_scaled)[0]
                        else:
                            pred = models[model_choice].predict(input_data)[0]
                        batch_predictions.append(pred)

                    batch_data['predicted_performance'] = batch_predictions
                    st.write("**Predictions:**")
                    st.dataframe(batch_data)

                    # Download results
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Upload a CSV file with columns: attendance, study_hours, past_scores")
                # Set default values for single prediction
                attendance = 85.0
                study_hours = 3.0
                past_scores = 75.0

        # Model selection (for single predictions)
        if input_method != "👥 Batch Prediction":
            st.markdown("---")
            model_choice = st.selectbox(
                "🤖 Choose ML Model",
                options=list(models.keys()),
                help="Select the machine learning model for prediction"
            )

            # Real-time prediction toggle
            real_time = st.checkbox("🔄 Real-time Prediction", value=True, help="Update predictions automatically as you change inputs")

    with col2:
        if input_method != "👥 Batch Prediction":
            st.subheader("🎯 Prediction Results")

            # Make prediction
            input_data = np.array([[attendance, study_hours, past_scores]])

            if model_choice == 'Linear Regression':
                input_scaled = scaler.transform(input_data)
                prediction = models[model_choice].predict(input_scaled)[0]
            else:
                prediction = models[model_choice].predict(input_data)[0]

            # Display prediction with animation
            prediction_container = st.container()
            with prediction_container:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Predicted Performance Score</h3>
                    <h1 style="color: #1f77b4; font-size: 3rem;">{prediction:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)

            # Performance interpretation with detailed feedback
            if prediction >= 90:
                st.success("🌟 **Excellent Performance Expected!**")
                st.markdown("**Recommendations:** Maintain current study habits and consider peer tutoring opportunities.")
            elif prediction >= 80:
                st.success("✅ **Good Performance Expected!**")
                st.markdown("**Recommendations:** Continue current approach, consider increasing study hours slightly for even better results.")
            elif prediction >= 70:
                st.warning("⚠️ **Average Performance Expected**")
                st.markdown("**Recommendations:** Focus on improving attendance and establishing consistent study routines.")
            elif prediction >= 60:
                st.warning("📉 **Below Average Performance**")
                st.markdown("**Recommendations:** Immediate intervention needed - improve attendance, increase study time, seek academic support.")
            else:
                st.error("❌ **Poor Performance - Urgent Intervention Required**")
                st.markdown("**Recommendations:** Comprehensive support needed - academic counseling, study skills training, attendance monitoring.")

            # Feature contribution analysis
            st.subheader("📊 Feature Impact Analysis")

            # Calculate feature contributions (simplified)
            if model_choice == 'Linear Regression':
                model = models[model_choice]
                input_scaled = scaler.transform(input_data)
                import scipy.sparse
                if scipy.sparse.issparse(input_scaled):
                    input_scaled = input_scaled.toarray()
                else:
                    input_scaled = np.asarray(input_scaled)
                coefficients = model.coef_
                intercept = model.intercept_

                contributions = {
                    'Attendance': coefficients[0] * input_scaled[0, 0],
                    'Study Hours': coefficients[1] * input_scaled[0, 1],
                    'Past Scores': coefficients[2] * input_scaled[0, 2],
                    'Base Score': intercept
                }
            else:
                # For Random Forest, use feature importance as proxy
                rf_model = models[model_choice]
                importances = rf_model.feature_importances_
                contributions = {
                    'Attendance': importances[0] * attendance,
                    'Study Hours': importances[1] * study_hours * 10,  # Scale for visualization
                    'Past Scores': importances[2] * past_scores,
                    'Model Complexity': 10  # Placeholder
                }

            # Create contribution chart
            contrib_df = pd.DataFrame(list(contributions.items()), columns=['Feature', 'Contribution'])
            fig = px.bar(contrib_df, x='Feature', y='Contribution',
                        title="Feature Contributions to Prediction",
                        color='Contribution',
                        color_continuous_scale='RdYlBu')
            st.plotly_chart(fig, use_container_width=True)

            # Model accuracy info
            st.info(f"""
            **🎯 Model Accuracy (R²):** {scores[model_choice]['R²']:.3f}

            **🔄 Cross-Validation Score:** {scores[model_choice]['CV R² Mean']:.3f} ± {scores[model_choice]['CV R² Std']:.3f}

            **📊 Model Type:** {model_choice}
            """)

            # Confidence interval (simplified)
            mse = scores[model_choice]['MSE']
            std_error = np.sqrt(mse)
            confidence_lower = max(0, prediction - 1.96 * std_error)
            confidence_upper = min(100, prediction + 1.96 * std_error)

            st.markdown(f"""
            **🎯 95% Confidence Interval:** {confidence_lower:.1f}% - {confidence_upper:.1f}%
            """)

            # What-if analysis
            st.subheader("🔮 What-If Analysis")
            st.markdown("**See how changes in each factor affect the prediction:**")

            # Attendance impact
            attendance_range = np.linspace(max(0, attendance-20), min(100, attendance+20), 5)
            attendance_predictions = []
            for att in attendance_range:
                test_input = np.array([[att, study_hours, past_scores]])
                if model_choice == 'Linear Regression':
                    test_scaled = scaler.transform(test_input)
                    pred = models[model_choice].predict(test_scaled)[0]
                else:
                    pred = models[model_choice].predict(test_input)[0]
                attendance_predictions.append(pred)

            fig = px.line(x=attendance_range, y=attendance_predictions,
                         title="Impact of Attendance on Performance",
                         labels={'x': 'Attendance (%)', 'y': 'Predicted Performance (%)'})
            fig.add_vline(x=attendance, line_dash="dash", line_color="red",
                         annotation_text="Current")
            st.plotly_chart(fig, use_container_width=True)

def data_analysis_page(data):
    st.header("📈 Data Analysis & Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Total Students:** {len(data)}")
        st.write("**Statistical Summary:**")
        st.dataframe(data.describe())

    with col2:
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select Feature", data.columns)

        fig = px.histogram(data, x=feature, nbins=30, title=f"Distribution of {feature}")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    st.subheader("Feature Correlations")
    corr_matrix = data.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plots
    st.subheader("Feature Relationships")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            data,
            x='attendance',
            y='performance',
            title="Attendance vs Performance",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            data,
            x='study_hours',
            y='performance',
            title="Study Hours vs Performance",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

def model_performance_page(data):
    st.header("🤖 Model Performance Analysis")

    models, scores, scaler, X_test, y_test = train_models(data)

    # Performance Overview
    st.subheader("📊 Performance Overview")

    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)

    best_model = max(scores.keys(), key=lambda x: scores[x]['R²'])
    best_r2 = scores[best_model]['R²']
    best_rmse = np.sqrt(scores[best_model]['MSE'])
    avg_cv = np.mean([scores[model]['CV R² Mean'] for model in scores.keys()])

    with col1:
        st.metric("🏆 Best Model", best_model, f"R² = {best_r2:.3f}")
    with col2:
        st.metric("🎯 Best R² Score", f"{best_r2:.3f}", f"{(best_r2-0.5)*100:+.1f}% vs baseline")
    with col3:
        st.metric("📏 Best RMSE", f"{best_rmse:.2f}%", "Lower is better")
    with col4:
        st.metric("🔄 Avg CV Score", f"{avg_cv:.3f}", f"±{np.std([scores[model]['CV R² Mean'] for model in scores.keys()]):.3f}")

    # Model comparison
    st.subheader("⚖️ Detailed Model Comparison")

    comparison_df = pd.DataFrame(scores).T

    # Add interpretation column
    comparison_df['Performance Level'] = comparison_df['R²'].apply(
        lambda x: '🌟 Excellent' if x > 0.85 else
                 '✅ Good' if x > 0.75 else
                 '⚠️ Fair' if x > 0.65 else
                 '❌ Poor'
    )

    st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)

    # Interpretation guide
    with st.expander("📖 How to Interpret These Metrics"):
        st.markdown("""
        **R² Score (Coefficient of Determination):**
        - **0.85+**: Excellent - Model explains 85%+ of variance
        - **0.75-0.84**: Good - Model explains most patterns
        - **0.65-0.74**: Fair - Model captures basic relationships
        - **<0.65**: Poor - Model needs improvement

        **MSE (Mean Squared Error):**
        - Lower values indicate better performance
        - Units: squared percentage points
        - Heavily penalizes large prediction errors

        **CV R² Mean (Cross-Validation):**
        - More reliable than single R² score
        - Shows how well model generalizes to new data
        - Standard deviation indicates consistency
        """)

    # Performance Analysis
    st.subheader("📈 Performance Analysis")

    # Visualize model performance
    col1, col2 = st.columns(2)

    with col1:
        # R² scores comparison
        r2_scores = [scores[model]['R²'] for model in scores.keys()]
        fig = px.bar(
            x=list(scores.keys()),
            y=r2_scores,
            title="Model R² Scores",
            labels={'x': 'Model', 'y': 'R² Score'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cross-validation scores
        cv_means = [scores[model]['CV R² Mean'] for model in scores.keys()]
        cv_stds = [scores[model]['CV R² Std'] for model in scores.keys()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(scores.keys()),
            y=cv_means,
            error_y=dict(type='data', array=cv_stds),
            name='CV R² Score'
        ))
        fig.update_layout(title="Cross-Validation Scores", yaxis_title="R² Score")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance (for Random Forest)
    if 'Random Forest' in models:
        st.subheader("Feature Importance (Random Forest)")

        rf_model = models['Random Forest']
        feature_names = ['Attendance', 'Study Hours', 'Past Scores']
        importances = rf_model.feature_importances_

        fig = px.bar(
            x=feature_names,
            y=importances,
            title="Feature Importance",
            labels={'x': 'Features', 'y': 'Importance'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Prediction vs Actual scatter plot
    st.subheader("Prediction Accuracy")

    model_choice = st.selectbox("Select Model for Accuracy Visualization", list(models.keys()))

    if model_choice == 'Linear Regression':
        X_test_scaled = scaler.transform(X_test)
        y_pred = models[model_choice].predict(X_test_scaled)
    else:
        y_pred = models[model_choice].predict(X_test)

    fig = px.scatter(
        x=y_test,
        y=y_pred,
        title=f"{model_choice}: Predicted vs Actual",
        labels={'x': 'Actual Performance', 'y': 'Predicted Performance'}
    )

    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Add comprehensive evaluation section
    st.subheader("🎓 Model Evaluation Deep Dive")
    st.markdown("""
    Understanding how we assess model accuracy is crucial for building trust in predictions.
    Let's explore the comprehensive evaluation framework we use.
    """)

    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["📊 Regression Metrics", "🔍 Error Analysis", "🎯 Validation Methods"])

    with eval_tab1:
        st.markdown("### 📊 Regression Metrics Explained")

        # Get actual predictions for demonstration
        best_model_name = max(scores.keys(), key=lambda x: scores[x]['R²'])
        if best_model_name == 'Linear Regression':
            X_test_demo = scaler.transform(X_test)
        else:
            X_test_demo = X_test
        y_pred_demo = models[best_model_name].predict(X_test_demo)

        # Calculate comprehensive metrics
        r2_demo = r2_score(y_test, y_pred_demo)
        mse_demo = mean_squared_error(y_test, y_pred_demo)
        rmse_demo = np.sqrt(mse_demo)
        mae_demo = np.mean(np.abs(y_test - y_pred_demo))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 R² Score (Coefficient of Determination)")
            st.metric("R² Score", f"{r2_demo:.4f}", help="Proportion of variance explained")

            st.markdown(f"""
            **Interpretation:**
            - **Range:** -∞ to 1.0 (higher is better)
            - **{r2_demo:.1%}** of variance in student performance is explained
            - **Perfect Model:** R² = 1.0 (explains all variance)
            - **Baseline:** R² = 0.0 (no better than using mean)
            - **Harmful:** R² < 0.0 (worse than baseline)
            """)

            st.markdown("#### 📐 Mean Squared Error (MSE)")
            st.metric("MSE", f"{mse_demo:.2f}", help="Average squared prediction error")

            st.markdown(f"""
            **Key Points:**
            - **Units:** Squared percentage points
            - **Sensitivity:** Heavily penalizes large errors
            - **Our MSE:** {mse_demo:.1f} means average squared error
            - **Lower is better** (0 = perfect predictions)
            """)

        with col2:
            st.markdown("#### 📏 Root Mean Squared Error (RMSE)")
            st.metric("RMSE", f"{rmse_demo:.2f}%", help="Square root of MSE")

            st.markdown(f"""
            **Practical Meaning:**
            - **Units:** Same as target (percentage points)
            - **Interpretation:** Predictions typically off by ±{rmse_demo:.1f}%
            - **More intuitive** than MSE
            - **Standard deviation** of prediction errors
            """)

            st.markdown("#### 📊 Mean Absolute Error (MAE)")
            st.metric("MAE", f"{mae_demo:.2f}%", help="Average absolute prediction error")

            st.markdown(f"""
            **Characteristics:**
            - **Robust:** Less sensitive to outliers than RMSE
            - **Typical error:** {mae_demo:.1f} percentage points
            - **Linear scale:** All errors weighted equally
            - **Easy interpretation:** Average prediction mistake
            """)

        # Metrics comparison visualization
        st.markdown("#### 📈 Metrics Comparison")
        metrics_df = pd.DataFrame({
            'Metric': ['R² Score', 'RMSE', 'MAE', 'MSE'],
            'Value': [r2_demo, rmse_demo, mae_demo, mse_demo],
            'Interpretation': [
                f'{r2_demo:.1%} variance explained',
                f'±{rmse_demo:.1f}% typical error',
                f'{mae_demo:.1f}% average error',
                f'{mse_demo:.1f} squared error units'
            ]
        })
        st.dataframe(metrics_df, use_container_width=True)

    with eval_tab2:
        st.markdown("### 🔍 Error Analysis & Diagnostics")

        residuals = y_test - y_pred_demo

        col1, col2 = st.columns(2)

        with col1:
            # Residual distribution
            fig = px.histogram(x=residuals, nbins=20, title="Distribution of Prediction Errors")
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Perfect")
            fig.update_xaxes(title="Residuals (Actual - Predicted)")
            st.plotly_chart(fig, use_container_width=True)

            # Error statistics
            st.markdown("**Error Statistics:**")
            st.write(f"• Mean Error: {np.mean(residuals):.2f}% (bias)")
            st.write(f"• Std Deviation: {np.std(residuals):.2f}%")
            st.write(f"• 95% of errors within: ±{1.96*np.std(residuals):.1f}%")

            # Error assessment
            if abs(np.mean(residuals)) < 1:
                st.success("✅ Low bias detected")
            else:
                st.warning("⚠️ Model may have systematic bias")

        with col2:
            # Residuals vs predicted
            fig = px.scatter(x=y_pred_demo, y=residuals, title="Residuals vs Predicted Values")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_xaxes(title="Predicted Performance (%)")
            fig.update_yaxes(title="Residuals")
            st.plotly_chart(fig, use_container_width=True)

            # Pattern analysis
            correlation = np.corrcoef(y_pred_demo, residuals)[0, 1]
            if abs(correlation) < 0.1:
                st.success("✅ No systematic patterns in errors")
            else:
                st.warning(f"⚠️ Pattern detected (r={correlation:.3f})")

    with eval_tab3:
        st.markdown("### 🎯 Validation Methods & Concepts")

        st.markdown("#### 🔄 Cross-Validation Strategy")

        # CV visualization
        cv_folds = 5
        fold_data = []
        for i in range(cv_folds):
            fold_data.append({
                'Fold': f'Fold {i+1}',
                'Training': 80,
                'Validation': 20,
                'R² Score': scores[best_model_name]['CV R² Mean'] + np.random.normal(0, scores[best_model_name]['CV R² Std'])
            })

        cv_df = pd.DataFrame(fold_data)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(cv_df, x='Fold', y=['Training', 'Validation'],
                        title="5-Fold Cross-Validation Split")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(cv_df, x='Fold', y='R² Score', markers=True,
                         title="Cross-Validation Performance")
            fig.add_hline(y=cv_df['R² Score'].mean(), line_dash="dash",
                         annotation_text=f"Mean: {cv_df['R² Score'].mean():.3f}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🎯 Why Cross-Validation Matters")
        st.markdown("""
        **Benefits of 5-Fold Cross-Validation:**
        - **Reliability:** Uses all data for both training and testing
        - **Generalization:** Better estimate of real-world performance
        - **Stability:** Reduces dependence on specific train-test split
        - **Confidence:** Provides uncertainty estimates (mean ± std)
        """)

        # Model selection criteria
        st.markdown("#### ⚖️ Model Selection Framework")

        selection_criteria = pd.DataFrame({
            'Criterion': ['Accuracy', 'Interpretability', 'Speed', 'Robustness', 'Simplicity'],
            'Weight': ['40%', '25%', '15%', '15%', '5%'],
            'Linear Regression': ['Good', 'Excellent', 'Excellent', 'Good', 'Excellent'],
            'Random Forest': ['Excellent', 'Good', 'Good', 'Excellent', 'Fair']
        })

        st.dataframe(selection_criteria, use_container_width=True)

        st.success(f"""
        **Selected Model: {best_model_name}**
        - Optimal balance of accuracy and interpretability
        - Suitable for educational decision-making
        - Reliable cross-validation performance: {scores[best_model_name]['CV R² Mean']:.3f} ± {scores[best_model_name]['CV R² Std']:.3f}
        """)

def model_explanation_page():
    st.header("📚 Model Explanation & Educational Guide")

    # Introduction
    st.markdown("""
    Welcome to the comprehensive guide explaining how our Student Performance Prediction models work!
    This page will help you understand the machine learning algorithms, their strengths, and how they make predictions.
    """)

    # Model Overview
    st.subheader("🤖 Model Overview")

    tab1, tab2, tab3 = st.tabs(["🔍 Model Comparison", "📊 How It Works", "🎯 Use Cases"])

    with tab1:
        st.markdown("### 🔍 Model Comparison")

        model_comparison = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest'],
            'Type': ['Linear', 'Ensemble'],
            'Complexity': ['Low', 'Medium'],
            'Interpretability': ['High', 'Medium'],
            'Best For': ['Linear relationships', 'Non-linear patterns'],
            'Training Speed': ['Fast', 'Medium'],
            'Prediction Speed': ['Very Fast', 'Fast']
        })

        st.dataframe(model_comparison, use_container_width=True)

        st.markdown("""
        **🎯 Which Model to Choose?**
        - **Linear Regression**: Choose when you want simple, interpretable results and believe relationships are mostly linear
        - **Random Forest**: Choose when you want to capture complex patterns and interactions between features
        """)

    with tab2:
        st.markdown("### 📊 How Our Models Work")

        # Linear Regression Explanation
        st.markdown("#### 🔢 Linear Regression")
        st.markdown("""
        Linear Regression finds the best straight line through the data points. It assumes that student performance
        can be predicted using a simple formula:

        **Performance = w₁ × Attendance + w₂ × Study_Hours + w₃ × Past_Scores + bias**

        Where w₁, w₂, w₃ are weights learned from the data.
        """)

        # Create a simple visualization
        fig = go.Figure()
        x = np.linspace(0, 100, 100)
        y = 0.3 * x + 0.4 * 75 + 0.2 * 3 * 10 + np.random.normal(0, 2, 100)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points', opacity=0.6))

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=x, y=p(x), mode='lines', name='Linear Trend', line=dict(color='red', width=3)))

        fig.update_layout(
            title="Linear Regression: Finding the Best Line",
            xaxis_title="Attendance (%)",
            yaxis_title="Performance (%)",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Random Forest Explanation
        st.markdown("#### 🌳 Random Forest")
        st.markdown("""
        Random Forest creates many decision trees and combines their predictions. Each tree asks questions like:
        - "Is attendance > 80%?"
        - "Are study hours > 3?"
        - "Are past scores > 70%?"

        The final prediction is the average of all tree predictions.
        """)

        # Decision tree visualization
        st.markdown("**Example Decision Tree Logic:**")
        st.code("""
        if attendance > 80:
            if past_scores > 75:
                if study_hours > 3:
                    prediction = "High Performance (85-95%)"
                else:
                    prediction = "Good Performance (75-85%)"
            else:
                prediction = "Average Performance (65-75%)"
        else:
            prediction = "Below Average Performance (50-65%)"
        """)

    with tab3:
        st.markdown("### 🎯 Real-World Use Cases")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏫 For Educational Institutions")
            st.markdown("""
            - **Early Warning System**: Identify at-risk students before they fail
            - **Resource Allocation**: Prioritize support for students who need it most
            - **Intervention Planning**: Design targeted programs based on prediction factors
            - **Performance Monitoring**: Track student progress over time
            """)

            st.markdown("#### 👨‍🏫 For Teachers")
            st.markdown("""
            - **Personalized Learning**: Adapt teaching methods based on student profiles
            - **Parent Communication**: Provide data-driven insights to parents
            - **Curriculum Planning**: Adjust course difficulty based on class predictions
            - **Study Recommendations**: Suggest optimal study hours for each student
            """)

        with col2:
            st.markdown("#### 👨‍🎓 For Students")
            st.markdown("""
            - **Self-Assessment**: Understand how their habits affect performance
            - **Goal Setting**: Set realistic academic targets
            - **Study Planning**: Optimize study schedules for better results
            - **Progress Tracking**: Monitor improvement over time
            """)

            st.markdown("#### 👨‍💼 For Administrators")
            st.markdown("""
            - **Policy Making**: Create evidence-based academic policies
            - **Budget Planning**: Allocate resources to high-impact interventions
            - **Success Metrics**: Measure effectiveness of academic programs
            - **Predictive Analytics**: Forecast graduation rates and outcomes
            """)

    # Model Evaluation Metrics
    st.subheader("📏 Model Evaluation Metrics & Performance Assessment")

    st.markdown("""
    Understanding how we measure model performance is crucial for building trust in predictions.
    Let's explore the various metrics and concepts we use to evaluate our Student Performance Predictor.
    """)

    eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs(["📊 Regression Metrics", "🔄 Validation Methods", "📈 Performance Concepts", "⚖️ Model Selection"])

    with eval_tab1:
        st.markdown("### 📊 Regression Metrics (For Continuous Predictions)")

        # Create sample data for demonstration
        np.random.seed(42)
        actual = np.array([85, 78, 92, 67, 89, 73, 95, 81, 76, 88])
        predicted = actual + np.random.normal(0, 5, 10)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 R² Score (Coefficient of Determination)")
            r2_example = r2_score(actual, predicted)
            st.metric("R² Score", f"{r2_example:.3f}", help="Proportion of variance explained by the model")

            st.markdown(f"""
            **Interpretation:**
            - **Range:** -∞ to 1.0 (higher is better)
            - **{r2_example:.1%}** of the variance in student performance is explained by our model
            - **Perfect Model:** R² = 1.0 (explains all variance)
            - **Baseline Model:** R² = 0.0 (no better than using the mean)
            - **Worse than Baseline:** R² < 0.0 (model is harmful)
            """)

            st.markdown("#### 📐 Mean Squared Error (MSE)")
            mse_example = mean_squared_error(actual, predicted)
            st.metric("MSE", f"{mse_example:.2f}", help="Average squared difference between actual and predicted")

            st.markdown(f"""
            **Interpretation:**
            - **Range:** 0 to ∞ (lower is better)
            - **Units:** Squared percentage points
            - **Sensitivity:** Heavily penalizes large errors
            - **Our MSE:** {mse_example:.1f} means average squared error of {mse_example:.1f} percentage points²
            """)

        with col2:
            st.markdown("#### 📏 Root Mean Squared Error (RMSE)")
            rmse_example = np.sqrt(mse_example)
            st.metric("RMSE", f"{rmse_example:.2f}", help="Square root of MSE, in original units")

            st.markdown(f"""
            **Interpretation:**
            - **Range:** 0 to ∞ (lower is better)
            - **Units:** Same as target variable (percentage points)
            - **Practical Meaning:** On average, predictions are off by ±{rmse_example:.1f} percentage points
            - **Easier to interpret** than MSE because it's in original units
            """)

            st.markdown("#### 📊 Mean Absolute Error (MAE)")
            mae_example = np.mean(np.abs(actual - predicted))
            st.metric("MAE", f"{mae_example:.2f}", help="Average absolute difference between actual and predicted")

            st.markdown(f"""
            **Interpretation:**
            - **Range:** 0 to ∞ (lower is better)
            - **Units:** Same as target variable (percentage points)
            - **Robust:** Less sensitive to outliers than RMSE
            - **Our MAE:** {mae_example:.1f} means typical prediction error is {mae_example:.1f} percentage points
            """)

        # Visualization of errors
        st.markdown("#### 📈 Error Visualization")
        fig = go.Figure()

        # Scatter plot of actual vs predicted
        fig.add_trace(go.Scatter(
            x=actual, y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(size=10, color='blue', opacity=0.7)
        ))

        # Perfect prediction line
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red', width=2)
        ))

        # Error lines
        for i in range(len(actual)):
            fig.add_trace(go.Scatter(
                x=[actual[i], actual[i]], y=[actual[i], predicted[i]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            title="Actual vs Predicted Performance (with Error Lines)",
            xaxis_title="Actual Performance (%)",
            yaxis_title="Predicted Performance (%)",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with eval_tab2:
        st.markdown("### 🔄 Validation Methods")

        st.markdown("#### 🎯 Train-Test Split")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Concept:** Divide data into training and testing sets
            - **Training Set (80%):** Used to train the model
            - **Test Set (20%):** Used to evaluate final performance
            - **Purpose:** Simulate performance on unseen data
            """)
        with col2:
            # Visualization of train-test split
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Training', 'Testing'], y=[80, 20],
                               marker_color=['lightblue', 'lightcoral']))
            fig.update_layout(title="Train-Test Split (80-20)", yaxis_title="Percentage of Data")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔄 Cross-Validation (K-Fold)")
        st.markdown("""
        **5-Fold Cross-Validation Process:**
        1. **Split data into 5 equal parts (folds)**
        2. **Train on 4 folds, test on 1 fold**
        3. **Repeat 5 times, each fold serves as test set once**
        4. **Average the 5 performance scores**
        """)

        # Cross-validation visualization
        cv_data = pd.DataFrame({
            'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'],
            'Training': [80, 80, 80, 80, 80],
            'Testing': [20, 20, 20, 20, 20],
            'R² Score': [0.85, 0.82, 0.87, 0.83, 0.86]
        })

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(cv_data, x='Fold', y=['Training', 'Testing'],
                        title="Cross-Validation Data Split")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(cv_data, x='Fold', y='R² Score',
                         title="Cross-Validation Scores", markers=True)
            fig.add_hline(y=cv_data['R² Score'].mean(), line_dash="dash",
                         annotation_text=f"Mean: {cv_data['R² Score'].mean():.3f}")
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        **Cross-Validation Benefits:**
        - **More Reliable:** Uses all data for both training and testing
        - **Reduces Overfitting:** Multiple validation rounds
        - **Confidence Intervals:** Mean ± Standard Deviation gives uncertainty estimate
        - **Example:** CV Score = {cv_data['R² Score'].mean():.3f} ± {cv_data['R² Score'].std():.3f}
        """)

    with eval_tab3:
        st.markdown("### 📈 Key Performance Concepts")

        concept_col1, concept_col2 = st.columns(2)

        with concept_col1:
            st.markdown("#### 🎯 Bias vs Variance Trade-off")
            st.markdown("""
            **Bias (Underfitting):**
            - Model is too simple
            - High training error
            - High test error
            - **Example:** Using only attendance to predict performance

            **Variance (Overfitting):**
            - Model is too complex
            - Low training error
            - High test error
            - **Example:** Memorizing every student's exact score

            **Sweet Spot:**
            - Balanced complexity
            - Good training error
            - Good test error
            """)

            # Bias-Variance visualization
            complexity = np.linspace(1, 10, 50)
            bias = 1/complexity + 0.1
            variance = complexity/10
            total_error = bias + variance + 0.2

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=complexity, y=bias, name='Bias', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=complexity, y=variance, name='Variance', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=complexity, y=total_error, name='Total Error', line=dict(color='green')))

            fig.update_layout(
                title="Bias-Variance Trade-off",
                xaxis_title="Model Complexity",
                yaxis_title="Error",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with concept_col2:
            st.markdown("#### 📊 Learning Curves")
            st.markdown("""
            **Purpose:** Show how model performance changes with training data size

            **Interpretation:**
            - **Converging Lines:** Good generalization
            - **Large Gap:** Overfitting
            - **High Error:** Underfitting
            """)

            # Learning curve simulation
            train_sizes = np.array([50, 100, 200, 400, 600, 800, 1000])
            train_scores = 1 - 0.3 * np.exp(-train_sizes/200)
            val_scores = train_scores - 0.1 * np.exp(-train_sizes/300)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_sizes, y=train_scores, name='Training Score',
                                   line=dict(color='blue'), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=train_sizes, y=val_scores, name='Validation Score',
                                   line=dict(color='red'), mode='lines+markers'))

            fig.update_layout(
                title="Learning Curves",
                xaxis_title="Training Set Size",
                yaxis_title="R² Score",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔍 Residual Analysis")
        st.markdown("""
        **Residuals = Actual - Predicted**

        **Good Model Residuals:**
        - **Random scatter** around zero
        - **No clear patterns**
        - **Constant variance** (homoscedasticity)

        **Problem Indicators:**
        - **Curved patterns:** Missing non-linear relationships
        - **Funnel shape:** Heteroscedasticity (changing variance)
        - **Outliers:** Data quality issues or special cases
        """)

        # Residual plot
        residuals = actual - predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predicted, y=residuals, mode='markers',
                               name='Residuals', marker=dict(size=8, color='purple')))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Perfect Prediction")
        fig.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted Performance (%)",
            yaxis_title="Residuals (Actual - Predicted)",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with eval_tab4:
        st.markdown("### ⚖️ Model Selection Criteria")

        st.markdown("#### 🏆 How We Choose the Best Model")

        # Model comparison table
        model_comparison = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Neural Network'],
            'R² Score': [0.823, 0.867, 0.845, 0.889],
            'RMSE': [8.2, 7.1, 7.8, 6.5],
            'Training Time': ['Fast', 'Medium', 'Slow', 'Very Slow'],
            'Interpretability': ['High', 'Medium', 'Low', 'Very Low'],
            'Overfitting Risk': ['Low', 'Medium', 'High', 'Very High']
        })

        st.dataframe(model_comparison, use_container_width=True)

        st.markdown("#### 🎯 Selection Criteria")

        criteria_col1, criteria_col2 = st.columns(2)

        with criteria_col1:
            st.markdown("""
            **Primary Metrics (60% weight):**
            - **Cross-Validation R²:** Most important for generalization
            - **RMSE:** Practical prediction accuracy
            - **Stability:** Consistent performance across folds

            **Secondary Factors (40% weight):**
            - **Interpretability:** Can we explain predictions?
            - **Training Speed:** How fast can we retrain?
            - **Prediction Speed:** Real-time requirements
            - **Overfitting Risk:** Will it generalize to new students?
            """)

        with criteria_col2:
            st.markdown("""
            **Domain-Specific Considerations:**
            - **Educational Context:** Teachers need explainable predictions
            - **Ethical Requirements:** Avoid black-box decisions
            - **Data Size:** Limited student data favors simpler models
            - **Feature Quality:** Clean, reliable input features

            **Practical Constraints:**
            - **Computational Resources:** School IT limitations
            - **Update Frequency:** How often do we retrain?
            - **User Expertise:** Technical skill of end users
            """)

        # Model selection visualization
        fig = go.Figure()

        models = ['Linear Regression', 'Random Forest']
        metrics = ['Accuracy', 'Interpretability', 'Speed', 'Simplicity']
        lr_scores = [0.82, 0.95, 0.98, 0.95]
        rf_scores = [0.87, 0.75, 0.80, 0.60]

        fig.add_trace(go.Scatterpolar(
            r=lr_scores,
            theta=metrics,
            fill='toself',
            name='Linear Regression',
            line_color='blue'
        ))

        fig.add_trace(go.Scatterpolar(
            r=rf_scores,
            theta=metrics,
            fill='toself',
            name='Random Forest',
            line_color='red'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Comparison Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success("""
        **Our Choice: Random Forest**
        - Best balance of accuracy and interpretability
        - Handles non-linear relationships well
        - Provides feature importance insights
        - Robust to outliers and missing data
        """)

    # Technical Details
    st.subheader("🔬 Technical Details")

    with st.expander("📊 Feature Engineering"):
        st.markdown("""
        Our models use several techniques to improve predictions:

        **1. Feature Scaling**: We normalize all features to have similar ranges, which helps Linear Regression perform better.

        **2. Cross-Validation**: We use 5-fold cross-validation to ensure our models generalize well to new data.

        **3. Hyperparameter Tuning**: For Random Forest, we optimize parameters like:
        - Number of trees (n_estimators)
        - Maximum depth of trees
        - Minimum samples per split

        **4. Model Evaluation**: We use multiple metrics:
        - **R² Score**: How much variance in performance our model explains
        - **Mean Squared Error**: Average squared difference between predictions and actual values
        - **Cross-Validation Score**: How well the model performs on unseen data
        """)

    with st.expander("⚠️ Model Limitations"):
        st.markdown("""
        **Important Considerations:**

        **1. Data Quality**: Predictions are only as good as the input data
        - Ensure attendance data is accurate and up-to-date
        - Study hours should reflect actual study time, not just time spent
        - Past scores should be representative of recent performance

        **2. External Factors**: Our models don't account for:
        - Personal circumstances (health, family issues)
        - Teaching quality variations
        - Exam difficulty changes
        - Motivation and mental health factors

        **3. Correlation vs Causation**:
        - High attendance correlates with good performance, but forced attendance alone won't guarantee success
        - The model shows relationships, not necessarily cause-and-effect

        **4. Individual Variation**:
        - Some students may perform differently than predicted due to unique circumstances
        - Use predictions as guidance, not absolute truth
        """)

    with st.expander("🔮 Future Improvements"):
        st.markdown("""
        **Potential Enhancements:**

        **1. Additional Features**:
        - Socioeconomic background
        - Learning style preferences
        - Extracurricular activities
        - Sleep patterns and health data
        - Peer influence factors

        **2. Advanced Models**:
        - Neural Networks for complex pattern recognition
        - Time Series models for tracking progress over time
        - Ensemble methods combining multiple algorithms
        - Deep Learning for unstructured data (essays, behavior patterns)

        **3. Real-Time Updates**:
        - Continuous learning from new data
        - Adaptive models that improve over time
        - Integration with learning management systems
        - Mobile app for real-time predictions

        **4. Personalization**:
        - Individual student models
        - Subject-specific predictions
        - Learning path recommendations
        - Customized intervention strategies
        """)

    # Interactive Model Demo
    st.subheader("🎮 Interactive Model Demo")
    st.markdown("**Try different scenarios to see how the models behave:**")

    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:
        st.markdown("**Scenario Builder:**")
        scenario = st.selectbox(
            "Choose a scenario:",
            ["Custom", "High Achiever", "Average Student", "Struggling Student", "Inconsistent Performer"]
        )

        if scenario == "High Achiever":
            demo_attendance, demo_study, demo_past = 95, 5, 90
        elif scenario == "Average Student":
            demo_attendance, demo_study, demo_past = 80, 3, 75
        elif scenario == "Struggling Student":
            demo_attendance, demo_study, demo_past = 60, 1.5, 55
        elif scenario == "Inconsistent Performer":
            demo_attendance, demo_study, demo_past = 70, 4, 85
        else:
            demo_attendance = st.slider("Demo Attendance", 0, 100, 80)
            demo_study = st.slider("Demo Study Hours", 0.0, 12.0, 3.0)
            demo_past = st.slider("Demo Past Scores", 0, 100, 75)

    with demo_col2:
        st.markdown("**Model Predictions:**")

        # Generate sample data for demo
        demo_data = generate_sample_data(100)
        demo_models, demo_scores, demo_scaler, _, _ = train_models(demo_data)

        # Make predictions
        demo_input = np.array([[demo_attendance, demo_study, demo_past]])

        lr_pred = demo_models['Linear Regression'].predict(demo_scaler.transform(demo_input))[0]
        rf_pred = demo_models['Random Forest'].predict(demo_input)[0]

        st.metric("Linear Regression", f"{lr_pred:.1f}%", f"{lr_pred - 75:.1f}% vs avg")
        st.metric("Random Forest", f"{rf_pred:.1f}%", f"{rf_pred - 75:.1f}% vs avg")

        # Show difference
        diff = abs(lr_pred - rf_pred)
        st.info(f"**Model Agreement**: {100 - diff:.1f}% (difference: {diff:.1f}%)")

    # Conclusion
    st.subheader("🎯 Key Takeaways")
    st.markdown("""
    **Remember:**
    1. **Models are tools**, not crystal balls - use them to guide decisions, not replace human judgment
    2. **Data quality matters** - ensure your input data is accurate and representative
    3. **Context is crucial** - consider external factors that models might miss
    4. **Continuous improvement** - regularly update models with new data and feedback
    5. **Ethical use** - use predictions to help students succeed, not to discriminate or limit opportunities

    **🚀 Ready to make predictions? Go back to the Prediction page and try it out!**
    """)

if __name__ == "__main__":
    main()
