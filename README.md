# 🎓 Student Performance Predictor

A comprehensive machine learning web application built with **Streamlit**, **Scikit-learn**, and **Python** to predict student academic performance based on various factors including attendance, study hours, and past academic scores.

## 🌟 Features

- **Interactive Web Interface**: Built with Streamlit for real-time predictions
- **Multiple ML Models**: Implements and compares various regression models
- **Advanced Analytics**: Comprehensive data analysis and visualization
- **Feature Engineering**: Includes interaction terms and polynomial features
- **Model Optimization**: Uses cross-validation and hyperparameter tuning
- **Real-time Predictions**: Interactive UI for immediate performance predictions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Student-Performance-Predictor-.git
cd Student-Performance-Predictor-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📊 Application Overview

### Main Features

1. **🔮 Enhanced Prediction Page**:
   - **Multiple Input Methods**: Interactive sliders, manual number input, or batch CSV upload
   - **Real-time Predictions**: Instant updates as you change inputs
   - **Quick Presets**: One-click scenarios (Excellent, Average, At-Risk students)
   - **What-If Analysis**: See how changes in each factor affect predictions
   - **Feature Impact Analysis**: Understand which factors contribute most to the prediction
   - **Confidence Intervals**: Get prediction ranges with uncertainty estimates
   - **Detailed Recommendations**: Actionable advice based on prediction results

2. **📊 Interactive Data Analysis Page**:
   - Explore dataset statistics and distributions
   - View feature correlations with interactive heatmaps
   - Dynamic visualizations with Plotly
   - Real-time filtering and exploration

3. **🤖 Advanced Model Performance Page**:
   - Compare multiple ML models side-by-side
   - Cross-validation scores with error bars
   - Feature importance analysis for Random Forest
   - Prediction vs Actual scatter plots with trend lines
   - Model accuracy metrics and interpretations

4. **📚 Comprehensive Model Explanation Page**:
   - **Educational Content**: Learn how each ML algorithm works
   - **Interactive Demos**: Try different scenarios and see model behavior
   - **Technical Details**: Feature engineering, limitations, and best practices
   - **Real-World Use Cases**: Applications for schools, teachers, and students
   - **Future Improvements**: Roadmap for enhanced capabilities

### Machine Learning Models

- **Linear Regression**: Baseline model with feature scaling
- **Ridge Regression**: Regularized linear model
- **Lasso Regression**: Feature selection through L1 regularization
- **Random Forest**: Ensemble method for non-linear relationships
- **Gradient Boosting**: Advanced ensemble technique
- **Support Vector Regression**: Kernel-based regression

## 🔧 Project Structure

```
Student-Performance-Predictor-/
├── app.py                    # Enhanced Streamlit application with 4 pages
├── model_trainer.py          # Advanced model training module
├── data_generator.py         # Synthetic data generation
├── test_app.py              # Comprehensive test suite
├── requirements.txt         # Python dependencies
├── sample_students.csv      # Sample data for batch testing
├── README.md               # Project documentation
├── MODEL_EVALUATION_GUIDE.md # Comprehensive evaluation methodology
└── models/                 # Saved model files (created after training)
```

## 📈 Model Performance

The application uses multiple evaluation metrics:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-Validation**: 5-fold CV for robust evaluation

### Feature Engineering

The model includes several engineered features:
- Attendance-Study Hours Interaction
- Past Scores Squared (non-linear relationship)
- Study Efficiency (past scores per study hour)

## 🎯 Use Cases

- **Educational Institutions**: Identify at-risk students early
- **Academic Analytics**: Understand factors affecting performance
- **Student Counseling**: Data-driven intervention strategies
- **Research**: Academic performance modeling and analysis

## 🛠️ Advanced Usage

### Training Custom Models

```python
from model_trainer import StudentPerformancePredictor

# Initialize predictor
predictor = StudentPerformancePredictor()

# Generate or load your data
data = predictor.generate_enhanced_dataset(n_samples=2000)

# Train models
results, X_test, y_test = predictor.train_models(data)

# Save trained model
predictor.save_model('my_model.pkl')
```

### Generating Custom Datasets

```python
from data_generator import StudentDataGenerator

generator = StudentDataGenerator()

# Generate basic dataset
basic_data = generator.generate_basic_dataset(1000)

# Generate advanced dataset with additional features
advanced_data = generator.generate_advanced_dataset(1500)

# Create class scenarios
class_data = generator.create_class_scenarios()
```

## 📊 Data Features

### Primary Features
- **Attendance Rate** (0-100%): Student's class attendance percentage
- **Study Hours** (0-12 hours): Daily study time
- **Past Academic Scores** (0-100%): Historical academic performance

### Advanced Features (in model_trainer.py)
- Socioeconomic status
- Parental education level
- Extracurricular activities
- Sleep hours
- Screen time
- Stress level
- Family support

## 🔍 Model Insights

### Key Findings
1. **Past academic scores** are the strongest predictor (40% weight)
2. **Attendance** significantly impacts performance (30% weight)
3. **Study hours** show diminishing returns after 8 hours/day
4. **Interaction effects** between features improve prediction accuracy

### Performance Benchmarks
- Best model typically achieves R² > 0.85
- RMSE usually < 8 points on 100-point scale
- Cross-validation ensures robust performance

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
The application can be deployed on:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Docker containers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** for machine learning algorithms
- **Streamlit** for the interactive web framework
- **Plotly** for interactive visualizations
- **Pandas** and **NumPy** for data manipulation

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through GitHub issues or email.

---

**Built with ❤️ for educational analytics and student success**
