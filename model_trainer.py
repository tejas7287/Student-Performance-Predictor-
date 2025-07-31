"""
Advanced model training module for Student Performance Predictor
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class StudentPerformancePredictor:
    """
    A comprehensive machine learning model for predicting student performance
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = ['attendance', 'study_hours', 'past_scores']
        
    def generate_enhanced_dataset(self, n_samples=2000):
        """Generate a more realistic and complex dataset"""
        np.random.seed(42)
        
        # Generate base features
        attendance = np.random.beta(8, 2, n_samples) * 100  # Skewed towards higher attendance
        study_hours = np.random.gamma(2, 1.5, n_samples)  # Realistic study hour distribution
        study_hours = np.clip(study_hours, 0, 12)
        
        past_scores = np.random.normal(75, 15, n_samples)
        past_scores = np.clip(past_scores, 0, 100)
        
        # Add some realistic correlations and non-linear relationships
        # Students with higher past scores tend to have better attendance
        attendance += (past_scores - 75) * 0.2 + np.random.normal(0, 5, n_samples)
        attendance = np.clip(attendance, 0, 100)
        
        # Performance calculation with non-linear relationships
        performance = (
            0.35 * attendance +
            0.40 * past_scores +
            0.15 * study_hours * 8 +
            0.05 * (study_hours ** 1.5) * 3 +  # Diminishing returns on study hours
            0.05 * (attendance * past_scores) / 100 +  # Interaction term
            np.random.normal(0, 6, n_samples)
        )
        
        # Add some outliers (students who perform unexpectedly)
        outlier_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        performance[outlier_indices] += np.random.normal(0, 15, len(outlier_indices))
        
        performance = np.clip(performance, 0, 100)
        
        # Create additional engineered features
        data = pd.DataFrame({
            'attendance': attendance,
            'study_hours': study_hours,
            'past_scores': past_scores,
            'performance': performance
        })
        
        # Feature engineering
        data['attendance_study_interaction'] = data['attendance'] * data['study_hours'] / 100
        data['past_score_squared'] = data['past_scores'] ** 2 / 100
        data['study_efficiency'] = data['past_scores'] / (data['study_hours'] + 1)  # Avoid division by zero
        
        return data
    
    def prepare_features(self, data, include_engineered=True):
        """Prepare feature matrix"""
        base_features = ['attendance', 'study_hours', 'past_scores']
        
        if include_engineered:
            engineered_features = ['attendance_study_interaction', 'past_score_squared', 'study_efficiency']
            features = base_features + engineered_features
        else:
            features = base_features
            
        return data[features], data['performance']
    
    def train_models(self, data, test_size=0.2):
        """Train multiple models and compare performance"""
        
        # Prepare data
        X, y = self.prepare_features(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models with hyperparameter tuning
        model_configs = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {},
                'use_scaled': True
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                'use_scaled': True
            },
            'Lasso Regression': {
                'model': Lasso(),
                'params': {'alpha': [0.01, 0.1, 1.0, 10.0]},
                'use_scaled': True
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'use_scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'use_scaled': False
            },
            'Support Vector Regression': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                },
                'use_scaled': True
            }
        }
        
        results = {}
        
        for name, config in model_configs.items():
            print(f"Training {name}...")
            
            # Prepare data based on scaling requirement
            if config['use_scaled']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Hyperparameter tuning
            if config['params']:
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=5, 
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train_model, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                best_model = config['model']
                best_model.fit(X_train_model, y_train)
                best_params = {}
            
            # Make predictions
            y_pred = best_model.predict(X_test_model)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train_model, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': best_model,
                'best_params': best_params,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'use_scaled': config['use_scaled']
            }
            
            self.models[name] = best_model
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
        
        return results, X_test, y_test
    
    def predict(self, attendance, study_hours, past_scores, model_name=None):
        """Make prediction for a single student"""
        if model_name is None:
            model = self.best_model
        else:
            model = self.models[model_name]
        
        # Create feature vector (including engineered features)
        attendance_study_interaction = attendance * study_hours / 100
        past_score_squared = past_scores ** 2 / 100
        study_efficiency = past_scores / (study_hours + 1)
        
        features = np.array([[
            attendance, study_hours, past_scores,
            attendance_study_interaction, past_score_squared, study_efficiency
        ]])
        
        # Scale if necessary (check if model needs scaling)
        # This is a simplified check - in practice, you'd store this info
        if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
            features = self.scaler.transform(features)
        
        prediction = model.predict(features)[0]
        return np.clip(prediction, 0, 100)
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'best_model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.best_model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
    
    def plot_results(self, results, X_test, y_test):
        """Plot model comparison and results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        rmse_scores = [results[name]['rmse'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores)
        axes[0, 0].set_title('Model R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(model_names, rmse_scores)
        axes[0, 1].set_title('Model RMSE Scores')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Best model predictions vs actual
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        y_pred = results[best_model_name]['predictions']
        
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Performance')
        axes[1, 0].set_ylabel('Predicted Performance')
        axes[1, 0].set_title(f'{best_model_name}: Predicted vs Actual')
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Performance')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Plot')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """Main function to demonstrate the model training"""
    predictor = StudentPerformancePredictor()
    
    # Generate dataset
    print("Generating enhanced dataset...")
    data = predictor.generate_enhanced_dataset(n_samples=2000)
    
    # Train models
    print("Training models...")
    results, X_test, y_test = predictor.train_models(data)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  R² Score: {result['r2']:.4f}")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  MAE: {result['mae']:.4f}")
        print(f"  CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        if result['best_params']:
            print(f"  Best Params: {result['best_params']}")
    
    # Save the model
    predictor.save_model('student_performance_model.pkl')
    
    # Example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    test_cases = [
        (95, 4, 85),  # High attendance, good study habits, good past scores
        (70, 2, 60),  # Average attendance, low study hours, average past scores
        (85, 6, 90),  # Good attendance, high study hours, excellent past scores
        (50, 1, 45),  # Poor attendance, minimal study, poor past scores
    ]
    
    for attendance, study_hours, past_scores in test_cases:
        prediction = predictor.predict(attendance, study_hours, past_scores)
        print(f"Attendance: {attendance}%, Study Hours: {study_hours}h, Past Scores: {past_scores}% → Predicted: {prediction:.1f}%")

if __name__ == "__main__":
    main()