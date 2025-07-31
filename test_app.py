"""
Test script to verify the Streamlit app functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation functionality"""
    try:
        import numpy as np
        import pandas as pd
        
        # Simulate the data generation from app.py
        np.random.seed(42)
        n_samples = 100
        
        attendance = np.random.normal(85, 15, n_samples)
        attendance = np.clip(attendance, 0, 100)
        
        study_hours = np.random.exponential(3, n_samples)
        study_hours = np.clip(study_hours, 0, 12)
        
        past_scores = np.random.normal(75, 12, n_samples)
        past_scores = np.clip(past_scores, 0, 100)
        
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
        
        print(f"✅ Data generation successful: {data.shape}")
        print(f"   Sample data:\n{data.head()}")
        return True
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

def test_model_training():
    """Test basic model training"""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score
        import numpy as np
        import pandas as pd
        
        # Generate test data
        np.random.seed(42)
        n_samples = 200
        
        X = np.random.rand(n_samples, 3) * 100  # attendance, study_hours, past_scores
        y = 0.3 * X[:, 0] + 0.4 * X[:, 2] + 0.2 * X[:, 1] * 10 + np.random.normal(0, 5, n_samples)
        y = np.clip(y, 0, 100)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model training successful: R² = {r2:.3f}")
        return True
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def test_advanced_modules():
    """Test advanced modules"""
    try:
        from model_trainer import StudentPerformancePredictor
        from data_generator import StudentDataGenerator
        
        # Test predictor initialization
        predictor = StudentPerformancePredictor()
        print("✅ StudentPerformancePredictor initialized")
        
        # Test data generator
        generator = StudentDataGenerator()
        basic_data = generator.generate_basic_dataset(50)
        print(f"✅ Data generator working: {basic_data.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Advanced modules error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Student Performance Predictor Application")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Generation Test", test_data_generation),
        ("Model Training Test", test_model_training),
        ("Advanced Modules Test", test_advanced_modules)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        print("\n🚀 To run the Streamlit app:")
        print("   streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()