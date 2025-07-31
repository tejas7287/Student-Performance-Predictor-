"""
Data generation module for creating realistic student performance datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StudentDataGenerator:
    """
    Generate realistic student performance data with various scenarios
    """
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.random_seed = random_seed
    
    def generate_basic_dataset(self, n_students=1000):
        """Generate basic student performance dataset"""
        
        # Student demographics
        student_ids = [f"STU{i:04d}" for i in range(1, n_students + 1)]
        
        # Generate attendance (beta distribution for realistic skew)
        attendance = np.random.beta(7, 2, n_students) * 100
        attendance = np.clip(attendance, 0, 100)
        
        # Generate study hours (gamma distribution)
        study_hours = np.random.gamma(2, 1.5, n_students)
        study_hours = np.clip(study_hours, 0, 12)
        
        # Generate past scores (normal distribution with some correlation to study habits)
        base_past_scores = np.random.normal(75, 12, n_students)
        # Students who study more tend to have better past scores
        past_scores = base_past_scores + (study_hours - 3) * 2 + np.random.normal(0, 5, n_students)
        past_scores = np.clip(past_scores, 0, 100)
        
        # Generate current performance with realistic relationships
        performance = self._calculate_performance(attendance, study_hours, past_scores)
        
        data = pd.DataFrame({
            'student_id': student_ids,
            'attendance': attendance,
            'study_hours': study_hours,
            'past_scores': past_scores,
            'performance': performance
        })
        
        return data
    
    def generate_advanced_dataset(self, n_students=1500):
        """Generate advanced dataset with additional features"""
        
        # Basic data
        basic_data = self.generate_basic_dataset(n_students)
        
        # Additional features
        
        # Socioeconomic factors
        socioeconomic_status = np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.3, 0.5, 0.2])
        
        # Parental education
        parental_education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                            n_students, p=[0.4, 0.35, 0.2, 0.05])
        
        # Extracurricular activities (hours per week)
        extracurricular_hours = np.random.exponential(2, n_students)
        extracurricular_hours = np.clip(extracurricular_hours, 0, 20)
        
        # Sleep hours
        sleep_hours = np.random.normal(7.5, 1.2, n_students)
        sleep_hours = np.clip(sleep_hours, 4, 12)
        
        # Screen time (hours per day)
        screen_time = np.random.gamma(2, 2, n_students)
        screen_time = np.clip(screen_time, 1, 16)
        
        # Stress level (1-10 scale)
        stress_level = np.random.normal(5, 2, n_students)
        stress_level = np.clip(stress_level, 1, 10)
        
        # Family support (1-10 scale)
        family_support = np.random.normal(7, 1.5, n_students)
        family_support = np.clip(family_support, 1, 10)
        
        # Adjust performance based on additional factors
        performance_adjustment = (
            (family_support - 5) * 1.5 +
            (sleep_hours - 7.5) * 2 +
            (stress_level - 5) * (-1) +
            (extracurricular_hours * 0.5) +
            np.random.normal(0, 3, n_students)
        )
        
        # Apply socioeconomic adjustments
        ses_adjustment = np.where(socioeconomic_status == 'High', 5,
                                np.where(socioeconomic_status == 'Medium', 0, -5))
        
        # Apply parental education adjustments
        edu_adjustment = np.where(parental_education == 'PhD', 4,
                                np.where(parental_education == 'Master', 2,
                                        np.where(parental_education == 'Bachelor', 1, 0)))
        
        adjusted_performance = (basic_data['performance'] + 
                              performance_adjustment + 
                              ses_adjustment + 
                              edu_adjustment)
        adjusted_performance = np.clip(adjusted_performance, 0, 100)
        
        # Create advanced dataset
        advanced_data = basic_data.copy()
        advanced_data['performance'] = adjusted_performance
        advanced_data['socioeconomic_status'] = socioeconomic_status
        advanced_data['parental_education'] = parental_education
        advanced_data['extracurricular_hours'] = extracurricular_hours
        advanced_data['sleep_hours'] = sleep_hours
        advanced_data['screen_time'] = screen_time
        advanced_data['stress_level'] = stress_level
        advanced_data['family_support'] = family_support
        
        return advanced_data
    
    def generate_time_series_data(self, n_students=100, n_weeks=12):
        """Generate time series data showing student progress over time"""
        
        data_list = []
        
        for student_id in range(1, n_students + 1):
            # Student baseline characteristics
            baseline_attendance = np.random.normal(80, 15)
            baseline_study_hours = np.random.gamma(2, 1.5)
            baseline_performance = np.random.normal(75, 12)
            
            # Generate trends
            attendance_trend = np.random.normal(0, 0.5, n_weeks).cumsum()
            study_trend = np.random.normal(0, 0.1, n_weeks).cumsum()
            
            for week in range(1, n_weeks + 1):
                # Weekly variations
                weekly_attendance = baseline_attendance + attendance_trend[week-1] + np.random.normal(0, 5)
                weekly_study_hours = baseline_study_hours + study_trend[week-1] + np.random.normal(0, 0.5)
                
                # Clip values
                weekly_attendance = np.clip(weekly_attendance, 0, 100)
                weekly_study_hours = np.clip(weekly_study_hours, 0, 12)
                
                # Calculate weekly performance
                weekly_performance = self._calculate_performance(
                    weekly_attendance, weekly_study_hours, baseline_performance
                ) + np.random.normal(0, 5)
                weekly_performance = np.clip(weekly_performance, 0, 100)
                
                data_list.append({
                    'student_id': f"STU{student_id:04d}",
                    'week': week,
                    'attendance': weekly_attendance,
                    'study_hours': weekly_study_hours,
                    'performance': weekly_performance,
                    'baseline_performance': baseline_performance
                })
        
        return pd.DataFrame(data_list)
    
    def _calculate_performance(self, attendance, study_hours, past_scores):
        """Calculate performance based on input features"""
        
        # Base performance calculation
        performance = (
            0.30 * attendance +
            0.40 * past_scores +
            0.20 * study_hours * 8 +
            0.10 * np.sqrt(study_hours * attendance / 10)  # Interaction term
        )
        
        # Add some noise
        performance += np.random.normal(0, 5, len(performance) if hasattr(performance, '__len__') else 1)
        
        # Add diminishing returns for excessive study
        if hasattr(study_hours, '__len__'):
            excessive_study_penalty = np.where(study_hours > 8, (study_hours - 8) * -2, 0)
        else:
            excessive_study_penalty = max(0, (study_hours - 8) * -2) if study_hours > 8 else 0
        
        performance += excessive_study_penalty
        
        # Clip to valid range
        performance = np.clip(performance, 0, 100)
        
        return performance
    
    def add_outliers(self, data, outlier_fraction=0.05):
        """Add outliers to the dataset"""
        n_outliers = int(len(data) * outlier_fraction)
        outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
        
        data_with_outliers = data.copy()
        
        for idx in outlier_indices:
            # Create different types of outliers
            outlier_type = np.random.choice(['high_performer', 'low_performer', 'inconsistent'])
            
            if outlier_type == 'high_performer':
                # Student who performs much better than expected
                data_with_outliers.loc[idx, 'performance'] = min(100, 
                    data_with_outliers.loc[idx, 'performance'] + np.random.normal(20, 5))
            elif outlier_type == 'low_performer':
                # Student who performs much worse than expected
                data_with_outliers.loc[idx, 'performance'] = max(0, 
                    data_with_outliers.loc[idx, 'performance'] - np.random.normal(20, 5))
            else:  # inconsistent
                # Student with very different patterns
                data_with_outliers.loc[idx, 'attendance'] = np.random.uniform(0, 100)
                data_with_outliers.loc[idx, 'study_hours'] = np.random.uniform(0, 12)
        
        return data_with_outliers
    
    def create_class_scenarios(self, n_classes=5):
        """Create different class scenarios with varying characteristics"""
        
        scenarios = {
            'High Performing Class': {
                'attendance_mean': 90, 'attendance_std': 8,
                'study_hours_mean': 5, 'study_hours_std': 1.5,
                'past_scores_mean': 85, 'past_scores_std': 10
            },
            'Average Class': {
                'attendance_mean': 80, 'attendance_std': 15,
                'study_hours_mean': 3, 'study_hours_std': 2,
                'past_scores_mean': 75, 'past_scores_std': 12
            },
            'Struggling Class': {
                'attendance_mean': 65, 'attendance_std': 20,
                'study_hours_mean': 2, 'study_hours_std': 1.5,
                'past_scores_mean': 60, 'past_scores_std': 15
            },
            'Mixed Performance Class': {
                'attendance_mean': 75, 'attendance_std': 25,
                'study_hours_mean': 3.5, 'study_hours_std': 3,
                'past_scores_mean': 70, 'past_scores_std': 20
            },
            'Highly Variable Class': {
                'attendance_mean': 80, 'attendance_std': 30,
                'study_hours_mean': 4, 'study_hours_std': 4,
                'past_scores_mean': 75, 'past_scores_std': 25
            }
        }
        
        all_data = []
        
        for class_name, params in scenarios.items():
            n_students = np.random.randint(25, 40)  # Random class size
            
            attendance = np.random.normal(params['attendance_mean'], params['attendance_std'], n_students)
            attendance = np.clip(attendance, 0, 100)
            
            study_hours = np.random.normal(params['study_hours_mean'], params['study_hours_std'], n_students)
            study_hours = np.clip(study_hours, 0, 12)
            
            past_scores = np.random.normal(params['past_scores_mean'], params['past_scores_std'], n_students)
            past_scores = np.clip(past_scores, 0, 100)
            
            performance = self._calculate_performance(attendance, study_hours, past_scores)
            
            class_data = pd.DataFrame({
                'student_id': [f"{class_name.replace(' ', '_')}_{i:03d}" for i in range(n_students)],
                'class_name': class_name,
                'attendance': attendance,
                'study_hours': study_hours,
                'past_scores': past_scores,
                'performance': performance
            })
            
            all_data.append(class_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    def visualize_data(self, data, save_plots=False):
        """Create visualizations of the generated data"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Distribution plots
        axes[0, 0].hist(data['attendance'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Attendance Distribution')
        axes[0, 0].set_xlabel('Attendance (%)')
        
        axes[0, 1].hist(data['study_hours'], bins=30, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Study Hours Distribution')
        axes[0, 1].set_xlabel('Study Hours per Day')
        
        axes[0, 2].hist(data['past_scores'], bins=30, alpha=0.7, color='salmon')
        axes[0, 2].set_title('Past Scores Distribution')
        axes[0, 2].set_xlabel('Past Scores (%)')
        
        # Scatter plots
        axes[1, 0].scatter(data['attendance'], data['performance'], alpha=0.6)
        axes[1, 0].set_title('Attendance vs Performance')
        axes[1, 0].set_xlabel('Attendance (%)')
        axes[1, 0].set_ylabel('Performance (%)')
        
        axes[1, 1].scatter(data['study_hours'], data['performance'], alpha=0.6)
        axes[1, 1].set_title('Study Hours vs Performance')
        axes[1, 1].set_xlabel('Study Hours per Day')
        axes[1, 1].set_ylabel('Performance (%)')
        
        axes[1, 2].scatter(data['past_scores'], data['performance'], alpha=0.6)
        axes[1, 2].set_title('Past Scores vs Performance')
        axes[1, 2].set_xlabel('Past Scores (%)')
        axes[1, 2].set_ylabel('Performance (%)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('student_data_visualization.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        if save_plots:
            plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Demonstrate data generation capabilities"""
    
    generator = StudentDataGenerator()
    
    print("Generating basic dataset...")
    basic_data = generator.generate_basic_dataset(1000)
    print(f"Basic dataset shape: {basic_data.shape}")
    print(basic_data.head())
    
    print("\nGenerating advanced dataset...")
    advanced_data = generator.generate_advanced_dataset(500)
    print(f"Advanced dataset shape: {advanced_data.shape}")
    print(advanced_data.head())
    
    print("\nGenerating class scenarios...")
    class_data = generator.create_class_scenarios()
    print(f"Class scenarios dataset shape: {class_data.shape}")
    print(class_data.groupby('class_name').size())
    
    # Save datasets
    basic_data.to_csv('basic_student_data.csv', index=False)
    advanced_data.to_csv('advanced_student_data.csv', index=False)
    class_data.to_csv('class_scenarios_data.csv', index=False)
    
    print("\nDatasets saved to CSV files.")
    
    # Visualize basic data
    print("\nCreating visualizations...")
    generator.visualize_data(basic_data)

if __name__ == "__main__":
    main()