"""
Growth Chart Integration for Pediatric Bone Age Assessment

Provides growth chart plotting, percentile calculations, and 
comparison with standard growth curves.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json


class GrowthChartAnalyzer:
    """Analyze and plot growth charts for pediatric patients."""
    
    def __init__(self):
        """Initialize growth chart analyzer with WHO standards."""
        self.growth_standards = self._load_who_standards()
    
    def calculate_percentile(self, age_months: float, bone_age_months: float, 
                           gender: str, chronological_age_months: Optional[float] = None) -> Dict:
        """
        Calculate percentile and Z-score for bone age.
        
        Args:
            age_months: Bone age in months
            bone_age_months: Predicted bone age in months
            gender: 'M' or 'F'
            chronological_age_months: Chronological age (optional)
            
        Returns:
            Dictionary with percentile, Z-score, and interpretation
        """
        if chronological_age_months is None:
            chronological_age_months = age_months
        
        # Calculate bone age difference
        age_diff = bone_age_months - chronological_age_months
        
        # Simplified percentile calculation (based on normal distribution)
        # In practice, this would use WHO growth standards lookup tables
        z_score = age_diff / 12.0  # Approximate SD of 12 months
        
        # Convert Z-score to percentile
        from scipy import stats
        percentile = stats.norm.cdf(z_score) * 100
        
        # Interpret percentile
        if percentile >= 97:
            interpretation = "Very advanced bone age"
        elif percentile >= 90:
            interpretation = "Advanced bone age"
        elif percentile >= 75:
            interpretation = "Above average bone age"
        elif percentile >= 50:
            interpretation = "Average bone age"
        elif percentile >= 25:
            interpretation = "Below average bone age"
        elif percentile >= 10:
            interpretation = "Delayed bone age"
        else:
            interpretation = "Significantly delayed bone age"
        
        return {
            'percentile': round(percentile, 2),
            'z_score': round(z_score, 2),
            'bone_age_months': round(bone_age_months, 1),
            'chronological_age_months': round(chronological_age_months, 1),
            'age_difference_months': round(age_diff, 1),
            'interpretation': interpretation,
            'gender': gender
        }
    
    def calculate_growth_velocity(self, visits: List[Dict]) -> Dict:
        """
        Calculate growth velocity between visits.
        
        Args:
            visits: List of visit dictionaries with 'date' and 'bone_age'
            
        Returns:
            Growth velocity analysis
        """
        if len(visits) < 2:
            return {'error': 'Need at least 2 visits for velocity calculation'}
        
        velocities = []
        for i in range(1, len(visits)):
            time_diff = (visits[i]['date'] - visits[i-1]['date']).days / 30.44  # months
            bone_age_diff = visits[i]['bone_age'] - visits[i-1]['bone_age']
            velocity = bone_age_diff / time_diff if time_diff > 0 else 0
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities)
        
        return {
            'average_velocity': round(avg_velocity, 2),
            'velocities': [round(v, 2) for v in velocities],
            'interpretation': self._interpret_velocity(avg_velocity)
        }
    
    def _interpret_velocity(self, velocity: float) -> str:
        """Interpret growth velocity."""
        if velocity > 1.2:
            return "Accelerated growth - may indicate early puberty"
        elif velocity > 0.8:
            return "Normal growth velocity"
        elif velocity > 0.5:
            return "Slowed growth - monitor closely"
        else:
            return "Significantly slowed growth - consider evaluation"
    
    def compare_with_standards(self, age_months: float, bone_age_months: float, 
                              gender: str) -> Dict:
        """
        Compare bone age with growth standards.
        
        Args:
            age_months: Chronological age
            bone_age_months: Bone age
            gender: 'M' or 'F'
            
        Returns:
            Comparison with standards
        """
        percentile_data = self.calculate_percentile(age_months, bone_age_months, gender, age_months)
        
        # Calculate standard deviation score
        age_diff = bone_age_months - age_months
        sd_score = age_diff / 12.0
        
        # Clinical interpretation
        if abs(sd_score) > 2:
            clinical_significance = "Clinically significant difference"
            recommendation = "Consider endocrine evaluation"
        elif abs(sd_score) > 1.5:
            clinical_significance = "Moderate difference"
            recommendation = "Monitor closely, repeat in 6 months"
        else:
            clinical_significance = "Within normal variation"
            recommendation = "Routine follow-up"
        
        return {
            **percentile_data,
            'sd_score': round(sd_score, 2),
            'clinical_significance': clinical_significance,
            'recommendation': recommendation
        }
    
    def generate_growth_chart_data(self, patient_data: List[Dict], gender: str) -> Dict:
        """
        Generate data for growth chart plotting.
        
        Args:
            patient_data: List of visits with dates and bone ages
            gender: Patient gender
            
        Returns:
            Chart data dictionary
        """
        ages = [d['age_months'] for d in patient_data]
        bone_ages = [d['bone_age_months'] for d in patient_data]
        
        # Calculate percentiles for each point
        percentiles = []
        for age, bone_age in zip(ages, bone_ages):
            pct = self.calculate_percentile(age, bone_age, gender, age)
            percentiles.append(pct['percentile'])
        
        return {
            'ages': ages,
            'bone_ages': bone_ages,
            'percentiles': percentiles,
            'gender': gender,
            'data_points': len(patient_data)
        }
    
    def _load_who_standards(self) -> Dict:
        """Load WHO growth standards (simplified version)."""
        # In production, this would load actual WHO growth standard tables
        return {
            'male': {
                'mean': {},
                'sd': {}
            },
            'female': {
                'mean': {},
                'sd': {}
            }
        }


class PercentileCalculator:
    """Calculate and interpret percentiles for bone age."""
    
    @staticmethod
    def get_percentile_bands() -> List[float]:
        """Get standard percentile bands."""
        return [3, 10, 25, 50, 75, 90, 97]
    
    @staticmethod
    def interpret_percentile(percentile: float) -> Dict:
        """Interpret percentile value."""
        if percentile >= 97:
            category = "Very Advanced"
            color = "red"
            action = "Consider evaluation for precocious puberty"
        elif percentile >= 90:
            category = "Advanced"
            color = "orange"
            action = "Monitor closely"
        elif percentile >= 75:
            category = "Above Average"
            color = "yellow"
            action = "Normal variation"
        elif percentile >= 50:
            category = "Average"
            color = "green"
            action = "Normal"
        elif percentile >= 25:
            category = "Below Average"
            color = "yellow"
            action = "Normal variation"
        elif percentile >= 10:
            category = "Delayed"
            color = "orange"
            action = "Monitor closely"
        else:
            category = "Significantly Delayed"
            color = "red"
            action = "Consider endocrine evaluation"
        
        return {
            'category': category,
            'color': color,
            'action': action,
            'percentile': percentile
        }

