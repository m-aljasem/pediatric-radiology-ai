"""
Radiology Report Generation for Bone Age Assessment

Generates structured radiology reports with clinical recommendations.
"""

from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
import json


class RadiologyReportGenerator:
    """Generate structured radiology reports."""
    
    def __init__(self):
        """Initialize report generator."""
        self.template_path = Path("templates/radiology_report_template.txt")
    
    def generate_report(self, patient_info: Dict, bone_age_data: Dict, 
                       comparison_data: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive radiology report.
        
        Args:
            patient_info: Patient demographics
            bone_age_data: Bone age prediction results
            comparison_data: Previous scan comparison (optional)
            
        Returns:
            Structured report dictionary
        """
        report = {
            'report_header': self._generate_header(patient_info),
            'clinical_information': self._generate_clinical_info(patient_info),
            'findings': self._generate_findings(bone_age_data),
            'comparison': self._generate_comparison(comparison_data) if comparison_data else None,
            'impression': self._generate_impression(bone_age_data),
            'recommendations': self._generate_recommendations(bone_age_data),
            'report_footer': self._generate_footer()
        }
        
        return report
    
    def _generate_header(self, patient_info: Dict) -> Dict:
        """Generate report header."""
        return {
            'facility_name': 'Medical Imaging Center',
            'report_type': 'Bone Age Assessment',
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accession_number': patient_info.get('accession_number', 'N/A'),
            'patient_id': patient_info.get('patient_id', 'N/A'),
            'patient_name': patient_info.get('name', 'N/A'),
            'date_of_birth': patient_info.get('dob', 'N/A'),
            'gender': patient_info.get('gender', 'N/A')
        }
    
    def _generate_clinical_info(self, patient_info: Dict) -> Dict:
        """Generate clinical information section."""
        return {
            'exam_date': patient_info.get('exam_date', datetime.now().strftime('%Y-%m-%d')),
            'exam_type': 'Hand X-ray for Bone Age Assessment',
            'clinical_indication': patient_info.get('indication', 'Routine bone age assessment'),
            'technique': 'Digital radiography of left hand and wrist'
        }
    
    def _generate_findings(self, bone_age_data: Dict) -> Dict:
        """Generate findings section."""
        bone_age_months = bone_age_data.get('prediction', 0)
        chronological_age = bone_age_data.get('chronological_age_months', 0)
        age_difference = bone_age_months - chronological_age
        
        findings = {
            'bone_age': f"{bone_age_months:.1f} months ({bone_age_months/12:.1f} years)",
            'chronological_age': f"{chronological_age:.1f} months ({chronological_age/12:.1f} years)",
            'age_difference': f"{age_difference:+.1f} months",
            'interpretation': self._interpret_age_difference(age_difference)
        }
        
        if 'percentile' in bone_age_data:
            findings['percentile'] = f"{bone_age_data['percentile']:.1f}th percentile"
            findings['z_score'] = f"{bone_age_data.get('z_score', 0):.2f}"
        
        return findings
    
    def _interpret_age_difference(self, diff: float) -> str:
        """Interpret bone age difference."""
        abs_diff = abs(diff)
        if abs_diff <= 6:
            return "Bone age is consistent with chronological age"
        elif abs_diff <= 12:
            if diff > 0:
                return "Bone age is advanced compared to chronological age"
            else:
                return "Bone age is delayed compared to chronological age"
        else:
            if diff > 0:
                return "Significantly advanced bone age - may indicate precocious puberty"
            else:
                return "Significantly delayed bone age - may indicate growth disorder"
    
    def _generate_comparison(self, comparison_data: Dict) -> Dict:
        """Generate comparison with previous study."""
        if not comparison_data:
            return None
        
        prev_bone_age = comparison_data.get('previous_bone_age', 0)
        current_bone_age = comparison_data.get('current_bone_age', 0)
        time_interval = comparison_data.get('time_interval_months', 0)
        
        bone_age_progression = current_bone_age - prev_bone_age
        expected_progression = time_interval
        
        return {
            'previous_study_date': comparison_data.get('previous_date', 'N/A'),
            'previous_bone_age': f"{prev_bone_age:.1f} months",
            'current_bone_age': f"{current_bone_age:.1f} months",
            'time_interval': f"{time_interval:.1f} months",
            'bone_age_progression': f"{bone_age_progression:.1f} months",
            'expected_progression': f"{expected_progression:.1f} months",
            'interpretation': self._interpret_progression(bone_age_progression, expected_progression)
        }
    
    def _interpret_progression(self, actual: float, expected: float) -> str:
        """Interpret bone age progression."""
        ratio = actual / expected if expected > 0 else 0
        if ratio > 1.2:
            return "Accelerated bone age progression"
        elif ratio > 0.8:
            return "Normal bone age progression"
        else:
            return "Slowed bone age progression"
    
    def _generate_impression(self, bone_age_data: Dict) -> str:
        """Generate impression section."""
        bone_age_months = bone_age_data.get('prediction', 0)
        chronological_age = bone_age_data.get('chronological_age_months', 0)
        age_diff = bone_age_months - chronological_age
        
        impression = f"Bone age assessment reveals a bone age of {bone_age_months:.1f} months "
        impression += f"({bone_age_months/12:.1f} years) compared to a chronological age of "
        impression += f"{chronological_age:.1f} months ({chronological_age/12:.1f} years). "
        
        if abs(age_diff) <= 6:
            impression += "This is within normal variation."
        elif age_diff > 12:
            impression += "The bone age is significantly advanced, which may be associated with precocious puberty."
        elif age_diff < -12:
            impression += "The bone age is significantly delayed, which may warrant further evaluation for growth disorders."
        elif age_diff > 0:
            impression += "The bone age is moderately advanced."
        else:
            impression += "The bone age is moderately delayed."
        
        return impression
    
    def _generate_recommendations(self, bone_age_data: Dict) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []
        
        bone_age_months = bone_age_data.get('prediction', 0)
        chronological_age = bone_age_data.get('chronological_age_months', 0)
        age_diff = bone_age_months - chronological_age
        
        if abs(age_diff) <= 6:
            recommendations.append("Routine follow-up as clinically indicated")
        elif age_diff > 12:
            recommendations.append("Consider endocrine evaluation for precocious puberty")
            recommendations.append("Monitor growth velocity")
            recommendations.append("Consider bone age reassessment in 6-12 months")
        elif age_diff < -12:
            recommendations.append("Consider endocrine evaluation for growth delay")
            recommendations.append("Assess growth hormone levels if clinically indicated")
            recommendations.append("Consider bone age reassessment in 6-12 months")
        else:
            recommendations.append("Monitor growth parameters")
            recommendations.append("Consider repeat bone age assessment in 12 months")
        
        return recommendations
    
    def _generate_footer(self) -> Dict:
        """Generate report footer."""
        return {
            'radiologist': 'AI-Assisted Report',
            'disclaimer': 'This report was generated with AI assistance and should be reviewed by a qualified radiologist.',
            'report_status': 'Final'
        }
    
    def format_as_text(self, report: Dict) -> str:
        """Format report as plain text."""
        lines = []
        
        # Header
        header = report['report_header']
        lines.append("=" * 70)
        lines.append(f"RADIOLOGY REPORT - {header['report_type']}")
        lines.append("=" * 70)
        lines.append(f"Patient: {header['patient_name']} (ID: {header['patient_id']})")
        lines.append(f"DOB: {header['date_of_birth']} | Gender: {header['gender']}")
        lines.append(f"Report Date: {header['report_date']}")
        lines.append("")
        
        # Clinical Information
        lines.append("CLINICAL INFORMATION:")
        lines.append("-" * 70)
        clinical = report['clinical_information']
        for key, value in clinical.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Findings
        lines.append("FINDINGS:")
        lines.append("-" * 70)
        findings = report['findings']
        for key, value in findings.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Comparison
        if report['comparison']:
            lines.append("COMPARISON:")
            lines.append("-" * 70)
            for key, value in report['comparison'].items():
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
            lines.append("")
        
        # Impression
        lines.append("IMPRESSION:")
        lines.append("-" * 70)
        lines.append(report['impression'])
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 70)
        for i, rec in enumerate(report['recommendations'], 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
        
        # Footer
        footer = report['report_footer']
        lines.append("-" * 70)
        lines.append(footer['disclaimer'])
        lines.append("=" * 70)
        
        return "\n".join(lines)

