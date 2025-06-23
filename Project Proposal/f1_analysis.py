import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class Formula1DNFAnalyzer:
    def __init__(self, data_path='f1_data.csv'):
        """Initialize with data validation and processing"""
        try:
            self.df = pd.read_csv(data_path)
            self.df['Year'] = pd.to_numeric(self.df['Year'])
            self.df['Total Starters'] = pd.to_numeric(self.df['Total Starters'])
            self.df['DNF count'] = pd.to_numeric(self.df['DNF count'])
            self.df['DNF rate'] = self.df['DNF count'] / self.df['Total Starters']
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.df = None

    def _validate_data(self):
        return self.df is not None and not self.df.empty

    def generate_visualizations(self):
        """Generate all visualizations"""
        if not self._validate_data():
            return

        plt.figure(figsize=(15, 10))
        
        # 1. DNF Trend Over Years
        plt.subplot(2, 2, 1)
        yearly_data = self.df.groupby('Year').agg({
            'DNF count': 'mean',
            'Total Starters': 'mean',
            'DNF rate': 'mean'
        }).reset_index()
        
        sns.lineplot(data=yearly_data, x='Year', y='DNF count', marker='o', label='Mean DNF Count')
        plt.title('Average DNF Counts per Race (2015-2025)')
        plt.ylabel('Average DNF Count')
        plt.xticks(np.arange(2015, 2026, 1))
        plt.grid(True)

        # 2. DNF Rate Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(self.df['DNF rate'], kde=True, bins=15)
        plt.title('Distribution of DNF Rates')
        plt.xlabel('DNF Rate')

        # 3. Yearly DNF Rate Boxplot
        plt.subplot(2, 2, 4)
        sns.boxplot(data=self.df, x='Year', y='DNF rate')
        plt.title('Yearly DNF Rate Distribution')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # 4. Hypothesis Test Visualization
        test_result = self.perform_hypothesis_test()
        self._plot_hypothesis_test(test_result)

    def _plot_hypothesis_test(self, test_result):
        """Accurate visualization of the 20% reduction hypothesis test"""
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        groups = ['2015 (Baseline)', '2016-2025 (Actual)', '2016-2025 (Target)']
        values = [
            test_result['p2015'], 
            test_result['p_post'],
            test_result['p2015'] * 0.8  # 20% reduction target
        ]
        colors = ['#1f77b4', "#87BDF0", "#447392"]
        
        # Create the plot
        bars = plt.bar(groups, values, color=colors)
        
        # Add reference line and annotations
        plt.axhline(y=test_result['p2015'], color='r', linestyle=':', alpha=0.5, label='2015 Baseline')
        
        # Add exact values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        # Add hypothesis test results
        plt.text(1.5, max(values)*0.9, 
                f'Z = {test_result["z_score"]:.3f}\n'
                f'P = {test_result["p_value"]:.4f}\n'
                f'SE = {test_result["standard_error"]:.5f}',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Formatting
        plt.ylabel('DNF Rate')
        plt.title('DNF Rate: Actual vs 20% Reduction Target (2015 vs 2016-2025)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def generate_full_report(self):
        """Generate complete analysis with visuals"""
        if not self._validate_data():
            return
        
        # Statistical analysis
        self.generate_report()
        
        # Visualizations
        self.generate_visualizations()

    def _get_year_data(self, year):
        """Helper to get aggregated data for a specific year"""
        if not self._validate_data():
            return None
        year_data = self.df[self.df['Year'] == year]
        return {
            'total_dnf': year_data['DNF count'].sum(),
            'total_starters': year_data['Total Starters'].sum(),
            'race_count': len(year_data)
        }

    def calculate_confidence_interval(self, year=None):
        """Calculate exact CI as per your manual method"""
        if not self._validate_data():
            return None
        
        if year:
            data = self.df[self.df['Year'] == year]['DNF count']
        else:
            data = self.df['DNF count']
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Using normal distribution for all CIs as per your approach
        z = stats.norm.ppf(0.975)  # 95% CI
        margin = z * (std / np.sqrt(n))
        return (mean - margin, mean + margin)

    def perform_hypothesis_test(self):
        """Compare 2015 against ALL post-2015 years (2016-2025)"""
        if not self._validate_data():
            return None
            
        # Get 2015 data
        df2015 = self.df[self.df['Year'] == 2015]
        x2015 = df2015['DNF count'].sum()
        n2015 = df2015['Total Starters'].sum()
        p2015 = x2015 / n2015
        
        # Get ALL post-2015 data (2016-2025)
        df_post = self.df[self.df['Year'] > 2015]
        x_post = df_post['DNF count'].sum()
        n_post = df_post['Total Starters'].sum()
        p_post = x_post / n_post
        
        # Pooled proportion under H0 (that p_post = 0.8*p2015)
        p_pool = (0.8*p2015*n2015 + p_post*n_post) / (0.8*n2015 + n_post)
        
        # Standard error
        se = np.sqrt(p_pool*(1-p_pool)*(1/(0.8*n2015) + 1/n_post))
        
        # Test statistic
        z = (p_post - 0.8*p2015) / se
        
        # One-tailed test
        z_crit = stats.norm.ppf(0.05)
        p_val = stats.norm.cdf(z)
        
        return {
            'p2015': p2015,
            'p_post': p_post,
            'n2015': n2015,
            'n_post': n_post,
            'z_score': z,
            'z_critical': z_crit,
            'p_value': p_val,
            'reject_null': z < z_crit,
            'standard_error': se
        }

    def _validate_data(self):
        return self.df is not None and not self.df.empty

    def generate_report(self):
        """Generate complete analysis report"""
        if not self._validate_data():
            print("No valid data available")
            return
        
        # Descriptive statistics
        print("Descriptive Statistics (2015-2025):")
        stats_all = {
            'Sample mean': np.mean(self.df['DNF count']),
            'Sample variance': np.var(self.df['DNF count'], ddof=1),
            'Sample median': np.median(self.df['DNF count']),
            'Sample size': len(self.df)
        }
        for k, v in stats_all.items():
            print(f"{k}: {v:.4f}")
        
        # Confidence intervals
        print("\nConfidence Intervals:")
        ci_all = self.calculate_confidence_interval()
        print(f"2015-2025 95% CI: ({ci_all[0]:.4f}, {ci_all[1]:.4f})")
        
        ci2015 = self.calculate_confidence_interval(2015)
        print(f"2015 95% CI: ({ci2015[0]:.4f}, {ci2015[1]:.4f})")
        
        ci2025 = self.calculate_confidence_interval(2025)
        print(f"2025 95% CI: ({ci2025[0]:.4f}, {ci2025[1]:.4f})")
        
        # Hypothesis test
        test_result = self.perform_hypothesis_test()
        print("\nHypothesis Test Results (2015 vs 2016-2025):")
        print(f"P2015: {test_result['p2015']:.8f} (72 DNF / 374 starters)")
        print(f"P_post2015: {test_result['p_post']:.8f} (616 DNF / 3971 starters)")
        print(f"Standard Error: {test_result['standard_error']:.8f}")
        print(f"Z-score: {test_result['z_score']:.8f}")
        print(f"Critical value: {test_result['z_critical']:.6f}")
        print(f"P-value: {test_result['p_value']:.8f}")
        print(f"Reject null? {'Yes' if test_result['reject_null'] else 'No'}")

def main():
    analyzer = Formula1DNFAnalyzer()
    analyzer.generate_full_report()

if __name__ == "__main__":
    main()