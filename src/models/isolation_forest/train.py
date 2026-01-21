#Isolation Forest Pipeline for FFIEC Bank Call Report Anomaly Detection
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class IsolationForestPipeline:
    """
    Isolation Forests work by:
    1. Randomly selecting a feature
    2. Randomly selecting a split value for that feature
    3. Building many such trees (a forest)
    4. Anomalies are easier to isolate (fewer splits needed) than normal points
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the pipeline.
        
        Args:
            contamination: Expected proportion of outliers (default 10%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.results_df = None
        
    def load_data(self, filepath: str, feature_cols: Optional[list] = None) -> pd.DataFrame:
        """
        Load CSV data and select features for analysis.
        
        Args:
            filepath: Path to CSV file
            feature_cols: List of column names to use (if None, uses all numeric)
        
        Returns:
            DataFrame with selected features
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Select features
        if feature_cols is None:
            # Use all numeric columns
            df_features = df.select_dtypes(include=[np.number])
            print(f"Auto-selected {len(df_features.columns)} numeric features")
        else:
            df_features = df[feature_cols]
            print(f"Using {len(feature_cols)} specified features")
        
        self.feature_names = df_features.columns.tolist()
        
        # Store full dataframe for later reference
        self.original_df = df
        
        print(f"Data loaded: {df_features.shape[0]} banks, {df_features.shape[1]} features")
        return df_features
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data: handle missing values and scale features.
        
        Scaling is important for Isolation Forest because it prevents
        features with larger ranges from dominating the split decisions.
        """
        print("\nPreprocessing data...")
        
        # Handle missing values - fill with median
        df_clean = df.fillna(df.median())
        
        # Report any remaining issues
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain")
        
        # Standardize features (mean=0, std=1)
        X_scaled = self.scaler.fit_transform(df_clean)
        
        print(f"Preprocessing complete. Shape: {X_scaled.shape}")
        return X_scaled
    
    def train(self, X: np.ndarray, n_estimators: int = 100):
        """
        Train the Isolation Forest model.
        
        Args:
            X: Preprocessed feature matrix
            n_estimators: Number of trees in the forest (more = more stable)
        """
        print(f"\nTraining Isolation Forest with {n_estimators} trees...")
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.model.fit(X)
        print("Training complete!")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in the data.
        
        Returns:
            predictions: 1 for normal, -1 for anomaly
            scores: Anomaly scores (lower = more anomalous)
        """
        print("\nDetecting anomalies...")
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        n_anomalies = (predictions == -1).sum()
        print(f"Found {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%)")
        
        return predictions, scores
    
    def create_results_dataframe(self, predictions: np.ndarray, 
                                 scores: np.ndarray) -> pd.DataFrame:
        """
        Create a results dataframe with anomaly flags and scores.
        """
        results = self.original_df.copy()
        results['anomaly'] = predictions == -1
        results['anomaly_score'] = scores
        
        # Rank by how anomalous (most anomalous = rank 1)
        results['anomaly_rank'] = results['anomaly_score'].rank()
        
        self.results_df = results
        return results
    
    def plot_anomaly_distribution(self, scores: np.ndarray, predictions: np.ndarray):
        """
        Visualize the distribution of anomaly scores.
        """
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Histogram of anomaly scores
        plt.subplot(1, 2, 1)
        plt.hist(scores[predictions == 1], bins=50, alpha=0.7, label='Normal', color='blue')
        plt.hist(scores[predictions == -1], bins=50, alpha=0.7, label='Anomaly', color='red')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        plt.axvline(x=self.model.threshold_, color='black', linestyle='--', label='Threshold')
        
        # Plot 2: Scatter of index vs score
        plt.subplot(1, 2, 2)
        colors = ['red' if p == -1 else 'blue' for p in predictions]
        plt.scatter(range(len(scores)), scores, c=colors, alpha=0.6, s=20)
        plt.xlabel('Bank Index')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores by Bank')
        plt.axhline(y=self.model.threshold_, color='black', linestyle='--', label='Threshold')
        
        plt.tight_layout()
        plt.savefig('anomaly_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\nPlot saved as 'anomaly_distribution.png'")
    
    def plot_feature_importance_pca(self, X: np.ndarray, predictions: np.ndarray):
        """
        Use PCA to visualize anomalies in 2D space.
        
        This helps understand which feature combinations separate anomalies.
        """
        print("\nCreating PCA visualization...")
        
        # Reduce to 2 dimensions for visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 7))
        
        # Plot normal vs anomaly
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                   c='blue', alpha=0.5, s=30, label='Normal')
        plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                   c='red', alpha=0.8, s=50, label='Anomaly', edgecolors='black')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.title('Anomaly Detection Results (PCA Projection)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.savefig('pca_anomalies.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("PCA plot saved as 'pca_anomalies.png'")
    
    def get_top_anomalies(self, n: int = 10) -> pd.DataFrame:
        """
        Return the top N most anomalous banks.
        """
        if self.results_df is None:
            raise ValueError("Must run create_results_dataframe first")
        
        top_anomalies = self.results_df[self.results_df['anomaly']].nsmallest(n, 'anomaly_score')
        return top_anomalies
    
    def run_pipeline(self, filepath: str, feature_cols: Optional[list] = None, 
                     n_estimators: int = 100, plot: bool = True):
        """
        Execute the complete pipeline end-to-end.
        
        Args:
            filepath: Path to CSV file
            feature_cols: Feature columns to use
            n_estimators: Number of trees
            plot: Whether to generate visualizations
        """
        print("="*60)
        print("ISOLATION FOREST ANOMALY DETECTION PIPELINE")
        print("="*60)
        
        # 1. Load data
        df_features = self.load_data(filepath, feature_cols)
        
        # 2. Preprocess
        X = self.preprocess(df_features)
        
        # 3. Train
        self.train(X, n_estimators)
        
        # 4. Predict
        predictions, scores = self.predict(X)
        
        # 5. Create results
        results = self.create_results_dataframe(predictions, scores)
        
        # 6. Visualize
        if plot:
            self.plot_anomaly_distribution(scores, predictions)
            self.plot_feature_importance_pca(X, predictions)
        
        # 7. Summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE - SUMMARY")
        print("="*60)
        print(f"Total banks analyzed: {len(predictions)}")
        print(f"Anomalies detected: {(predictions == -1).sum()}")
        print(f"Contamination rate: {self.contamination*100}%")
        
        return results


# Example usage when you have your data
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = IsolationForestPipeline(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42
    )
    
    # Example 1: Run with specific features (recommended for Call Reports)
    # Uncomment when you have your data:
    """
    feature_columns = [
        'total_assets',
        'total_loans', 
        'total_deposits',
        'net_income',
        'tier1_capital_ratio',
        # Add your specific line items here
    ]
    
    results = pipeline.run_pipeline(
        filepath='path/to/your/call_report_data.csv',
        feature_cols=feature_columns,
        n_estimators=100,
        plot=True
    )
    
    # View top 10 anomalies
    top_anomalies = pipeline.get_top_anomalies(n=10)
    print("\nTop 10 Most Anomalous Banks:")
    print(top_anomalies[['bank_name', 'anomaly_score', 'anomaly_rank']])
    
    # Save results
    results.to_csv('anomaly_detection_results.csv', index=False)
    """
    
    # Example 2: Create synthetic data for testing the pipeline
    print("Creating synthetic data for demonstration...")
    np.random.seed(42)
    n_banks = 500
    
    # Normal banks
    normal_data = {
        'bank_id': range(n_banks),
        'total_assets': np.random.normal(1000, 200, n_banks),
        'total_loans': np.random.normal(600, 100, n_banks),
        'total_deposits': np.random.normal(800, 150, n_banks),
        'net_income': np.random.normal(50, 15, n_banks),
        'equity_ratio': np.random.normal(0.1, 0.02, n_banks),
    }
    
    # Add some anomalies
    for i in range(20):
        idx = np.random.randint(0, n_banks)
        normal_data['total_assets'][idx] *= np.random.uniform(3, 5)
        normal_data['net_income'][idx] *= np.random.uniform(-2, 0.2)
    
    df_synthetic = pd.DataFrame(normal_data)
    df_synthetic.to_csv('synthetic_call_reports.csv', index=False)
    
    # Run pipeline on synthetic data
    results = pipeline.run_pipeline(
        filepath='synthetic_call_reports.csv',
        feature_cols=['total_assets', 'total_loans', 'total_deposits', 'net_income', 'equity_ratio'],
        n_estimators=100,
        plot=True
    )
    
    print("\n" + "="*60)
    print("Demo complete! Check the generated plots and CSV files.")
    print("When your teammate has the real data, just update the filepath")
    print("and feature_cols list, and you're good to go!")
    print("="*60)