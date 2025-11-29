import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import os
from data_preprocessing import HateSpeechPreprocessor

class ModelEvaluator:
    def __init__(self):
        self.preprocessor = HateSpeechPreprocessor()
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neutral']
        self.colors = ['#e74c3c', '#f39c12', '#2ecc71']
        self.algorithms = ['rf', 'svm', 'nb', 'dt', 'knn']
        self.algorithm_names = {
            'rf': 'Random Forest',
            'svm': 'SVM',
            'nb': 'Naïve Bayes',
            'dt': 'Decision Tree',
            'knn': 'K-Nearest Neighbors'
        }
        
    def load_models_and_data(self):
        """Load trained models and test data"""
        print("📥 Loading models and data...")
        
        # Load test data
        X_train, X_test, y_train, y_test, _ = self.preprocessor.prepare_training_data(
            self.preprocessor.load_kaggle_dataset('labeled_data.csv')
        )
        
        # Load vectorizers and models
        models = {}
        vectorizers = {}
        
        for algo in self.algorithms:
            try:
                models[algo] = joblib.load(f'models/hate_speech_{algo}_model.joblib')
                vectorizers[algo] = joblib.load(f'models/tfidf_vectorizer_{algo}.joblib')
                print(f"✅ Loaded {self.algorithm_names[algo]}")
            except Exception as e:
                print(f"❌ Error loading {algo}: {e}")
        
        return models, vectorizers, X_test, y_test
    
    def plot_confusion_matrices(self, models, vectorizers, X_test, y_test):
        """Generate confusion matrices for all models"""
        print("\n📊 Generating Confusion Matrices...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for idx, algo in enumerate(self.algorithms):
            if algo not in models:
                continue
                
            # Transform test data
            X_test_tfidf = vectorizers[algo].transform(X_test)
            
            # Get predictions
            y_pred = models[algo].predict(X_test_tfidf)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       ax=axes[idx])
            
            axes[idx].set_title(f'Figure 6.{idx+1} - {self.algorithm_names[algo]}\nConfusion Matrix', 
                              fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
            
            # Calculate accuracy for this model
            accuracy = accuracy_score(y_test, y_pred)
            axes[idx].text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
                          ha='center', transform=axes[idx].transAxes, fontsize=12)
        
        # Remove empty subplot
        if len(self.algorithms) < 6:
            axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, models, vectorizers, X_test, y_test):
        """Generate ROC curves for all models"""
        print("\n📈 Generating ROC Curves...")
        
        plt.figure(figsize=(12, 8))
        
        # Binarize the output for ROC curve
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        
        for algo in self.algorithms:
            if algo not in models:
                continue
                
            # Transform test data
            X_test_tfidf = vectorizers[algo].transform(X_test)
            
            # Get predicted probabilities
            y_score = models[algo].predict_proba(X_test_tfidf)
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # Plot ROC curve
            plt.plot(fpr["micro"], tpr["micro"],
                    label=f'{self.algorithm_names[algo]} (AUC = {roc_auc["micro"]:.3f})',
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Figure 6.6 - ROC Curve Comparison\n(One-vs-Rest Micro-average)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, models, vectorizers, X_test, y_test):
        """Generate Precision-Recall curves for all models"""
        print("\n📉 Generating Precision-Recall Curves...")
        
        plt.figure(figsize=(12, 8))
        
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        
        for algo in self.algorithms:
            if algo not in models:
                continue
                
            # Transform test data
            X_test_tfidf = vectorizers[algo].transform(X_test)
            
            # Get predicted probabilities
            y_score = models[algo].predict_proba(X_test_tfidf)
            
            # Compute Precision-Recall curve
            precision = dict()
            recall = dict()
            average_precision = dict()
            
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_test_bin.ravel(), y_score.ravel()
            )
            average_precision["micro"] = auc(recall["micro"], precision["micro"])
            
            # Plot Precision-Recall curve
            plt.plot(recall["micro"], precision["micro"],
                    label=f'{self.algorithm_names[algo]} (AP = {average_precision["micro"]:.3f})',
                    linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Figure 6.7 - Precision-Recall Curve Comparison\n(Micro-average)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_accuracy_comparison(self, models, vectorizers, X_test, y_test):
        """Generate accuracy comparison bar chart"""
        print("\n📊 Generating Accuracy Comparison...")
        
        accuracies = []
        
        for algo in self.algorithms:
            if algo not in models:
                accuracies.append(0)
                continue
                
            X_test_tfidf = vectorizers[algo].transform(X_test)
            y_pred = models[algo].predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar([self.algorithm_names[algo] for algo in self.algorithms], 
                      accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.0)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Figure 6.8 - Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracies
    
    def plot_f1_comparison(self, models, vectorizers, X_test, y_test):
        """Generate F1-score comparison bar chart"""
        print("\n📈 Generating F1-Score Comparison...")
        
        f1_scores = []
        
        for algo in self.algorithms:
            if algo not in models:
                f1_scores.append(0)
                continue
                
            X_test_tfidf = vectorizers[algo].transform(X_test)
            y_pred = models[algo].predict(X_test_tfidf)
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores.append(f1)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar([self.algorithm_names[algo] for algo in self.algorithms], 
                      f1_scores, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        
        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.0)
        plt.ylabel('F1-Score (Weighted)', fontsize=12)
        plt.title('Figure 6.9 - Model F1-Score Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return f1_scores
    
    def plot_feature_importance(self, vectorizers):
        """Generate feature importance plot for Random Forest"""
        print("\n🔍 Generating Feature Importance Plot...")
        
        try:
            # Load Random Forest model
            rf_model = joblib.load('models/hate_speech_rf_model.joblib')
            vectorizer = vectorizers['rf']
            
            # Get feature importance
            feature_importance = rf_model.feature_importances_
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top 20 features
            top_indices = np.argsort(feature_importance)[-20:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            plt.figure(figsize=(12, 10))
            bars = plt.barh(range(len(top_features)), top_importance, color='#3498db')
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title('Figure 6.10 - Top 20 Feature Importance (Random Forest)', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generating feature importance: {e}")
    
    def plot_misclassification_analysis(self, models, vectorizers, X_test, y_test):
        """Generate misclassification analysis chart"""
        print("\n🔍 Generating Misclassification Analysis...")
        
        misclassification_patterns = {}
        
        for algo in self.algorithms:
            if algo not in models:
                continue
                
            X_test_tfidf = vectorizers[algo].transform(X_test)
            y_pred = models[algo].predict(X_test_tfidf)
            
            # Count misclassifications by true-predicted pairs
            for true_label, pred_label in zip(y_test, y_pred):
                if true_label != pred_label:
                    pattern = f"{self.class_names[true_label]}→{self.class_names[pred_label]}"
                    misclassification_patterns[pattern] = misclassification_patterns.get(pattern, 0) + 1
        
        # Prepare data for plotting
        patterns = list(misclassification_patterns.keys())
        counts = list(misclassification_patterns.values())
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(patterns, counts, color='#e74c3c')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Figure 6.11 - Misclassification Patterns Across All Models', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('misclassification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, models, vectorizers, X_test, y_test):
        """Generate comprehensive performance report"""
        print("\n📋 Generating Comprehensive Performance Report...")
        
        results = []
        
        for algo in self.algorithms:
            if algo not in models:
                continue
                
            X_test_tfidf = vectorizers[algo].transform(X_test)
            y_pred = models[algo].predict(X_test_tfidf)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'Algorithm': self.algorithm_names[algo],
                'Accuracy': f"{accuracy:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1:.4f}"
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Save to CSV
        results_df.to_csv('comprehensive_model_performance.csv', index=False)
        print(f"\n💾 Results saved to 'comprehensive_model_performance.csv'")
        
        return results_df

def main():
    """Main evaluation function"""
    print("🚀 HATE SPEECH DETECTION MODEL EVALUATION")
    print("="*60)
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models and data
    models, vectorizers, X_test, y_test = evaluator.load_models_and_data()
    
    # Generate all diagrams
    evaluator.plot_confusion_matrices(models, vectorizers, X_test, y_test)
    evaluator.plot_roc_curves(models, vectorizers, X_test, y_test)
    evaluator.plot_precision_recall_curves(models, vectorizers, X_test, y_test)
    evaluator.plot_accuracy_comparison(models, vectorizers, X_test, y_test)
    evaluator.plot_f1_comparison(models, vectorizers, X_test, y_test)
    evaluator.plot_feature_importance(vectorizers)
    evaluator.plot_misclassification_analysis(models, vectorizers, X_test, y_test)
    
    # Generate comprehensive report
    evaluator.generate_comprehensive_report(models, vectorizers, X_test, y_test)
    
    print("\n✅ All diagrams and reports generated successfully!")
    print("📊 Generated files:")
    print("   - confusion_matrices.png")
    print("   - roc_curves.png")
    print("   - precision_recall_curves.png")
    print("   - accuracy_comparison.png")
    print("   - f1_comparison.png")
    print("   - feature_importance.png")
    print("   - misclassification_analysis.png")
    print("   - comprehensive_model_performance.csv")

if __name__ == "__main__":
    main()