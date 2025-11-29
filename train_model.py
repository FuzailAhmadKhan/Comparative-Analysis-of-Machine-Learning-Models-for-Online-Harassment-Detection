import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

from data_preprocessing import HateSpeechPreprocessor

class HateSpeechModel:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.preprocessor = HateSpeechPreprocessor()
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neither']
    
    def load_and_preprocess_data(self, file_path='labeled_data.csv'):
        """Load and preprocess data"""
        print("📥 Loading and preprocessing data...")
        X_train, X_test, y_train, y_test, df = self.preprocessor.prepare_training_data(
            self.preprocessor.load_kaggle_dataset(file_path)
        )
        return X_train, X_test, y_train, y_test
    
    def create_tfidf_features(self, X_train, X_test, feature_type='bigram'):
        """Create TF-IDF features"""
        if feature_type == 'bigram':
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),      # Unigram + Bigram
                max_features=5000,
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )
        else:  # unigram
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),      # Only Unigram
                max_features=5000,
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )
        
        print(f"🔤 Creating {feature_type} TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"   Training features: {X_train_tfidf.shape}")
        print(f"   Test features: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train_classifier(self, X_train, y_train, algorithm='rf'):
        """Train the selected classifier"""
        if algorithm == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algorithm == 'svm':
            self.model = SVC(kernel='linear', probability=True, random_state=42)
        elif algorithm == 'nb':
            self.model = MultinomialNB(alpha=0.1)
        elif algorithm == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif algorithm == 'dt':
            self.model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        print(f"🤖 Training {algorithm.upper()} classifier...")
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test, algorithm_name):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📈 {algorithm_name.upper()} Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return {
            'algorithm': algorithm_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, algorithm):
        """Save trained model and vectorizer"""
        os.makedirs('models', exist_ok=True)
        
        model_filename = f'models/hate_speech_{algorithm}_model.joblib'
        vectorizer_filename = f'models/tfidf_vectorizer_{algorithm}.joblib'
        
        joblib.dump(self.model, model_filename)
        joblib.dump(self.vectorizer, vectorizer_filename)
        
        print(f"💾 Model saved: {model_filename}")
        print(f"💾 Vectorizer saved: {vectorizer_filename}")

def main():
    """Main training function"""
    print("🚀 HATE SPEECH DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # Initialize model
    hs_model = HateSpeechModel()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = hs_model.load_and_preprocess_data('labeled_data.csv')
    
    # Algorithms to compare
    algorithms = ['rf', 'svm', 'nb', 'knn', 'dt']
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*40}")
        print(f"🧠 Training {algo.upper()}")
        print(f"{'='*40}")
        
        # Create features
        X_train_tfidf, X_test_tfidf = hs_model.create_tfidf_features(X_train, X_test, 'bigram')
        
        # Train model
        hs_model.train_classifier(X_train_tfidf, y_train, algo)
        
        # Evaluate model
        result = hs_model.evaluate_model(X_test_tfidf, y_test, algo)
        results[algo] = result
        
        # Save model
        hs_model.save_model(algo)
    
    # Display comparison
    print(f"\n{'='*50}")
    print(f"🏆 FINAL RESULTS COMPARISON")
    print(f"{'='*50}")
    
    comparison_df = pd.DataFrame.from_dict(results, orient='index')
    comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1_score']].round(4)
    print(comparison_df)
    
    # Save results
    comparison_df.to_csv('model_results.csv')
    print("💾 Results saved to 'model_results.csv'")
    
    # Find best model
    best_algo = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🎯 BEST MODEL: {best_algo[0].upper()} (Accuracy: {best_algo[1]['accuracy']:.4f})")

if __name__ == "__main__":
    main()