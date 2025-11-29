import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split

class HateSpeechPreprocessor:
    def __init__(self):
        # Basic stop words list (no NLTK dependency)
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
            "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
            'wouldn', "wouldn't"
        }
        self.punctuation = set(string.punctuation)
        
    def load_kaggle_dataset(self, file_path='labeled_data.csv'):
        """Load the Kaggle hate speech dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
            
            # Display dataset info
            print(f"📊 Columns: {df.columns.tolist()}")
            print(f"🏷️ Class distribution:")
            class_counts = df['class'].value_counts().sort_index()
            class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
            for class_id, count in class_counts.items():
                print(f"   {class_names[class_id]}: {count} samples ({count/len(df)*100:.1f}%)")
            
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def simple_tokenize(self, text):
        """Simple tokenization without NLTK"""
        return text.split()
    
    def simple_stemmer(self, word):
        """Simple stemming without NLTK"""
        if len(word) <= 3:
            return word
            
        # Common stemming rules
        if word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith('s'):
            return word[:-1]
        elif word.endswith('ly'):
            return word[:-2]
        else:
            return word
    
    def clean_tweet(self, text):
        """Clean tweet text - EXACTLY as described in your paper"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase (as mentioned in paper)
        text = text.lower()
        
        # Remove URLs (as mentioned in paper)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove usernames/@mentions (as mentioned in paper)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text (as mentioned in paper)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove HTML entities
        text = re.sub(r'&amp;', 'and', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        
        # Remove punctuation (as mentioned in paper)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra white spaces (as mentioned in paper)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove RT (retweet) markers
        text = re.sub(r'^rt\s+', '', text)
        
        return text
    
    def full_preprocess(self, text):
        """Complete preprocessing pipeline without NLTK dependencies"""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Step 1: Clean text
        cleaned_text = self.clean_tweet(text)
        
        # Step 2: Simple tokenization
        tokens = self.simple_tokenize(cleaned_text)
        
        # Step 3: Remove stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Step 4: Simple stemming
        stemmed_tokens = [self.simple_stemmer(token) for token in filtered_tokens]
        
        # Step 5: Rejoin for TF-IDF processing
        processed_text = ' '.join(stemmed_tokens)
        
        return processed_text
    
    def prepare_training_data(self, df, test_size=0.2, random_state=42):
        """Prepare data for model training"""
        print("🔄 Starting data preprocessing pipeline...")
        
        # Apply preprocessing to tweets
        df['processed_text'] = df['tweet'].apply(self.full_preprocess)
        
        # Remove empty texts after preprocessing
        original_size = len(df)
        df = df[df['processed_text'].str.len() > 0]
        removed_count = original_size - len(df)
        
        if removed_count > 0:
            print(f"⚠️  Removed {removed_count} empty texts after preprocessing")
        
        print(f"✅ Preprocessing completed. Final dataset: {len(df)} samples")
        
        # Prepare features and labels
        X = df['processed_text'].values
        y = df['class'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f"📁 Data split:")
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test, df

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = HateSpeechPreprocessor()
    
    # Load dataset
    df = preprocessor.load_kaggle_dataset('labeled_data.csv')
    
    if df is not None:
        # Test preprocessing on a sample
        sample_text = "I hate this stupid website! @user123 #annoying http://example.com"
        processed = preprocessor.full_preprocess(sample_text)
        print(f"\n🧪 Preprocessing Test:")
        print(f"   Original: {sample_text}")
        print(f"   Processed: {processed}")
        
        # Prepare training data
        X_train, X_test, y_train, y_test, processed_df = preprocessor.prepare_training_data(df)
        
        # Save processed data
        processed_df[['tweet', 'processed_text', 'class']].to_csv('processed_hate_speech_data.csv', index=False)
        print("💾 Processed data saved to 'processed_hate_speech_data.csv'")