import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

class LanguageExplorer:
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str = None):
        """
        Initialize LanguageExplorer.

        Args:
            df (pd.DataFrame): The dataframe containing text data.
            text_col (str): Name of the text column.
            label_col (str): Name of label column (optional, for class-wise plots).
        """
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.df['text_length'] = self.df[text_col].apply(len)
        self.df['num_tokens'] = self.df[text_col].apply(lambda x: len(nltk.word_tokenize(x)))
        self.df['word_count'] = self.df[text_col].apply(lambda x: len(x.split()))

    def show_dataframe_info(self):
        print("Dataframe Info:")
        print("-" * 50)
        print(self.df.info())

    def show_basic_stats(self):
        print(f"Number of documents: {len(self.df)}")
        print(f"Avg text length (chars): {self.df['text_length'].mean():.2f}")
        print(f"Avg number of tokens: {self.df['num_tokens'].mean():.2f}")
        print(f"Max text length: {self.df['text_length'].max()}")
        print(f"Min text length: {self.df['text_length'].min()}")
        print(f"Number of duplicate texts: {self.df.duplicated(subset=[self.text_col]).sum()}")

    def plot_text_length_distribution(self, bins='auto', weights=None, stat='count', kde=True, log_scale=False):
        plt.figure(figsize=(10, 5))
        sns.histplot(
            self.df['text_length'], 
            bins=bins, 
            weights=weights, 
            stat=stat, 
            kde=kde, 
            log_scale=log_scale
        )
        plt.title("Text Length Distribution")
        plt.xlabel("Text Length (chars)")
        plt.ylabel(stat.capitalize())
        plt.show()

    def plot_token_count_distribution(self, bins='auto', weights=None, stat='count', kde=True, log_scale=False):
        plt.figure(figsize=(10, 5))
        sns.histplot(
            self.df['num_tokens'], 
            bins=bins, 
            weights=weights, 
            stat=stat, 
            kde=kde, 
            log_scale=log_scale
        )
        plt.title("Token Count Distribution")
        plt.xlabel("Number of tokens")
        plt.ylabel(stat.capitalize())
        plt.show()

    def plot_word_count_histogram(self, bins='auto', weights=None, stat='count', log_scale=False):
        plt.figure(figsize=(8, 5))
        sns.histplot(
            self.df['word_count'], 
            bins=bins, 
            weights=weights, 
            stat=stat, 
            log_scale=log_scale, 
            edgecolor='black', 
            color='#4C72B0'
        )
        plt.title('Histogram of Word Counts per Description')
        plt.xlabel('Number of Words')
        plt.ylabel(stat.capitalize())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_top_categories(self, cat_col: str, top_n=10):
        plt.figure(figsize=(10, 6))
        self.df[cat_col].value_counts().head(top_n).plot(kind='barh', color='skyblue')
        plt.title(f"Top {top_n} Categories in '{cat_col}'")
        plt.xlabel("Frequency")
        plt.ylabel(cat_col)
        plt.tight_layout()
        plt.show()

    def plot_score_histogram(self, score_col: str, bins='auto', weights=None, stat='count', log_scale=False):
        self.df[score_col] = pd.to_numeric(self.df[score_col], errors='coerce')
        valid_scores = self.df[score_col].dropna()

        plt.figure(figsize=(8, 5))
        sns.histplot(
            valid_scores, 
            bins=bins, 
            weights=weights, 
            stat=stat, 
            log_scale=log_scale, 
            edgecolor='black', 
            color='#E84C3D'
        )
        plt.title(f"Histogram of '{score_col}'")
        plt.xlabel(score_col)
        plt.ylabel(stat.capitalize())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_top_k_words(self, k=20, stopwords=None):
        if stopwords is None:
            stopwords = set(nltk.corpus.stopwords.words('english'))

        all_tokens = [
            token.lower()
            for text in self.df[self.text_col]
            for token in nltk.word_tokenize(text)
            if token.isalpha() and token.lower() not in stopwords
        ]
        counter = Counter(all_tokens)
        common_words = counter.most_common(k)

        words, counts = zip(*common_words)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(counts), y=list(words))
        plt.title(f"Top {k} Most Frequent Words")
        plt.xlabel("Frequency")
        plt.ylabel("Word")
        plt.show()

    def plot_ngram_frequencies(self, n=2, k=20, stopwords=None):
        if stopwords is None:
            stopwords = set(nltk.corpus.stopwords.words('english'))

        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=stopwords)
        X = vectorizer.fit_transform(self.df[self.text_col])
        sum_words = X.sum(axis=0)

        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:k]

        ngrams, counts = zip(*words_freq)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(counts), y=list(ngrams))
        plt.title(f"Top {k} {n}-grams")
        plt.xlabel("Frequency")
        plt.ylabel("N-gram")
        plt.show()

    def plot_tfidf_top_terms(self, k=20, stopwords=None):
        if stopwords is None:
            stopwords = set(nltk.corpus.stopwords.words('english'))

        vectorizer = TfidfVectorizer(stop_words=stopwords)
        X = vectorizer.fit_transform(self.df[self.text_col])
        mean_tfidf = np.asarray(X.mean(axis=0)).ravel()

        tfidf_scores = sorted(zip(vectorizer.get_feature_names_out(), mean_tfidf), key=lambda x: x[1], reverse=True)[:k]

        words, scores = zip(*tfidf_scores)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(scores), y=list(words))
        plt.title(f"Top {k} TF-IDF Terms")
        plt.xlabel("TF-IDF Score")
        plt.ylabel("Term")
        plt.show()

    def plot_duplicates(self):
        dup_df = self.df[self.df.duplicated(subset=[self.text_col], keep=False)]
        if dup_df.empty:
            print("No duplicate texts found.")
            return

        print(f"Found {len(dup_df)} duplicate rows:")
        display(dup_df[[self.text_col]].head(10))

    def plot_long_tail(self, stopwords=None):
        if stopwords is None:
            stopwords = set(nltk.corpus.stopwords.words('english'))

        all_tokens = [
            token.lower()
            for text in self.df[self.text_col]
            for token in nltk.word_tokenize(text)
            if token.isalpha() and token.lower() not in stopwords
        ]
        counter = Counter(all_tokens)
        words, counts = zip(*counter.most_common())

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(counts) + 1), counts)
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Long Tail Distribution of Word Frequencies")
        plt.xlabel("Rank of word")
        plt.ylabel("Frequency (log)")
        plt.show()

    def plot_classwise_token_distribution(self, bins='auto', weights=None, stat='count', kde=True):
        if self.label_col is None:
            print("Label column not provided. Please provide 'label_col' when initializing the class.")
            return

        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=self.df, 
            x='num_tokens', 
            hue=self.label_col, 
            bins=bins, 
            weights=weights, 
            stat=stat, 
            kde=kde, 
            multiple='stack'
        )
        plt.title("Token Count Distribution by Class")
        plt.xlabel("Number of tokens")
        plt.ylabel(stat.capitalize())
        plt.show()

    def show_top_matches(self, score_col: str, top_n=10):
        """Display top N rows with highest scores."""
        df_sorted = self.df.copy()
        df_sorted[score_col] = pd.to_numeric(df_sorted[score_col], errors='coerce')
        df_sorted = df_sorted.dropna(subset=[score_col])

        print(f"\nTop {top_n} Matches (by '{score_col}'):")
        display(df_sorted.sort_values(by=score_col, ascending=False).head(top_n))
