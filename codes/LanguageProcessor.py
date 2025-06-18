import os
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from sentence_transformers import SentenceTransformer, util

# Ensure stopwords are available
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))


class LanguageProcessor:
    """
    A class to perform the follows tasks: 
    1. Load datasets
    2. Text Preprocessing
    3. Calculate Lexical overlap: Jaccard Similarity 
    4. Load LLM model: e.g., SBERT
    5. Semantic Embedding
    6. Cosine Similarity
    7. Combine Lexical and Semantic similairity 
    8. Product Matching based on Combined similarity
    
    Attributes:
        model_name (str): Name of the SBERT model to use.
        lexical_weight (float): Weight for lexical similarity in combined score.
        semantic_weight (float): Weight for semantic similarity in combined score.
    """

    def __init__(self, model_name='all-mpnet-base-v2', lexical_weight=0.1):
        """
        Initialize the LanguageProcessor with a SBERT model and similarity weights.

        Args:
            model_name (str): Name of the SentenceTransformer model to load.
            lexical_weight (float): Weight for lexical similarity (0 to 1).
        """
        self.model_name = model_name
        self.lexical_weight = lexical_weight
        self.semantic_weight = 1 - lexical_weight
        self.model = SentenceTransformer(self.model_name)

    def load_data(self, filepath, n_rows=-1, __print__=True):
        """
        Load the client dataset from CSV file.

        Args:
            filepath (str): Path to the CSV file containing 'sentence1', 'sentence2', 'score'.
            n_rows (int): Number of rows to load for analysis, default is all.
        """
        df = pd.read_csv(filepath)
        self.catalog_a = df['sentence1'][:n_rows].tolist()
        self.catalog_b = df['sentence2'][:n_rows].tolist()
        self.ref_score = df['score'][:n_rows].tolist()
        self.df = df
        if __print__:
            display(df.head(5))
    
    def tokenizer(self, text):
        """
        Tokenize and clean a string:
        - Lowercase
        - Remove punctuation
        - Remove stopwords
        - Normalize whitespace
    
        Args:
            text (str): Input string.
    
        Returns:
            str: Cleaned and tokenized string.
        """
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text #' '.join(word for word in text.split() if word not in stop_words)
    
    
    def tokenize_columns(self, df: pd.DataFrame, columns: list, __print__=True) -> pd.DataFrame:
        """
        Apply tokenizer to multiple text columns in a DataFrame and add results as new columns.
    
        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of column names to tokenize.
    
        Returns:
            pd.DataFrame: DataFrame with new tokenized columns appended.
        """
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            token_col = f"{col}_token"
            df[token_col] = df[col].apply(self.tokenizer)
        
        self.df = df
        if __print__:
            return df.head(5)
        # return df

    
    def compute_pairwise_jaccard(self, 
                                 col_a: str, 
                                 col_b: str, 
                                 df: pd.DataFrame = None,
                                 __print__ = True,
                                 __plot__ = True,
                                ) -> pd.DataFrame:
        """
        Compute pairwise Jaccard similarity between two text columns in a DataFrame.
    
        Stores the result in `self.df_jaccard_sim`.
    
        Args:
            col_a (str): Column name for Catalog A sentences.
            col_b (str): Column name for Catalog B sentences.
            df (pd.DataFrame, optional): Input DataFrame. If None, uses self.df.
    
        Returns:
            pd.DataFrame: DataFrame of Jaccard similarity scores with columns:
                ['index_a', 'index_b', 'catalog_a', 'catalog_b', 
                 'catalog_a_clean', 'catalog_b_clean', 'jaccard_score']
        """
        if df is None:
            if not hasattr(self, 'df'):
                raise ValueError("No DataFrame provided and `self.df` is not set.")
            df = self.df.iloc[:10]
    
        catalog_a_raw = df[col_a].dropna().unique().tolist()
        catalog_b_raw = df[col_b].dropna().unique().tolist()
    
        catalog_a_clean = [self.tokenizer(text) for text in catalog_a_raw]
        catalog_b_clean = [self.tokenizer(text) for text in catalog_b_raw]
    
        results = []
        for i, text_a in enumerate(catalog_a_clean):
            tokens_a = set(text_a.split())
            for j, text_b in enumerate(catalog_b_clean):
                tokens_b = set(text_b.split())
                intersection = tokens_a & tokens_b
                union = tokens_a | tokens_b
                score = len(intersection) / len(union) if union else 0.0
                results.append({
                    "index_a": i,
                    "index_b": j,
                    "catalog_a": catalog_a_raw[i],
                    "catalog_b": catalog_b_raw[j],
                    "catalog_a_clean": text_a,
                    "catalog_b_clean": text_b,
                    "jaccard_score": score
                })
    
        # Save to class attribute
        self.df_jaccard_sim = pd.DataFrame(results)
        if __print__:
            display(self.df_jaccard_sim.head())

        if __plot__:
            self.show_jaccard_heatmap()
    

    def show_jaccard_heatmap(self,
                             df_jaccard_sim = None,
                             max_label_len: int = 40, 
                             figsize: tuple = (10, 6), 
                             annot: bool = True, 
                             save_path: str = '../../outputs/'):
        """
        Plot a heatmap of Jaccard similarity scores stored in `self.df_jaccard_sim`.
    
        Args:
            max_label_len (int, optional): Max length of sentence labels for display. Default = 40.
            figsize (tuple, optional): Figure size. Default = (10, 6).
            annot (bool, optional): Whether to annotate heatmap cells. Default = True.
            save_path (str, optional): If provided, save the figure to this path.
        """
        if not hasattr(self, 'df_jaccard_sim'):
            raise ValueError("You must run `compute_pairwise_jaccard()` first.")
    
        # Pivot to matrix form
        if not df_jaccard_sim:
            pivot_df = self.df_jaccard_sim.pivot(index="index_a", columns="index_b", values="jaccard_score").values
    
        # Prepare labels
        def truncate(text, max_len=max_label_len):
            return text if len(text) <= max_len else text[:max_len] + '...'
    
        labels_a = [truncate(text) for text in self.df_jaccard_sim.drop_duplicates("index_a")["catalog_a_clean"]]
        labels_b = [truncate(text) for text in self.df_jaccard_sim.drop_duplicates("index_b")["catalog_b_clean"]]
    
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_df, annot=annot, fmt=".2f", cmap="YlOrBr",
                    xticklabels=labels_b, yticklabels=labels_a)
    
        plt.title("Jaccard Similarity Heatmap (Token-Level)", fontsize=14)
        plt.xlabel("Catalog B")
        plt.ylabel("Catalog A")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        if save_path:
            plt.savefig(save_path, dpi=600)
            print(f"✅ Jaccard heatmap saved to: {save_path}")
            

    
    def show_semantic_heatmap(self, max_label_len: int = 40, figsize: tuple = (10, 6), annot: bool = True, save_path: str = None):
        """
        Plot a heatmap of cosine semantic similarity (e.g., from SBERT).
    
        Args:
            max_label_len (int): Max length of axis labels. Default is 40.
            figsize (tuple): Figure size (width, height). Default is (10, 6).
            annot (bool): Whether to annotate each cell. Default is True.
            save_path (str, optional): If provided, saves the plot to this file.
        
        Raises:
            ValueError: If `self.semantic_sim` or catalog data is missing.
        """
        if not hasattr(self, 'semantic_sim'):
            raise ValueError("`self.semantic_sim` not found. Run SBERT embedding + cosine similarity first.")
        if not hasattr(self, 'catalog_a_clean') or not hasattr(self, 'catalog_b_clean'):
            raise ValueError("Preprocessed catalogs not found. Make sure `catalog_a_clean` and `catalog_b_clean` are set.")
    
        def truncate(text, max_len=max_label_len):
            return text if len(text) <= max_len else text[:max_len] + "..."
    
        labels_a = [f"A{i+1}: {truncate(s)}" for i, s in enumerate(self.catalog_a_clean)]
        labels_b = [f"B{i+1}: {truncate(s)}" for i, s in enumerate(self.catalog_b_clean)]
    
        plt.figure(figsize=figsize)
        sns.heatmap(self.semantic_sim, annot=annot, fmt=".2f", cmap="YlGnBu",
                    xticklabels=labels_b, yticklabels=labels_a)
    
        plt.title("Semantic Similarity Heatmap (SBERT Cosine Scores)", fontsize=14)
        plt.xlabel("Catalog B Sentences")
        plt.ylabel("Catalog A Sentences")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    
        if save_path:
            plt.savefig(save_path, dpi=300)
            

    
    def load_sentence_transformer_model(self, 
                                        model_name_or_path: str = '../../models/', 
                                        local_dir: str = None):
        """
        Load a SentenceTransformer model from local directory or HuggingFace Hub.
    
        If local_dir is provided and the model is not found, it will be downloaded and saved there.
    
        Args:
            model_name_or_path (str): Model name (HF Hub) or local path.
            local_dir (str, optional): Directory to save/load local model. Default = None.
    
        Sets:
            self.model: Loaded SentenceTransformer model.
    
        Prints:
            Model config and key features.
        """
        from sentence_transformers import SentenceTransformer
        import os
    
        # If local_dir is provided, construct the path
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            model_path = os.path.join(local_dir, model_name_or_path.replace("/", "_"))
        else:
            model_path = model_name_or_path
    
        # Check if model already exists locally
        if os.path.exists(model_path):
            print(f"🔍 Loading model from local path: {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            print(f"⬇️ Downloading model from HuggingFace: {model_name_or_path}")
            self.model = SentenceTransformer(model_name_or_path)
            if local_dir:
                print(f"💾 Saving model to local path: {model_path}")
                self.model.save(model_path)
    
        # Print model config
        print("✅ Model loaded!")
        try:
            config = self.model[0].auto_model.config
            print("📄 Model Config:")
            print(f" - Model type: {config.model_type}")
            print(f" - Hidden size: {config.hidden_size}")
            print(f" - Num layers: {config.num_hidden_layers}")
            print(f" - Num attention heads: {config.num_attention_heads}")
            total_params = sum(p.numel() for p in self.model[0].auto_model.parameters())
            print(f" - Total parameters: {total_params/1e6:.1f}M")
        except Exception as e:
            print(f"⚠️ Could not print full model config: {e}")

    
    def compute_semantic_embeddings(self,
                                    df: pd.DataFrame,
                                    columns: list,
                                    __print__ = True,
                                    __plot__ = True,
                                   ) -> pd.DataFrame:
        """
        Compute semantic embeddings for specified columns using the loaded SBERT model.
        Adds new columns with suffix '_embedding' containing tensor embeddings.
    
        Args:
            df (pd.DataFrame): Input dataframe.
            columns (list): List of column names to embed.
    
        Returns:
            pd.DataFrame: DataFrame with new embedding columns added.
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not loaded. Please load the SBERT model first.")
    
        for col in columns:
            if col not in df.columns:
                print(f"Warning: column '{col}' not found in dataframe, skipping.")
                continue
    
            # Convert to string and preprocess (optional)
            text_list = df[col].astype(str).tolist()
            text_list_clean = [self.tokenizer(t) for t in text_list]
    
            # Encode with model (returns tensors)
            embeddings = self.model.encode(text_list_clean, convert_to_tensor=True)
    
            # Add embeddings column to df
            df[f"{col}_embedding"] = list(embeddings)
    
            print(f"Added embeddings for column '{col}' as '{col}_embedding'")
    
        self.df = df
        if __print__:
            display(df.head())
    
        if __plot__:
            self.plot_embeddings_pca(
                emb_col_a=f"{columns[0]}_embedding",
                emb_col_b=f"{columns[1]}_embedding",
                text_col_a=columns[0],
                text_col_b=columns[1]
            )
    

    def plot_embeddings_pca(self, 
                            emb_col_a: str, 
                            emb_col_b: str,
                            text_col_a: str = None,
                            text_col_b: str = None,
                            df: pd.DataFrame = None, 
                            max_label_len: int = 40,
                            title: str = "SBERT Embedding Visualization (PCA Reduced)"):
        """
        Plot PCA-reduced 2D scatter plot of embeddings from two embedding columns,
        with truncated text labels from corresponding text columns.
    
        Args:
            emb_col_a (str): Name of embedding column for Catalog A.
            emb_col_b (str): Name of embedding column for Catalog B.
            text_col_a (str, optional): Name of text column for Catalog A (for labels).
                                        If None, no labels plotted for A.
            text_col_b (str, optional): Name of text column for Catalog B (for labels).
                                        If None, no labels plotted for B.
            df (pd.DataFrame, optional): DataFrame containing embedding and text columns.
                                         Defaults to self.df if None.
            max_label_len (int): Max length of text labels.
            title (str): Plot title.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
    
        if df is None:
            df = self.df
    
        def truncate(text, max_len=max_label_len):
            return text if len(text) <= max_len else text[:max_len] + "..."
    
        # Extract embeddings lists from dataframe columns
        emb_a_list = df[emb_col_a].tolist()
        emb_b_list = df[emb_col_b].tolist()
    
        # Convert tensor to numpy arrays if needed
        emb_a_np = np.array([e.cpu().numpy() if hasattr(e, 'cpu') else np.array(e) for e in emb_a_list])
        emb_b_np = np.array([e.cpu().numpy() if hasattr(e, 'cpu') else np.array(e) for e in emb_b_list])
    
        # Extract and truncate text labels if columns provided
        labels_a = []
        labels_b = []
        if text_col_a and text_col_a in df.columns:
            labels_a = [truncate(str(txt)) for txt in df[text_col_a].tolist()]
        if text_col_b and text_col_b in df.columns:
            labels_b = [truncate(str(txt)) for txt in df[text_col_b].tolist()]
    
        # Combine for PCA
        all_embeddings = np.vstack((emb_a_np, emb_b_np))
    
        # PCA reduction to 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_embeddings)
    
        n = len(emb_a_np)
        reduced_a = reduced[:n]
        reduced_b = reduced[n:]
    
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=reduced_a[:, 0], y=reduced_a[:, 1], label="Catalog A", marker='o', color='blue')
        sns.scatterplot(x=reduced_b[:, 0], y=reduced_b[:, 1], label="Catalog B", marker='s', color='green')
    
        # Add text labels near points
        for i in range(n):
            if labels_a:
                plt.text(reduced_a[i, 0] + 0.01, reduced_a[i, 1], f"A{i+1}: {labels_a[i]}", fontsize=9, color='blue')
            if labels_b:
                plt.text(reduced_b[i, 0] + 0.01, reduced_b[i, 1], f"B{i+1}: {labels_b[i]}", fontsize=9, color='green')
    
        # Draw dashed lines between pairs
        for i in range(n):
            plt.plot([reduced_a[i, 0], reduced_b[i, 0]],
                     [reduced_a[i, 1], reduced_b[i, 1]],
                     color='gray', alpha=0.5, linestyle='--')
    
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    
    
    ###### FIND ME
    def compute_pairwise_cosine_similarity(self,
                                           emb_col_a: str,
                                           emb_col_b: str,
                                           raw_col_a: str = None,
                                           raw_col_b: str = None,
                                           __print__: bool = True,
                                           __plot__: bool = True
                                          ) -> pd.DataFrame:
        """
        Compute pairwise cosine similarity between two embedding columns in self.df.
    
        Args:
            emb_col_a (str): Name of embedding column for Catalog A.
            emb_col_b (str): Name of embedding column for Catalog B.
            raw_col_a (str, optional): Name of raw text column for Catalog A (for display). 
                                       If None, defaults to emb_col_a.replace('_embedding', '').
            raw_col_b (str, optional): Name of raw text column for Catalog B (for display). 
                                       If None, defaults to emb_col_b.replace('_embedding', '').
            __print__ (bool): Whether to print the top rows of resulting DataFrame.
            __plot__ (bool): Whether to plot cosine similarity heatmap.
    
        Returns:
            pd.DataFrame: DataFrame with columns:
                ['index_a', 'index_b', 'catalog_a', 'catalog_b', 'cosine_score']
        """
        if not hasattr(self, 'df'):
            raise ValueError("self.df not found. You must first load a dataframe with embeddings.")
    
        df = self.df
    
        # Load embeddings
        embeddings_a = df[emb_col_a].tolist()
        embeddings_b = df[emb_col_b].tolist()
    
        # Store raw column names for later plotting
        self.raw_col_a = raw_col_a if raw_col_a else emb_col_a.replace("_embedding", "")
        self.raw_col_b = raw_col_b if raw_col_b else emb_col_b.replace("_embedding", "")
    
        # Convert embeddings to numpy arrays (cpu-safe)
        emb_a_np = np.array([e.cpu().numpy() if hasattr(e, 'cpu') else np.array(e) for e in embeddings_a])
        emb_b_np = np.array([e.cpu().numpy() if hasattr(e, 'cpu') else np.array(e) for e in embeddings_b])
    
        # Compute cosine similarity matrix
        cos_sim_matrix = util.cos_sim(emb_a_np, emb_b_np).cpu().numpy()
    
        # Prepare raw text columns (fallback if not present)
        catalog_a_raw = df[self.raw_col_a].astype(str).tolist() if self.raw_col_a in df.columns \
                        else [f"A_{i}" for i in range(len(emb_a_np))]
        catalog_b_raw = df[self.raw_col_b].astype(str).tolist() if self.raw_col_b in df.columns \
                        else [f"B_{i}" for i in range(len(emb_b_np))]
    
        # Build result DataFrame
        results = [
            {
                "index_a": i,
                "index_b": j,
                "catalog_a": catalog_a_raw[i],
                "catalog_b": catalog_b_raw[j],
                "cosine_score": cos_sim_matrix[i, j]
            }
            for i in range(len(catalog_a_raw))
            for j in range(len(catalog_b_raw))
        ]
    
        self.df_cosine_sim = pd.DataFrame(results)
    
        if __print__:
            display(self.df_cosine_sim.head())
    
        if __plot__:
            self.show_cosine_similarity_heatmap()
    
        return self.df_cosine_sim
    
    
    def show_cosine_similarity_heatmap(self,
                                       df_cosine_sim: pd.DataFrame = None,
                                       max_label_len: int = 40,
                                       figsize: tuple = (10, 6),
                                       annot: bool = True,
                                       cmap: str = "YlGnBu",
                                       save_path: str = None
                                      ):
        """
        Plot a heatmap of cosine similarity scores.
    
        Args:
            df_cosine_sim (pd.DataFrame, optional): If None, uses self.df_cosine_sim.
            max_label_len (int): Max label length for display.
            figsize (tuple): Figure size.
            annot (bool): Whether to show annotations on heatmap.
            cmap (str): Color map.
            save_path (str, optional): If provided, saves figure to path.
        """
        if df_cosine_sim is None:
            if hasattr(self, 'df_cosine_sim'):
                df_cosine_sim = self.df_cosine_sim
            else:
                raise ValueError("You must run `compute_pairwise_cosine_similarity()` first.")
    
        # Read column names
        raw_col_a = self.raw_col_a
        raw_col_b = self.raw_col_b
    
        # Pivot matrix
        pivot_df = df_cosine_sim.pivot(index="index_a", columns="index_b", values="cosine_score").values
    
        # Label helpers
        def truncate(text, max_len=max_label_len):
            return text if len(text) <= max_len else text[:max_len] + "..."
    
        labels_a = [truncate(text) for text in df_cosine_sim.drop_duplicates("index_a")["catalog_a"]]
        labels_b = [truncate(text) for text in df_cosine_sim.drop_duplicates("index_b")["catalog_b"]]
    
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_df, annot=annot, fmt=".2f", cmap=cmap,
                    xticklabels=labels_b, yticklabels=labels_a, linewidths=0.5)
    
        plt.title("Cosine Similarity Heatmap (SBERT Embeddings)", fontsize=14)
        plt.xlabel(f"{raw_col_b}")
        plt.ylabel(f"{raw_col_a}")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=600)
            print(f"✅ Cosine similarity heatmap saved to: {save_path}")
    
        plt.show()
    
    
    def combine_similarities(self,
                             semantic_weight: float = 0.8,
                             lexical_weight: float = 0.2,
                             __print__: bool = True,
                             __plot__: bool = True):
        """
        Combine Jaccard and semantic similarities into a single score matrix.
    
        Args:
            semantic_weight (float): weight of semantic (cosine) similarity
            lexical_weight (float): weight of Jaccard similarity
            __print__ (bool): Whether to print top combined similarity pairs
            __plot__ (bool): Whether to visualize combined similarity matrix
    
        Stores:
            self.combined_sim: Combined similarity matrix (2D np.array)
        """
        # Save weights
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
    
        # Pivot to similarity matrices
        pivot_cosine = self.df_cosine_sim.pivot(index="index_a", columns="index_b", values="cosine_score").values
        pivot_jaccard = self.df_jaccard_sim.pivot(index="index_a", columns="index_b", values="jaccard_score").values
    
        # Combine matrices
        self.combined_sim = (
            self.semantic_weight * pivot_cosine +
            self.lexical_weight * pivot_jaccard
        )
    
        if __print__:
            # Display top similarity pairs (flattened)
            flat_sim = self.combined_sim.flatten()
            top_idx = flat_sim.argsort()[::-1][:10]  # top 10
            rows = top_idx // self.combined_sim.shape[1]
            cols = top_idx % self.combined_sim.shape[1]
    
            print("\n🔝 Top combined similarity pairs:")
            for r, c, score in zip(rows, cols, flat_sim[top_idx]):
                label_a = self.df.iloc[r][self.raw_col_a] if self.raw_col_a in self.df.columns else f"A_{r}"
                label_b = self.df.iloc[c][self.raw_col_b] if self.raw_col_b in self.df.columns else f"B_{c}"
                print(f"[{r}, {c}] ({label_a} ↔ {label_b}) → {score:.4f}")
    
        if __plot__:
            # Prepare labels
            def truncate(text, max_len=40):
                return text if len(text) <= max_len else text[:max_len] + "..."
    
            labels_a = self.df.drop_duplicates(subset=self.raw_col_a)[self.raw_col_a].astype(str).apply(truncate).tolist()
            labels_b = self.df.drop_duplicates(subset=self.raw_col_b)[self.raw_col_b].astype(str).apply(truncate).tolist()
    
            # Plot heatmap with annotations
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.combined_sim,
                        annot=True, fmt=".2f", cmap="coolwarm",
                        xticklabels=labels_b, yticklabels=labels_a,
                        linewidths=0.5)
    
            plt.title(f"Combined Similarity Matrix\nSemantic {self.semantic_weight:.2f} + Jaccard {self.lexical_weight:.2f}",
                      fontsize=14)
            plt.xlabel("Catalog B")
            plt.ylabel("Catalog A")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()



    def match_products(self):
        """
        Find the best matching sentence in catalog B for each sentence in catalog A.

        Returns:
            list of tuples: Each tuple contains (index_a, index_b, similarity_score).
        """
        matches = []
        for i in range(len(self.catalog_a_clean)):
            j = np.argmax(self.combined_sim[i])
            score = self.combined_sim[i, j]
            matches.append((i, j, score))
        return matches

    def visualize_similarity(self, matrix, title, labels_a=None, labels_b=None, cmap="YlGnBu"):
        """
        Visualize a similarity matrix as a heatmap.

        Args:
            matrix (np.ndarray): 2D similarity matrix.
            title (str): Title for the heatmap.
            labels_a (list[str], optional): Y-axis labels.
            labels_b (list[str], optional): X-axis labels.
            cmap (str): Color map for heatmap.
        """
        def truncate(text, max_len=40):
            return text if len(text) <= max_len else text[:max_len] + '...'

        labels_a = labels_a or [f"A{i+1}: {truncate(s)}" for i, s in enumerate(self.catalog_a_clean)]
        labels_b = labels_b or [f"B{i+1}: {truncate(s)}" for i, s in enumerate(self.catalog_b_clean)]

        plt.figure(figsize=(12, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=labels_b, yticklabels=labels_a, linewidths=0.5)
        plt.title(title)
        plt.xlabel("Catalog B")
        plt.ylabel("Catalog A")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def visualize_embeddings(self):
        """
        Visualize SBERT embeddings using PCA reduction to 2D space.
        """
        emb_a_np = self.embeddings_a.cpu().numpy()
        emb_b_np = self.embeddings_b.cpu().numpy()
        all_emb = np.vstack((emb_a_np, emb_b_np))

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_emb)
        reduced_a, reduced_b = reduced[:len(self.catalog_a_clean)], reduced[len(self.catalog_a_clean):]

        plt.figure(figsize=(10, 6))
        for i, (x, y) in enumerate(reduced_a):
            plt.scatter(x, y, c='blue', marker='o')
            plt.text(x + 0.01, y, f"A{i+1}: {self.catalog_a_clean[i][:40]}", fontsize=9, color='blue')

        for i, (x, y) in enumerate(reduced_b):
            plt.scatter(x, y, c='green', marker='s')
            plt.text(x + 0.01, y, f"B{i+1}: {self.catalog_b_clean[i][:40]}", fontsize=9, color='green')

        for i in range(len(self.catalog_a_clean)):
            plt.plot(
                [reduced_a[i, 0], reduced_b[i, 0]],
                [reduced_a[i, 1], reduced_b[i, 1]],
                linestyle='--', color='gray', alpha=0.5
            )

        plt.title("SBERT Embedding Visualization with Sentence Labels")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def save_matches_to_csv(self, filepath='matched_results.csv'):
        """
        Save matched product pairs with similarity scores to a CSV file.

        Args:
            filepath (str): Output path for the CSV file.
        """
        matches = self.match_products()
        rows = []
        for i, j, score in matches:
            rows.append({
                'catalog_a_index': i,
                'catalog_b_index': j,
                'catalog_a_text': self.catalog_a[i],
                'catalog_b_text': self.catalog_b[j],
                'catalog_a_clean': self.catalog_a_clean[i],
                'catalog_b_clean': self.catalog_b_clean[j],
                'combined_similarity_score': score
            })
        df_matches = pd.DataFrame(rows)
        df_matches.to_csv(filepath, index=False)
        print(f"✅ Matches saved to {filepath}")
    