# unsupervised-gtzan-vae
Unsupervised Music Genre Clustering on the GTZAN Dataset using Variational Autoencoders
This repository implements unsupervised music genre clustering on the classic GTZAN dataset using Variational Autoencoders (VAEs). I explore progressive VAE variants: basic VAE on audio features, enhanced VAE with hybrid (audio + simulated lyrics) features, and advanced models (Beta-VAE and Conditional VAE).
The project extracts latent representations from VAEs and evaluates clustering performance with algorithms like K-Means, Agglomerative Clustering, and DBSCAN, using metrics such as Silhouette Score, Davies-Bouldin Index, and Adjusted Rand Index (ARI). Visualizations include PCA, t-SNE, and UMAP projections of latent spaces.
Note: The implementation does not strictly follow the originally suggested file naming (01_audio_vae.py, 02_hybrid_vae.py, 03_advanced_vae.py). Instead, the code is organized across several Python scripts (originally converted from Colab notebooks). Below is a clear description of each main file to help you navigate the repository.
Repository Structure and File Descriptions
dataset_preprocess.py
Handles dataset loading, verification, and preprocessing.
Mounts Google Drive and checks GTZAN folder structure.
Extracts 13 mean MFCC features from 30-second audio clips using Librosa.
Normalizes features, splits into train/test (800/199), creates simulated lyrics embeddings, and saves processed arrays (e.g., X_gtzan.npy, X_hybrid.npy).
eda.py
Performs Exploratory Data Analysis on the GTZAN dataset and extracted features.
Visualizes genre distribution, MFCC statistics, boxplots by genre, and correlation heatmaps.
Provides insights into data balance and feature characteristics.
audio_vae.py (or equivalent script, e.g., easy_task.py if retained)
Implements and trains a basic VAE on audio-only MFCC features.
Tests different latent dimensions (2–20).
Extracts latent representations, applies K-Means clustering, and compares against PCA baseline.
Generates visualizations (latent space colored by genres/clusters, t-SNE, UMAP).
hybrid_vae.py (or equivalent, e.g., medium_task.py)
Implements enhanced VAE architectures on hybrid features (MFCCs + 50D simulated lyrics embeddings).
Compares simple, enhanced (with BatchNorm/Dropout), and deep variants.
Applies multiple clustering algorithms (K-Means, Agglomerative, DBSCAN).
Includes ablation study (audio-only vs. lyrics-only vs. hybrid) and 2D visualizations.
advanced_vae.py (or equivalent, e.g., hard_task.py)
Implements Beta-VAE (β=4.0 for disentanglement) and Conditional VAE (conditioned on one-hot genre labels).
Trains on hybrid features, extracts latents, and evaluates clustering (primarily on Beta-VAE).
Generates advanced visualizations (PCA/t-SNE/UMAP of CVAE latents), contingency heatmaps, and reconstruction plots.
requirements
List of dependencies: torch, librosa, numpy, scikit-learn, matplotlib, seaborn, umap-learn, etc.
