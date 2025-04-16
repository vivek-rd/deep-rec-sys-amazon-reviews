# Deep Learning-Based Recommendation System

## Project Goal

This project implements and compares three different deep learning-based recommendation systems (DeepCoNN, NRCMA, and HSACN) that utilize the natural language of user reviews from the Amazon User Reviews dataset. The primary goal is to improve rating predictions and provide relevant product recommendations by leveraging the insights contained in review text. The system predicts user ratings for products and uses these predictions to rank relevant products based on a user's search query and past review history.

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/-W%26B-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)  
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/-NLTK-4d7a97?style=flat&logoColor=white)
![HuggingFace Datasets](https://img.shields.io/badge/-Datasets-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Gensim](https://img.shields.io/badge/-Gensim-3498DB?style=flat&logoColor=white)
![Surprise](https://img.shields.io/badge/-Surprise-FA8072?style=flat&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Torchinfo](https://img.shields.io/badge/-Torchinfo-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TQDM](https://img.shields.io/badge/-TQDM-FFA500?style=flat&logoColor=white)

## Approach and Methodology

### Models Implemented:
1.  **DeepCoNN (Deep Co-Operative Neural Networks):** This model uses two parallel convolutional neural networks (CNNs), one for user reviews and one for item reviews. It extracts semantic features from review text using convolutions and max-pooling, then uses a Factorization Machine (FM) layer to model the interaction between user and item latent representations for rating prediction.
2.  **NRCMA (Neural Recommendation with Cross-Modality Mutual Attention):** NRCMA improves upon the two-tower model by introducing cross-modality mutual attention mechanisms at both word and review levels. This allows the user and item encoders to exchange information, focusing on the most relevant words and reviews for a given user-item interaction. Embeddings are generated using pre-trained GloVe vectors and processed through CNNs before attention layers. The final prediction is made using an FM layer.
3.  **HSACN (Hierarchical Self-attentive Convolution Network):** HSACN uses a hierarchical approach, encoding words into sentences, sentences into reviews, and reviews into final user/item representations. It combines CNNs for local feature extraction and self-attention mechanisms for aggregation at different levels (sentence, review, entity). This structure allows the model to weigh different parts of the text based on their importance.

### Data Processing Pipeline:
- Raw review data filtered to include only unique user-item pairs.
- Reviews embedded using pretrained word embeddings (Google-Word2Vec-300).
- Data structured separately for each model’s specific requirements.

### Training and Evaluation:
- Models trained on Nvidia GPUs (V100-SXM2, T4) and Apple silicon (M1, M2).
- Evaluated using Mean Squared Error (MSE).

## How to Run the Code

### Step-by-step Guide

1. **Clone the Repository**
```bash
git clone https://github.com/your-repo/deep-rec-sys-amazon-reviews.git
cd deep-rec-sys-amazon-reviews
```

2. **Set Up the Environment**
- Ensure Python 3.12 is installed.
- Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r pyproject.toml
```

3. **Data Preparation**
- Obtain the Amazon Reviews dataset (Appliances category) from [HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
- Preprocess and filter the data using scripts in the `utils` folder:
```bash
python utils/data_loading.py
```

4. **Model Training**
- Train DeepCoNN, NRCMA, and HSACN models:
```bash
python -m modeling.DeepCoNN_train
python -m modeling.NRCMA_train
python -m modeling.HSACN_train
```

5. **Generate Embeddings**
- Generate and store embeddings:
```bash
python inference/generate_embeddings.py
```

6. **Run the Application**
- Launch the Streamlit app:
```bash
streamlit run app.py
```

## Requirements and Dependencies
- Python 3.12
- PyTorch
- numpy, pandas, nltk, datasets, gensim, surprise, matplotlib, torchinfo, tqdm, wandb
- Detailed dependencies available in `pyproject.toml`

## Results and Outputs

- **Best Performing Model:** NRCMA, demonstrating improved user-item interaction modeling with cross-attention, achieving the lowest MSE loss of 1.57.
- Evaluation results stored and visualized using Weights and Biases (W&B).

## Limitations

- Evaluation limited to the 'Appliances' category.
- Does not explicitly handle the cold-start problem for new users/items (< 2 reviews).
- Computational cost, particularly for HSACN, might limit scalability.

## Further Development

- Exploring additional architectures and enhancing scalability.
- Integration of richer metadata for improved recommendation accuracy.

For further details, please refer to the complete [Final Project Report](reports/Final_Project_Report.pdf).

