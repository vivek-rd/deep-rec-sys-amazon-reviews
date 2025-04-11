import torch
import sqlite3
import numpy as np
import pandas as pd
import csv
from modeling.NRCMA import NRCMA, NRCMAConfig
from utils.final_dataset import train_dataset

model_checkpoint = torch.load("checkpoints/self_attention_word_embedding_tuning_model_checkpoint_final.pt")
print(model_checkpoint['model_config'])

config = NRCMAConfig.from_config(model_checkpoint['model_config']['m'])
glove_embeddings = torch.load('data/required_embeddings.pt').to(torch.float32)
model = NRCMA(config, glove_embeddings)
model.eval()

def create_vector_db(db_file_path):
    """
    Create a SQLite database with tables for user and item embeddings
    """
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Create user table with id and embedding columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY,
        embedding BLOB
    )
    ''')
    
    # Create item table with id, embedding, and title columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS item (
        id TEXT PRIMARY KEY,
        embedding BLOB,
        title TEXT
    )
    ''')
    
    conn.commit()
    
    return conn, cursor

def save_embedding_to_db(cursor, table, id_value, embedding, title=None):
    """
    Save embedding to database table
    
    Args:
        cursor: Database cursor
        table: Table name ('user' or 'item')
        id_value: ID of the user or item
        embedding: Embedding tensor
        title: Title of the item (only for item table)
    """
    # Convert PyTorch tensor to numpy and then to binary blob
    embedding_blob = embedding.detach().cpu().numpy().tobytes()
    
    # Insert or replace in case the ID already exists
    if table == 'user':
        cursor.execute(f"INSERT OR REPLACE INTO {table} (id, embedding) VALUES (?, ?)",
                      (id_value, embedding_blob))
    elif table == 'item':
        cursor.execute(f"INSERT OR REPLACE INTO {table} (id, embedding, title) VALUES (?, ?, ?)",
                      (id_value, embedding_blob, title))

def load_item_titles(csv_file_path):
    """
    Load item titles from a CSV file
    
    Args:
        csv_file_path: Path to the CSV file containing item IDs and titles
        
    Returns:
        Dictionary mapping item IDs to titles
    """
    item_titles = {}
    
    try:
        # Read CSV file using pandas
        df = pd.read_csv(csv_file_path)
        
        # Check if required columns exist
        if 'item_id' not in df.columns or 'title' not in df.columns:
            print(f"Warning: CSV file {csv_file_path} missing required columns 'item_id' and/or 'title'")
            return item_titles
        
        # Convert to dictionary
        item_titles = dict(zip(df['item_id'], df['title']))
        print(f"Loaded {len(item_titles)} item titles from {csv_file_path}")
        
    except Exception as e:
        print(f"Error loading item titles from {csv_file_path}: {e}")
    
    return item_titles

def get_embeddings(model, train_dataset, item_titles_csv=None, db_file_path="embeddings.db"):
    """
    Extract embeddings from model and save to SQLite database
    
    Args:
        model: The NRCMA model
        train_dataset: Dataset to iterate over
        item_titles_csv: Path to CSV file containing item IDs and titles
        db_file_path: Path to save the SQLite database
    """
    # Create SQLite database
    conn, cursor = create_vector_db(db_file_path)
    
    # Load item titles if provided
    item_titles = {}
    if item_titles_csv:
        item_titles = load_item_titles(item_titles_csv)
    
    user_ids, item_ids = set(), set()
    
    for data in train_dataset:
        user_tower_input, item_tower_input, rating, user, item, item_id = data
        user_id = user.item()
        
        if user_id not in user_ids:
            user_ids.add(user_id)
            
            user_embedding = model.user_embedding(user)
            user_input = model.word2vec(user_tower_input.to(torch.int))
            d_u = model.process_single_tower(user_input, user_embedding, tower='user')
            
            # Add user.item() and the generated embedding to user table
            save_embedding_to_db(cursor, 'user', user_id, d_u)
        
        if item_id not in item_ids:
            item_ids.add(item_id)
            
            item_embedding = model.item_embedding(item)
            item_input = model.word2vec(item_tower_input.to(torch.int))
            d_i = model.process_single_tower(item_input, item_embedding, tower='item')
            
            # Get item title from the dictionary if it exists
            item_title = item_titles.get(item_id, None)
            
            # Add item.item() and the generated embedding to item table
            save_embedding_to_db(cursor, 'item', item_id, d_i, title=item_title)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database saved to {db_file_path}")
    print(f"Total users: {len(user_ids)}, Total items: {len(item_ids)}")
    
    return db_file_path

def calculate_embedding_similarity(user_id, item_id, db_file_path):
    """
    Calculate dot product similarity between user and item embeddings
    
    Args:
        user_id: ID of the user
        item_id: ID of the item
        db_file_path: Path to the SQLite database file
        
    Returns:
        Dot product of user and item embeddings
    """
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Get user embedding
    cursor.execute("SELECT embedding FROM user WHERE id = ?", (user_id,))
    user_embedding_blob = cursor.fetchone()
    
    if not user_embedding_blob:
        conn.close()
        raise ValueError(f"User ID {user_id} not found in database")
    
    # Get item embedding
    cursor.execute("SELECT embedding FROM item WHERE id = ?", (item_id,))
    item_embedding_blob = cursor.fetchone()
    
    if not item_embedding_blob:
        conn.close()
        raise ValueError(f"Item ID {item_id} not found in database")
    
    # Convert binary blobs back to numpy arrays
    user_embedding = np.frombuffer(user_embedding_blob[0], dtype=np.float32)
    item_embedding = np.frombuffer(item_embedding_blob[0], dtype=np.float32)
    
    # Calculate dot product
    similarity = model.factorization_machine(user_embedding, item_embedding)
    
    conn.close()
    
    return similarity

# Example usage:
# Process data and save to database
# Specify the path to your item titles CSV file
item_titles_csv = "data/item_titles_appliances.csv"

# Generate embeddings and save to database with titles from CSV
db_file = get_embeddings(model, train_dataset, 
                        item_titles_csv=item_titles_csv,
                        db_file_path="data/recommendation_embeddings.db")

# Example of calculating similarity
sample_user_id = 139681  # Replace with actual user ID
sample_item_id = 'B09W5PMK5X'  # Replace with actual item ID

try:
    similarity_score = calculate_embedding_similarity(sample_user_id, sample_item_id, db_file)
    
    # Connect to database to retrieve item title
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM item WHERE id = ?", (sample_item_id,))
    title_result = cursor.fetchone()
    item_title = title_result[0] if title_result and title_result[0] else "Unknown Title"
    conn.close()
    
    print(f"Similarity between user {sample_user_id} and item {sample_item_id} ({item_title}): {similarity_score}")
except ValueError as e:
    print(f"Error: {e}")