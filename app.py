import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from inference.generate_embeddings import calculate_embedding_similarity

def get_recommendations(user_id, search_query, num_recommendations, db_path):
    """
    Get product recommendations based on user ID and search query
    
    Args:
        user_id: ID of the user
        search_query: Text to search for in product titles
        num_recommendations: Number of recommendations to return
        db_path: Path to the SQLite database
        
    Returns:
        DataFrame with recommended products
    """
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT 1 FROM user WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            return None, "User ID not found in database"
        
        # First, filter products by title match (case insensitive)
        search_pattern = f"%{search_query.lower()}%"
        cursor.execute(
            "SELECT id, title FROM item WHERE lower(title) LIKE ?", 
            (search_pattern,)
        )
        # restrict to first 100 items
        matching_items = cursor.fetchall()[:100]
        
        if not matching_items:
            return None, f"No products found matching '{search_query}'"
        
        # Calculate similarity scores for matching items
        item_scores = []
        for item_id, title in matching_items:
            try:
                similarity = calculate_embedding_similarity(user_id, item_id, conn)
                item_scores.append((item_id, title, similarity))
            except Exception as e:
                # Skip items with errors
                continue
        
        # Sort by similarity score (descending)
        item_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Limit to requested number of recommendations
        top_recommendations = item_scores[:num_recommendations]
        
        # Create DataFrame for display
        if top_recommendations:
            df = pd.DataFrame(top_recommendations, columns=["Item ID", "Title", "Similarity Score"])
            return df, None
        else:
            return None, "No recommendations could be calculated"
        
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

# Function to generate a placeholder image
def get_placeholder_image(item_id, size=(200, 200)):
    """Generate a placeholder image with the item ID"""
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    # Create a random background color based on item_id
    random.seed(item_id)
    bg_color = (
        random.randint(200, 240),
        random.randint(200, 240),
        random.randint(200, 240)
    )
    text_color = (50, 50, 50)
    
    # Create image
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add item ID text
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    item_text = f"Item #{item_id}"
    text_width, text_height = draw.textsize(item_text, font=font) if hasattr(draw, 'textsize') else (80, 20)
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, item_text, fill=text_color, font=font)
    
    # Convert to base64 for displaying in Streamlit
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Product Recommendation System",
        page_icon="🛍️",
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 20px;
        margin-bottom: 10px;
    }
    .score {
        font-weight: bold;
        color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Database path
    db_path = "data/recommendation_embeddings.db"
    
    # Header
    st.markdown("<h1 class='main-header'>🛍️ Personalized Product Recommendation System</h1>", unsafe_allow_html=True)
    
    # Input section
    st.markdown("<h2 class='sub-header'>Enter Your Details</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
    
    with col2:
        num_recommendations = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)
    
    search_query = st.text_input("Search Products", placeholder="Enter product keywords...")
    
    # Search button
    search_clicked = st.button("Get Recommendations", type="primary")
    
    # Process search when button is clicked
    if search_clicked and user_id and search_query:
        with st.spinner("Finding the best products for you..."):
            # Get recommendations
            recommendations_df, error_message = get_recommendations(
                user_id, 
                search_query, 
                num_recommendations, 
                db_path
            )
            
            if error_message:
                st.error(error_message)
            elif recommendations_df is not None:
                # Display recommendations
                st.markdown("<h2 class='sub-header'>Recommended Products</h2>", unsafe_allow_html=True)
                
                # Display as cards
                for index, row in recommendations_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class='card'>
                            <h3>{row['Title']}</h3>
                            <p>Item ID: {row['Item ID']}</p>
                            <p>Similarity Score: <span class='score'>{row['Similarity Score']:.4f}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Also display as a table for comparison
                st.markdown("### Comparison Table")
                st.dataframe(
                    recommendations_df,
                    column_config={
                        "Similarity Score": st.column_config.NumberColumn(
                            "Similarity Score",
                            format="%.4f"
                        )
                    },
                    hide_index=True
                )

if __name__ == "__main__":
    main()