import csv
from datasets import load_dataset
import pandas as pd
import os

def extract_amazon_item_metadata(category="All_Beauty", output_file="item_titles.csv"):
    """
    Extracts parent_asin (as item_id) and title from Amazon dataset and saves to CSV
    
    Args:
        category: Category of Amazon products to extract
        output_file: Path to save the CSV file
    """
    print(f"Loading Amazon Reviews 2023 dataset for category: {category}")
    
    # Load the dataset
    try:
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                               f"raw_meta_{category}", 
                               split="full", 
                               trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Dataset loaded with {len(dataset)} items")
    
    # Sample the first record to verify structure
    print("Sample record structure:")
    print(dataset[0])
    
    # Create a directory for the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Extract and write to CSV with progress tracking
    total_items = len(dataset)
    processed = 0
    skipped = 0
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['item_id', 'title'])
        
        for item in dataset:
            try:
                # Check if the required fields exist
                if 'parent_asin' in item and 'title' in item and item['parent_asin'] and item['title']:
                    writer.writerow([item['parent_asin'], item['title']])
                    processed += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing item: {e}")
                skipped += 1
            
            # Print progress every 10,000 items
            if (processed + skipped) % 10000 == 0:
                progress = (processed + skipped) / total_items * 100
                print(f"Progress: {progress:.2f}% ({processed + skipped}/{total_items})")
    
    print(f"Processing complete!")
    print(f"Total items: {total_items}")
    print(f"Processed items: {processed}")
    print(f"Skipped items: {skipped}")
    print(f"Output saved to: {output_file}")
    
    # Verify the output with pandas
    try:
        df = pd.read_csv(output_file)
        print(f"CSV file statistics:")
        print(f"  - Total rows: {len(df)}")
        print(f"  - Unique item_ids: {df['item_id'].nunique()}")
        print(f"  - First 5 records:")
        print(df.head())
    except Exception as e:
        print(f"Error verifying output: {e}")

def remove_duplicates(input_file="item_titles.csv", output_file="item_titles_unique.csv"):
    """
    Removes duplicate item_ids from the CSV file, keeping the first occurrence
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the deduplicated CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"Original file has {len(df)} rows and {df['item_id'].nunique()} unique item_ids")
        
        # Remove duplicates, keeping the first occurrence
        df_unique = df.drop_duplicates(subset='item_id', keep='first')
        
        # Save to new file
        df_unique.to_csv(output_file, index=False)
        
        print(f"Deduplicated file saved to {output_file}")
        print(f"Deduplicated file has {len(df_unique)} rows")
        print(f"Removed {len(df) - len(df_unique)} duplicate entries")
        
    except Exception as e:
        print(f"Error removing duplicates: {e}")

if __name__ == "__main__":
    # Configuration
    category = "Appliances"  # Change this to the desired category
    output_file = "../data/item_titles_appliances.csv"
    
    # Extract the data
    extract_amazon_item_metadata(category, output_file)
    
    # Remove duplicates
    remove_duplicates(output_file, output_file.replace('.csv', '_unique.csv'))