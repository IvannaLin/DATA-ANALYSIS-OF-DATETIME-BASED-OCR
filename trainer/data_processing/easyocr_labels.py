import pandas as pd
import os
import re

def process_csv(input_path, output_dir='output'):
    """Process CSV file and split into multiple files grouped by directory.
    
    Args:
        input_path (str): Path to input CSV file
        output_dir (str): Output directory for split files (default 'output')
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        
        # Validate required columns exist
        required_columns = ['Directory', 'Filename', 'Parsed Timestamp']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input CSV is missing required columns: {missing_cols}")
        
        # Filter rows with valid timestamps (non-empty and not NaN)
        valid_df = df[df['Parsed Timestamp'].notna() & (df['Parsed Timestamp'] != '')]
        
        # Check if we have any valid data
        if valid_df.empty:
            print("Warning: No rows with valid timestamps found in the input file.")
            return
        
        # Extract directory names and group by them
        grouped = valid_df.groupby('Directory')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each directory group
        for dir_name, group in grouped:
            # Clean directory name for safe filename use
            clean_dir_name = re.sub(r'[^\w\-_]', '', dir_name.replace(' ', '_'))
            output_filename = f"{clean_dir_name}_timestamps.csv"
            
            # Select only needed columns and rename
            result = group[['Filename', 'Parsed Timestamp']].rename(columns={
                'Filename': 'filename',
                'Parsed Timestamp': 'words'
            })
            
            # Save to CSV
            output_path = os.path.join(output_dir, output_filename)
            result.to_csv(output_path, index=False)
            print(f"Saved {len(result)} entries to {output_path}")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    input_csv = r'C:\Users\User\Documents\merged_timestamps.csv'  # Replace with your actual CSV path
    process_csv(input_csv)