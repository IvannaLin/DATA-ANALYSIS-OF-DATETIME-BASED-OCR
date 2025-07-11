import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# List of dataset names to process
dataset_names = ["20231019", "20240110", "20241025"]  # Add all your dataset names here

for dataset_name in dataset_names:
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Input directory (original data - remains untouched)
    data_dir = r"C:\Users\User\Documents\College\FYP2\{} - Copy".format(dataset_name)
    input_csv_path = os.path.join(data_dir, r"{} filtered labels.csv".format(dataset_name))

    # Output directories (following your format)
    train_dir = r"C:\Users\User\EasyOCR\trainer\all_data\{}_train".format(dataset_name)
    val_dir = r"C:\Users\User\EasyOCR\trainer\all_data\{}_val\{}_val".format(dataset_name, dataset_name)

    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    try:
        # Load CSV and process
        df = pd.read_csv(input_csv_path)

        # First remove duplicate words (keeping first occurrence)
        unique_df = df.drop_duplicates(subset=['words'], keep='first')

        # Split the unique data into train and validation sets
        train_df, val_df = train_test_split(unique_df, test_size=0.2, random_state=42)

        # Save split CSVs (following your format)
        train_df.to_csv(os.path.join(train_dir, "labels.csv"), index=False)
        val_df.to_csv(os.path.join(val_dir, "labels.csv"), index=False)

        # Copy images to their respective directories (preserving originals)
        for img_name in train_df["filename"]:  # Adjust column name if needed
            src = os.path.join(data_dir, img_name)
            dst = os.path.join(train_dir, img_name)
            if os.path.exists(src):  # Check if file exists before copying
                shutil.copy2(src, dst)

        for img_name in val_df["filename"]:  # Adjust column name if needed
            src = os.path.join(data_dir, img_name)
            dst = os.path.join(val_dir, img_name)
            if os.path.exists(src):  # Check if file exists before copying
                shutil.copy2(src, dst)

        print("Data split complete (only unique words included):")
        print(f"- Training set: {len(train_df)} unique samples in {train_dir}")
        print(f"- Validation set: {len(val_df)} unique samples in {val_dir}")
        print(f"- Original data preserved in {data_dir}")
        print(f"- Removed {len(df) - len(unique_df)} duplicate word entries")

    except FileNotFoundError:
        print(f"Warning: Could not find files for dataset {dataset_name}. Skipping...")
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")

print("\nAll datasets processed!")