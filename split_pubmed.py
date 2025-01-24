import pandas as pd

def split_csv(input_file, output_prefix, num_splits=10):
    # Read the CSV in chunks to avoid memory issues
    df = pd.read_csv(input_file)

    # Calculate the number of rows per split
    total_rows = len(df)
    rows_per_split = total_rows // num_splits

    # Split and save each chunk
    for i in range(num_splits):
        start_row = i * rows_per_split
        # For the last split, include all remaining rows
        end_row = (i + 1) * rows_per_split if i != num_splits - 1 else total_rows
        split_df = df.iloc[start_row:end_row]
        split_df.to_csv(f"{output_prefix}_part{i+1}.csv", index=False)

# Example usage
split_csv("large_file.csv", "split_file", num_splits=10)
