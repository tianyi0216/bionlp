# Deduplication and Preprocessing for Bio Medical Research Data and QA Data

We have the deduplication and preprocessing code for the bio medical research data and also QA data here.

The `Bio_Med_Research/` directory contains the preprocessing code for the bio medical research data. The main function and code
is in the `Bio_Med_Research/utils/` directory, which is more streamlined and `Bio_Med_Research/notebooks` is less important and have what I used to test/run the code.

Similarly, the `QA_Data/` directory contains the preprocessing code for the QA data. The main function and code
is in the `QA_Data/utils/` directory, which is more streamlined and `QA_Data/notebooks` is less important and have what I used to test/run the code.

# Deduplication Algorithm

The deduplication pipeline basically deduplicates the data based on the cosine similarity in the embedding space of a pre-trained model. The algorithm consists of two main components:

## 1. Within-Dataset Deduplication

This process removes duplicates within a single dataset:

1. Text Preparation:
   - For QA data, we do it separately for the question and the answer columns.
   - For bio medical research data, we combine selected columns into a single text string.

2. Embedding Generation:
   - Uses MedImageInsight model (from HuggingFace) to generate text embeddings.
   - Processes texts in batches of 64 by default
   - Returns numpy array of embeddings

3. Similarity Computation:
   - Implements chunked processing (default chunk size: 8000) to handle large datasets
   - Computes cosine similarity between chunks of embeddings
   - Uses a similarity threshold (default: 0.9) to identify duplicates
   - Maintains a set of indices to remove

4. Remove Duplicates and Post-Processing:
   - Removes duplicates based on the indices from the similarity computation (step 3).

## 2. Cross-Dataset Deduplication

After self deduplication, we can use the same algorithm to deduplicate between a new dataset and the existing datasets.

1. Data Preparation:
   - Loads the embeddings from all the saved pickle files.
   - Combines the embeddings into a single numpy array.
   - Also loads the already self-deduplicated data from the csv file.
   - Similarly, for QA data, we do it separately for the question and the answer columns, for bio medical research data, we combine the selected columns into a single text string.

2. Chunked Similarity Computation:
   - Processes data in manageable chunks to optimize memory usage
   - Compares each chunk from the new dataset against all existing dataset embeddings
   - Identifies entries in the new dataset that are similar to any existing entry

3. Remove Duplicates and Post-Processing:
   - Removes duplicates based on the indices from the similarity computation (step 2).
   - Saves the deduplicated data to a new csv file to the set of deduplicated data.
   - Also saves the embeddings to a pickle file for future use.

# Usage

To be updated.

To count the number of data in the dataset, run the `count_data_num.py` file.  Currently it is for QA data only.

```bash
python count_data_num.py dataset_name file_type
```

We have the `Dataset Processing Information - QA.csv` file to record the number of data in each dataset after the deduplication. Before merging the datasets.

To merge the datasets, run `merge_qa_data.py` file in the `QA/utils/` directory. This assumes all the deduplicated datas are csv format.

```bash
python merge_qa_data.py
```

