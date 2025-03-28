import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from preprocess_trial import TrialPreprocessor

class TrialOutcomeProcessor:
    """Process clinical trial data for outcome prediction tasks.
    
    This class preprocesses trial data to prepare it for outcome prediction models.
    """
    
    def __init__(self, required_fields = None):
        """
        Initialize the trial outcome processor.
        required_fields: Required fields for outcome prediction
        """
        if required_fields is None:
            # the foolowing is from pytrial, however, note that it is also using smile and label, smile is not clinical trial data.
            self.required_fields = [
                'nct_id',                # Direct
                'brief_title',           # Direct
                'overall_status',        # Direct
                'study_first_submitted_date',  # For year
                'completion_date',       # For end_year
                'phase',                 # Direct
                'condition',             # For diseases
                'intervention_name',     # For drugs
                'intervention_type',     # To filter for drugs
                'eligibility_criteria',  # For inclusion/exclusion criteria
                'detailed_description',  # For description
                'why_stopped',           # For why_stop
            ]
        else:
            self.required_fields = required_fields
    
    def load_data(self, file_path):
        """
        Load trial data from file.
        file_path: path to trial data file
        returns: processed trial data
        """
        preprocessor = TrialPreprocessor(required_fields=self.required_fields)
        df = preprocessor.load_data(file_path)
        return self._process_outcome_labels(df)
    
    def _process_outcome_labels(self, df):
        """Process outcome labels based on trial status.
        df: trial data with status information
        returns: data with processed outcome labels
        """
        # map status to outcome labels (simple binary classification here, can be extended to multi-class/more complicated studies)
        # 0: Failed/Terminated, 1: Completed/Successful
        status_map = {
            'completed': 1,
            'completed with results': 1,
            'active': None,  
            'recruiting': None,
            'not yet recruiting': None,
            'enrolling by invitation': None,
            'withdrawn': 0,
            'terminated': 0,
            'suspended': 0,
            'unknown status': None,
            'available': None,
            'no longer available': 0,
            'approved for marketing': 1,
            'withheld': None,
            'temporarily not available': None,
            'none': None
        }
        
        # Standardize status column name
        status_col = 'overall_status'
        if status_col not in df.columns and 'study_status' in df.columns:
            status_col = 'study_status'
        if status_col in df.columns:
            # Convert status to lowercase for consistent mapping
            df[status_col] = df[status_col].str.lower()
            
            # Map status to outcome
            df['outcome'] = df[status_col].map(status_map)
            
            # Drop rows with missing outcomes if needed
            df_with_outcome = df.dropna(subset=['outcome'])
            
            if len(df_with_outcome) < len(df):
                print(f"Note: {len(df) - len(df_with_outcome)} trials dropped due to undefined outcomes.")
            
            return df_with_outcome
        else:
            print("Warning: No status column found, cannot determine outcomes.")
            df['outcome'] = None
            return df
    
    def extract_text_features(self, df, text_columns = None):
        """
        Extract and preprocess text features for outcome prediction.
        df: trial data
        text_columns: list of columns containing text to be processed 
        returns: Data with processed text features
        """
        if text_columns is None:
            text_columns = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility_criteria']
        
        # Combine available text columns
        available_columns = [col for col in text_columns if col in df.columns]
        
        if not available_columns:
            print("Warning: No text columns found for processing.")
            return df
        
        # create combined text field
        df['combined_text'] = df[available_columns].apply(
            lambda row: ' '.join([str(text) for text in row if text not in [None, 'none', '']]), 
            axis=1
        )
        
        # extract simple text features (as a demonstration)
        df['text_length'] = df['combined_text'].str.len()
        df['word_count'] = df['combined_text'].str.split().str.len()
        
        if 'eligibility_criteria' in df.columns:
            df = self._process_eligibility_criteria(df)
        
        return df
    
    def _process_eligibility_criteria(self, df):
        """Process eligibility criteria into inclusion and exclusion criteria."""
        def split_criteria(text):
            text = str(text).lower()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            inclusion = []
            exclusion = []
            current_list = inclusion
            
            for line in lines:
                if 'inclusion criteria' in line:
                    current_list = inclusion
                    continue
                elif 'exclusion criteria' in line:
                    current_list = exclusion
                    continue
                current_list.append(line)
            
            return {
                'inclusion_criteria': ' '.join(inclusion) if inclusion else '',
                'exclusion_criteria': ' '.join(exclusion) if exclusion else ''
            }
        
        if 'eligibility_criteria' in df.columns:
            criteria_split = df['eligibility_criteria'].apply(split_criteria)
            df['inclusion_criteria'] = criteria_split.apply(lambda x: x['inclusion_criteria'])
            df['exclusion_criteria'] = criteria_split.apply(lambda x: x['exclusion_criteria'])
        
        return df
    
    def preprocess_for_model(self, df, categorical_cols = None, numeric_cols = None, text_cols = None, normalize = False, OneHotEncode = False):
        """
        Preprocess data for model training and prediction.
        df: trial data
        categorical_cols: list of categorical columns to encode
        numeric_cols: list of numeric columns to scale
        text_cols: list of text columns to process
        normalize: Whether to normalize numeric columns (not used for text tasks or text models)
        OneHotEncode: Whether to OneHotEncode categorical columns (not used for text tasks or text models)
        returns: Preprocessed data and preprocessing objects
        """
        # Ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # set default columns if none provided
        if categorical_cols is None:
            categorical_cols = ['phase', 'condition']
        
        if numeric_cols is None:
            numeric_cols = ['text_length', 'word_count']
        
        if text_cols is None:
            text_cols = ['brief_title', 'brief_summary']
        
        # process text features if any text columns exist
        available_text_cols = [col for col in text_cols if col in df.columns]
        if available_text_cols:
            df = self.extract_text_features(df, available_text_cols)
        
        # prepare feature transformers
        categorical_processor = {}
        numeric_processor = {}
        
        # process categorical features
        cat_features = []
        for col in categorical_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    if OneHotEncode:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded = encoder.fit_transform(df[[col]])
                        categorical_processor[col] = encoder
                        feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                        cat_features.append(encoded_df)
                    else:
                        cat_features.append(df[[col]])
        
        # combine categorical features
        if cat_features:
            cat_features_df = pd.concat(cat_features, axis=1)
        else:
            cat_features_df = pd.DataFrame(index=df.index)
        
        # process numeric features
        num_features = []
        for col in numeric_cols:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if normalize:
                        scaler = StandardScaler()
                        scaled = scaler.fit_transform(df[[col]])
                        numeric_processor[col] = scaler
                        scaled_df = pd.DataFrame(scaled, columns=[col], index=df.index)
                        num_features.append(scaled_df)
                    else:
                        num_features.append(df[[col]])
        
        # combine numeric features
        if num_features:
            num_features_df = pd.concat(num_features, axis=1)
        else:
            num_features_df = pd.DataFrame(index=df.index)
        
        # combine all features
        features_df = pd.concat([cat_features_df, num_features_df], axis=1)
        
        # Ensure we have the outcome column
        if 'outcome' not in df.columns:
            raise ValueError("DataFrame must contain an 'outcome' column")
            
        # extract labels
        labels = df['outcome'].values
        
        return {
            'features': features_df,
            'labels': labels,
            'categorical_processor': categorical_processor,
            'numeric_processor': numeric_processor,
            'original_data': df
        }


class TrialOutcomeDataset(Dataset):
    """
    Dataset for trial outcome prediction.
    data : DataFrame containing trial data or path to trial data file
    categorical_cols : Categorical columns to encode
    numeric_cols : Numeric columns to scale
    text_cols : Text columns to process
    test_size : Proportion of the dataset to include in the test split
    random_state : Random state for reproducibility
    """
    
    def __init__(self, data, 
                categorical_cols = None,
                numeric_cols = None, 
                text_cols = None,
                test_size = 0.2,
                random_state = 42):
        """Initialize the dataset."""
        self.processor = TrialOutcomeProcessor()
        
        # Load data if a file path is provided
        if isinstance(data, str):
            self.df = self.processor.load_data(data)
        else:
            # If data is a DataFrame, just copy it
            self.df = data.copy()
        
        # Process data for model once
        processed_data = self.processor.preprocess_for_model(
            self.df, categorical_cols, numeric_cols, text_cols
        )
        
        self.features = processed_data['features']
        self.labels = processed_data['labels']
        self.categorical_processor = processed_data['categorical_processor']
        self.numeric_processor = processed_data['numeric_processor']
        self.original_data = processed_data['original_data']
        
        # Create train/test splits
        self.train_indices, self.test_indices = train_test_split(
            np.arange(len(self.labels)),
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
    
    def __len__(self):
        """Return the number of trials."""
        return len(self.features)
    
    def __getitem__(self, idx):
        """Get a trial by index."""
        features = torch.FloatTensor(self.features.iloc[idx].values)
        label = torch.FloatTensor([self.labels[idx]])
        
        return {
            'features': features,
            'label': label,
            'trial_id': self.original_data.iloc[idx]['nct_id'] if 'nct_id' in self.original_data.columns else None
        }
    
    def get_train_dataset(self):
        """Get the training subset."""
        return TrialOutcomeSubset(self, self.train_indices)
    
    def get_test_dataset(self):
        """Get the test subset."""
        return TrialOutcomeSubset(self, self.test_indices)
    
    def get_feature_dim(self):
        """Get the dimension of input features."""
        return self.features.shape[1]


class TrialOutcomeSubset(Dataset):
    """
    Subset of TrialOutcomeDataset for train/test splitting.
    dataset: The original dataset
    indices: Indices to include in the subset
    """
    
    def __init__(self, dataset, indices):
        """Initialize the subset."""
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        """Return the number of trials in the subset."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a trial by index in the subset."""
        return self.dataset[self.indices[idx]]
    
    def get_feature_dim(self):
        """Get the dimension of input features."""
        return self.dataset.get_feature_dim()


class TrialOutcomeCollator:
    """Collate function for batching trial outcome data."""
    
    def __call__(self, batch):
        """Collate a batch of examples."""
        features = torch.stack([item['features'] for item in batch])
        labels = torch.stack([item['label'] for item in batch]).squeeze()
        trial_ids = [item['trial_id'] for item in batch]
        
        return {
            'features': features,
            'labels': labels,
            'trial_ids': trial_ids
        }


def create_outcome_prediction_dataloader(
    data,
    batch_size = 32,
    categorical_cols = None,
    numeric_cols = None,
    text_cols = None,
    test_size = 0.2,
    random_state = 42,
    shuffle = True
):
    """Create DataLoaders for trial outcome prediction.
    
    data: DataFrame containing trial data or path to trial data file
    batch_size: Batch size
    categorical_cols: Categorical columns to encode
    numeric_cols: Numeric columns to scale
    text_cols: Text columns to process
    test_size: Proportion of the dataset to include in the test split
    random_state: Random state for reproducibility
    shuffle: Whether to shuffle the training data
        
    returns: Tuple[DataLoader, DataLoader]
        Training and test DataLoaders
    """
    dataset = TrialOutcomeDataset(
        data=data,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        text_cols=text_cols,
        test_size=test_size,
        random_state=random_state
    )
    
    train_dataset = dataset.get_train_dataset()
    test_dataset = dataset.get_test_dataset()
    
    collator = TrialOutcomeCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    return train_loader, test_loader


# This defines a simple MLP model for trial outcome prediction testing.
class TrialOutcomePredictionModel(torch.nn.Module):
    """Simple MLP model for trial outcome prediction.
    
    Parameters
    ----------
    input_dim : Dimension of input features
    hidden_dims : Dimensions of hidden layers
    dropout_rate : Dropout rate for regularization
    """
    
    def __init__(self, input_dim, hidden_dims = None, dropout_rate = 0.5):
        """Initialize the model."""
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(torch.nn.Linear(hidden_dims[-1], 1))
        layers.append(torch.nn.Sigmoid())
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
def add_llm_text_data_for_trial_outcome(dataset, columns):
    """Add a basic prompt-style string to each trial for LLM input"""
    dataset = dataset.copy()
    dataset['llm_text'] = ""
    for i, row in dataset.iterrows():
        text = "This is a clinical trial.\n"
        for col in columns:
            text += f"{col}: {row.get(col, 'N/A')}\n"
        dataset.at[i, 'llm_text'] = text.strip()
    return dataset
    
def create_llm_dataset_for_trial_outcome_prediction(trial_data, include_labels=True):
    """
    Create a dataset for LLM training by generating natural language prompts 
    from trial data for outcome prediction.

    Parameters:
        trial_data (pd.DataFrame): DataFrame of trial information, must include 'outcome'
        include_labels (bool): Whether to include the outcome as a response column

    Returns:
        pd.DataFrame: Dataset with 'prompt' and (optionally) 'response'
    """
    trial_data = trial_data.copy()

    if 'outcome' not in trial_data.columns:
        raise ValueError("Missing 'outcome' column in trial_data. Please preprocess with outcome labels.")

    prompts = []
    responses = []

    for _, trial_row in trial_data.iterrows():
        prompt = f"""
You are a clinical trial analyst. Based on the following information, determine whether this clinical trial was ultimately successful or failed.

Trial Information:
- ID: {trial_row.get('nct_id', 'N/A')}
- Title: {trial_row.get('brief_title', 'N/A')}
- Summary: {trial_row.get('brief_summary', 'N/A')}
- Phase: {trial_row.get('phase', 'N/A')}
- Condition: {trial_row.get('condition', 'N/A')}
- Eligibility Criteria: {trial_row.get('eligibility_criteria', 'N/A')}

Question: Based on this information, predict whether the trial completed successfully or failed.
Respond with either "Successful" or "Failed".
Answer:
""".strip()
        prompts.append(prompt)

        if include_labels:
            outcome_label = "Successful" if trial_row['outcome'] == 1 else "Failed"
            responses.append(outcome_label)

    output_df = pd.DataFrame({'prompt': prompts})
    if include_labels:
        output_df['response'] = responses

    return output_df

def embed_dataset_column(data, column, model, model_name, batch_size, max_length, device, normalize_embeddings, show_progress, cache_dir):
    """
    Create embeddings for a column in a dataset using a specified model.
    
    data : Input data containing the column to embed. Can be a DataFrame or Series.
    column : Name of the column to embed if data is DataFrame. Not needed if data is Series.
    model : Pre-initialized embedding model. If None, will load model based on model_name.
    model_name : Name of the model to use for embeddings if model not provided.
        Default is "all-MiniLM-L6-v2" from sentence-transformers.
    batch_size : Batch size for processing embeddings.
    max_length : Maximum sequence length for text processing.
    device : Device to run the model on ('cpu', 'cuda', etc.). If None, will use GPU if available.
    normalize_embeddings : Whether to L2-normalize the embeddings.
    show_progress : Whether to show a progress bar during embedding.
    cache_dir : Directory to cache the downloaded model. If None, uses default cache.
        
    returns: Array of embeddings with shape (n_samples, embedding_dim)
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("column name must be provided when input is DataFrame")
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        texts = data[column].astype(str).values
    elif isinstance(data, pd.Series):
        texts = data.astype(str).values
    else:
        raise ValueError("data must be either pandas DataFrame or Series")

    # choose a device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the model if only have name
    if model is None:
        try:
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
        except Exception as e:
            raise ValueError(f"Error loading model '{model_name}': {str(e)}")
    
    model = model.to(device)

    # process in batches
    embeddings = []
    
    # create batches
    n_samples = len(texts)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # create progress bar if requested
    batch_iterator = range(n_batches)
    if show_progress:
        batch_iterator = tqdm(batch_iterator, desc="Creating embeddings")
    
    try:
        for i in batch_iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_texts = texts[start_idx:end_idx]
            
            # get embeddings for batch
            with torch.no_grad():
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                    max_length=max_length
                )
            
            embeddings.append(batch_embeddings)
    
    except Exception as e:
        raise RuntimeError(f"Error during embedding creation: {str(e)}")
    
    # combine all batches
    embeddings = np.vstack(embeddings)  
    return embeddings



# Example usage
if __name__ == "__main__":
    # Sample trial data with more examples
    trial_data = pd.DataFrame({
        'nct_id': [f'NCT00{i}' for i in range(1, 11)],
        'brief_title': [
            'Study of Drug X for Diabetes Treatment',
            'Evaluation of Drug Y for Hypertension',
            'Testing Therapy Z for Asthma Control',
            'Analysis of Treatment W for Arthritis Pain',
            'Research on Drug V for Depression',
            'Study of Drug A for Diabetes Management',
            'Trial of Drug B for Blood Pressure',
            'Investigation of Therapy C for Asthma',
            'Testing of Treatment D for Joint Pain',
            'Evaluation of Drug E for Depression'
        ],
        'brief_summary': [
            'A study investigating the efficacy of Drug X in patients with type 2 diabetes.',
            'This trial evaluates the effect of Drug Y on blood pressure in hypertensive patients.',
            'A clinical trial testing if Therapy Z improves lung function in asthma patients.',
            'Research examining whether Treatment W reduces joint pain in arthritis patients.',
            'A study of Drug V for treating major depressive disorder.',
            'Investigation of Drug A effectiveness in diabetes treatment.',
            'Study of Drug B effects on hypertension management.',
            'Research on Therapy C for asthma symptom control.',
            'Analysis of Treatment D in reducing arthritis symptoms.',
            'Trial of Drug E in depression treatment.'
        ],
        'phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 2', 'Phase 1',
                 'Phase 2', 'Phase 3', 'Phase 2', 'Phase 1', 'Phase 2'],
        'condition': ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Depression',
                     'Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Depression'],
        'overall_status': ['Completed', 'Terminated', 'Completed', 'Withdrawn', 'Completed',
                          'Completed', 'Terminated', 'Completed', 'Withdrawn', 'Completed'],
        'outcome': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    })
    
    print("Creating trial outcome dataset and dataloaders...")
    
    # Create dataloaders
    train_loader, test_loader = create_outcome_prediction_dataloader(
        data=trial_data,
        batch_size=2,
        categorical_cols=['phase', 'condition'],
        numeric_cols=None,
        text_cols=['brief_title', 'brief_summary']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Create and initialize model
    input_dim = train_loader.dataset.get_feature_dim()
    model = TrialOutcomePredictionModel(input_dim=input_dim)
    
    # Get a batch
    batch = next(iter(train_loader))
    print("\nBatch keys:", batch.keys())
    print(f"Features shape: {batch['features'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # Forward pass
    outputs = model(batch['features'])
    print(f"Output predictions shape: {outputs.shape}")
    
    print("\nModel training example:")
    # Example of training loop (without actual training)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    # One training step
    model.train()
    optimizer.zero_grad()
    outputs = model(batch['features'])
    loss = criterion(outputs, batch['labels'].unsqueeze(1))
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    
    # Prediction example
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        predictions = model(test_batch['features'])
        print(f"Predictions: {predictions.squeeze().tolist()}")
        print(f"Ground truth: {test_batch['labels'].tolist()}")
