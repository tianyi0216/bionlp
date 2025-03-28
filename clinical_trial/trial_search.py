import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from preprocess_trial import TrialPreprocessor

class TrialSearchProcessor:
    """Processor for preparing clinical trial data for search tasks.
    
    This class handles the preprocessing of trial data to make it suitable for search operations.
    It focuses on cleaning text fields, creating combined representations, and preparing categorical fields for filtering.

    text_fields : Fields to use for text search
    filter_fields : Fields that can be used for filtering
    """
    
    def __init__(self, text_fields = None, filter_fields = None):
        """Initialize the processor."""
        # fields to search and filter on
        if text_fields is None:
            self.text_fields = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility_criteria']
        else:
            self.text_fields = text_fields
            
        if filter_fields is None:
            self.filter_fields = ['condition', 'intervention_name', 'phase', 'overall_status']
        else:
            self.filter_fields = filter_fields
    
    def process_data(self, data):
        """Process trial data for search.
        data :  DataFrame containing trial data or path to trial data file
        returns: Processed data ready for search
        """
        # load data if a file path is provided, else assume it's a dataframe
        if isinstance(data, str):
            preprocessor = TrialPreprocessor()
            df = preprocessor.load_data(data)
        else:
            df = data.copy()
        
        # process text fields
        df = self._process_text_fields(df)
        # process filter fields
        df = self._process_filter_fields(df)
        # extract keywords and n-grams
        df = self._extract_keywords(df)
        return df
    
    def _process_text_fields(self, df):
        """
        Process text fields for search.
        df : Input DataFrame
        returns: DataFrame with processed text fields
        """
        # make sure all text fields are strings
        for field in self.text_fields:
            if field in df.columns:
                df[field] = df[field].fillna('').astype(str)
        
        # create combined text field for easier searching
        df['combined_text'] = df[self.text_fields].apply(
            lambda row: ' '.join([str(text) for text in row if text not in [None, 'none', '']]), 
            axis=1
        )
        df['combined_text_lower'] = df['combined_text'].str.lower()
        return df
    
    def _process_filter_fields(self, df):
        """Process categorical fields for filtering.
        
        df : Input DataFrame
        returns: DataFrame with processed filter fields
        """
        # Prepare filter fields
        for field in self.filter_fields:
            if field in df.columns:
                df[field] = df[field].fillna('').astype(str)
                df[f"{field}_lower"] = df[field].str.lower()
        
        # extract unique filter values for each field
        filter_values = {}
        for field in self.filter_fields:
            if field in df.columns:
                # for multi-value fields (comma-separated), split and get unique values
                all_values = []
                for val in df[field]:
                    if ',' in val:
                        all_values.extend([v.strip() for v in val.split(',')])
                    else:
                        all_values.append(val.strip())
                
                filter_values[field] = sorted(list(set([v for v in all_values if v])))
        
        # store filter values as metadata
        df.attrs['filter_values'] = filter_values
        return df
    
    def _extract_keywords(self, df):
        """Extract keywords and n-grams from text fields.
        
        df : Input DataFrame
        returns: DataFrame with extracted keywords
        """
        # get important words from titles and conditions
        if 'brief_title' in df.columns:
            # extract words from titles using regex pattern for meaningful words (3+ chars)
            df['title_keywords'] = df['brief_title'].apply(
                lambda x: ' '.join(re.findall(r'\b[a-zA-Z]{3,}\b', x.lower()))
            )
        
        if 'condition' in df.columns:
            # get condition keywords using regex pattern for meaningful words (3+ chars)
            df['condition_keywords'] = df['condition'].apply(
                lambda x: ' '.join(re.findall(r'\b[a-zA-Z]{3,}\b', x.lower()))
            )
            
        # get bigrams from combined text (could be useful for phrase matching)
        if 'combined_text_lower' in df.columns:
            def extract_bigrams(text):
                # for bigrams, we do this by joining two words
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
                return [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            
            df['text_bigrams'] = df['combined_text_lower'].apply(
                lambda x: ' '.join(extract_bigrams(x))
            )
        return df

class TrialSearchDataset(Dataset):
    """Dataset for trial search tasks.
    
    This dataset prepares clinical trial data for search and retrieval tasks.
    
    data : dataFrame containing trial data or path to trial data file
    text_fields : fields to use for text search
    filter_fields : fields that can be used for filtering
    """
    
    def __init__(self, data, text_fields = None, filter_fields = None):
        """Initialize the dataset."""
        # process data
        self.processor = TrialSearchProcessor(
            text_fields=text_fields,
            filter_fields=filter_fields
        )
        self.data = self.processor.process_data(data)
        
        # extract field information
        self.text_fields = self.processor.text_fields
        self.filter_fields = self.processor.filter_fields
        
        # get filter values
        self.filter_values = self.data.attrs.get('filter_values', {})
    
    def __len__(self):
        """Return the number of trials."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a trial by index."""
        row = self.data.iloc[idx]
        item = row.to_dict()
        
        # Ensure all string fields are properly formatted for tokenization
        for field in item:
            if isinstance(item[field], str):
                item[field] = item[field].strip()
        
        return item
    
    def get_filter_values(self, field):
        """
        Get all possible values for a filter field.
        field : str filter field name
        returns: List of unique values for the field
        """
        return self.filter_values.get(field, [])
    
    def filter_by_criteria(self, criteria):
        """Filter the dataset by specified criteria.
        
        criteria : the dictionary of fields and values to filter by
        returns: Filtered data
        """
        filtered_data = self.data.copy()
        
        for field, values in criteria.items():
            if field in filtered_data.columns:
                # convert single value to list
                if isinstance(values, str):
                    values = [values]
                
                # use lowercase version if available
                search_field = f"{field}_lower" if f"{field}_lower" in filtered_data.columns else field
                
                # handle each value as a separate filter (OR condition)
                matches = np.zeros(len(filtered_data), dtype=bool)
                for value in values:
                    value_lower = value.lower()
                    matches |= filtered_data[search_field].str.contains(value_lower, regex=False, na=False)
                
                # apply filter
                filtered_data = filtered_data[matches]
        
        return filtered_data
    
    def search_by_text(self, query, fields = None):
        """
        Simple text-based search across specified fields.
        To be adding more sophisticated search methods in the future.
        query : Text query to search for
        fields : fields to search in (defaults to combined_text_lower)
        returns: Matching data
        """
        if not query:
            return self.data
        
        if fields is None:
            fields = ['combined_text_lower']
        
        query_lower = query.lower()
        matches = np.zeros(len(self.data), dtype=bool)
        
        for field in fields:
            if field in self.data.columns:
                matches |= self.data[field].str.contains(query_lower, regex=False, na=False)
        
        return self.data[matches]


class TrialSearchCollator:
    """Collator for batching trial data for search tasks.
    
    This collator prepares batches of trial data, including tokenization
    of text fields for models that use tokenized inputs.
    
    model_name : Transformer model name for tokenization
    max_seq_length : Maximum sequence length for tokenization
    fields : fields to tokenize
    """
    
    def __init__(self,
        model_name = "bert-base-uncased",
        max_seq_length = 512,
        fields = None):
        """Initialize the collator."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package is required for tokenization. Install with `pip install transformers`.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_seq_length
        
        if fields is None:
            self.fields = ['combined_text']
        else:
            self.fields = fields

    def __call__(self, features):
        """Collate a batch of examples."""
        return_dict = defaultdict(list)
        batch_df = pd.DataFrame(features)
        batch_df.fillna('', inplace=True)

        # Add all fields to return dict
        for field in batch_df.columns:
            if field not in self.fields:  # Skip fields that will be tokenized
                return_dict[field] = batch_df[field].tolist()

        # Add tokenized fields
        return_dict.update(self._batch_tokenize(batch_df=batch_df, fields=self.fields))
        
        return return_dict

    def _batch_tokenize(self, batch_df, fields):
        """
        Tokenize a batch of text.
        batch_df : pd.DataFrame
        fields : List[str]
        returns: Dictionary of tokenized fields
        """
        return_dict = {}
        for field in fields:
            if field in batch_df.columns:
                texts = batch_df[field].tolist()
                tokenized = self.tokenizer(
                    texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors='pt'
                )
                return_dict[field] = tokenized
        return return_dict


def create_trial_search_dataloader(
    data,
    batch_size = 32,
    text_fields = None,
    filter_fields = None,
    model_name = "bert-base-uncased",
    max_seq_length = 512,
    shuffle = False
):
    """Create a DataLoader for trial search tasks.
    
    data : DataFrame containing trial data or path to trial data file
    batch_size : batch size
    text_fields : fields to use for text search
    filter_fields : fields that can be used for filtering
    model_name : Transformer model name for tokenization
    max_seq_length : Maximum sequence length for tokenization
    shuffle : Whether to shuffle the data
    returns: DataLoader and the underlying Dataset
    """
    # Create dataset
    dataset = TrialSearchDataset(
        data=data,
        text_fields=text_fields,
        filter_fields=filter_fields
    )
    
    # Create collator
    collator = TrialSearchCollator(
        model_name=model_name,
        max_seq_length=max_seq_length,
        fields=['combined_text']  # Default to combined text
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )
    
    return dataloader, dataset

def create_llm_dataset_for_trial_search(trial_data, user_queries=None, include_labels=False):
    """
    Create an LLM dataset for trial search.

    Each row corresponds to a (query, trial) pair. The model is expected to decide 
    whether the trial matches the user query.

    Parameters:
        trial_data (pd.DataFrame): Clinical trial dataset (should include combined_text)
        user_queries (List[str]): Optional list of search queries to generate examples
        include_labels (bool): Whether to include binary labels (1 = match, 0 = not match)

    Returns:
        pd.DataFrame: LLM training data with 'prompt' and optionally 'response'
    """
    if user_queries is None:
        # Default: infer queries from condition or brief_title
        user_queries = trial_data['condition'].dropna().unique().tolist()

    prompts = []
    responses = []

    for query in user_queries:
        for _, row in trial_data.iterrows():
            prompt = f"""
You are a clinical research assistant helping patients find relevant clinical trials.

User query: "{query}"

Here is a clinical trial:
- Title: {row.get('brief_title', 'N/A')}
- Summary: {row.get('brief_summary', 'N/A')}
- Condition: {row.get('condition', 'N/A')}
- Intervention: {row.get('intervention_name', 'N/A')}
- Phase: {row.get('phase', 'N/A')}
- Status: {row.get('overall_status', 'N/A')}

Question: Is this clinical trial relevant to the user's query?
Respond with "Yes" or "No" and briefly justify your answer.
Answer:
""".strip()
            prompts.append(prompt)

            if include_labels:
                # Simple heuristic: label as relevant if condition matches query (case-insensitive)
                label = "Yes" if query.lower() in str(row.get('condition', '')).lower() else "No"
                responses.append(label)

    output_df = pd.DataFrame({'prompt': prompts})
    if include_labels:
        output_df['response'] = responses

    return output_df

def add_llm_text_data_for_trial_search(dataset, fields):
    """Add a textual description of each trial for LLM input"""
    dataset = dataset.copy()
    dataset['llm_text'] = ""
    for i, row in dataset.iterrows():
        llm_text = "Here is a clinical trial:\n"
        for field in fields:
            llm_text += f"{field}: {row.get(field, 'N/A')}\n"
        dataset.at[i, 'llm_text'] = llm_text.strip()
    return dataset



# Example usage
if __name__ == "__main__":
    # Sample trial data
    trial_data = pd.DataFrame({
        'nct_id': ['NCT001', 'NCT002', 'NCT003', 'NCT004', 'NCT005'],
        'brief_title': [
            'Study of Drug X for Diabetes Treatment', 
            'Evaluation of Drug Y for Hypertension', 
            'Testing Therapy Z for Asthma Control',
            'Analysis of Treatment W for Arthritis Pain', 
            'Research on Drug V for Depression'
        ],
        'brief_summary': [
            'A study investigating the efficacy of Drug X in patients with type 2 diabetes.',
            'This trial evaluates the effect of Drug Y on blood pressure in hypertensive patients.',
            'A clinical trial testing if Therapy Z improves lung function in asthma patients.',
            'Research examining whether Treatment W reduces joint pain in arthritis patients.',
            'A study of Drug V for treating major depressive disorder.'
        ],
        'condition': ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Depression'],
        'intervention_name': ['Drug X', 'Drug Y', 'Therapy Z', 'Treatment W', 'Drug V'],
        'phase': ['Phase 2', 'Phase 3', 'Phase 2', 'Phase 3', 'Phase 2'],
        'overall_status': ['Completed', 'Recruiting', 'Completed', 'Terminated', 'Completed']
    })

    text_fields = ['brief_title', 'brief_summary', 'condition', 'intervention_name', 'phase', 'overall_status']
    
    print("Creating trial search dataset...")
    
    # Process data for search
    dataset = TrialSearchDataset(trial_data, text_fields=text_fields)
    
    # test 1, search by text
    print("\nTest 1: Search for 'diabetes'")
    results = dataset.search_by_text('diabetes')
    print(results[['nct_id', 'brief_title']])
    
    # test 2, filter by criteria
    print("\nTest 2: Filter for Phase 2 trials with 'Completed' status")
    results = dataset.filter_by_criteria({
        'phase': 'Phase 2', 
        'overall_status': 'Completed'
    })
    print(results[['nct_id', 'brief_title', 'phase', 'overall_status']])
    
    # test 3, get filter values
    print("\nTest 3: Get all available conditions")
    conditions = dataset.get_filter_values('condition')
    print(conditions)
    
    # test 4, create dataloader
    print("\nTest 4: Creating dataloader")
    dataloader, _ = create_trial_search_dataloader(
        data=trial_data,
        batch_size=2,
        text_fields=text_fields
    )
    
    # test 5, get a batch from dataloader
    print("\nTest 5: Get a batch from dataloader")
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"NCT IDs in batch: {batch['nct_id']}")
    print(f"Tokenized combined_text shape: {batch['combined_text']['input_ids'].shape}")
