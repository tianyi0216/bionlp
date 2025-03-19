import os
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Union, Optional, Tuple, Callable
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from preprocess_trial import TrialPreprocessor


class TrialSearchEngine:
    """Search engine for clinical trials.
    
    This class provides functionality to search for clinical trials based on
    text queries, filters, and semantic search.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        DataFrame containing trial data or path to trial data file
    text_fields : List[str], optional
        Fields to search in for text queries
    filter_fields : List[str], optional
        Fields that can be used for filtering
    use_semantic_search : bool, optional
        Whether to use semantic search (requires transformers)
    model_name : str, optional
        Name of the transformer model to use for semantic search
    device : str, optional
        Device to use for transformer model
    """
    
    def __init__(self, 
                data: Union[pd.DataFrame, str],
                text_fields: List[str] = None,
                filter_fields: List[str] = None,
                use_semantic_search: bool = False,
                model_name: str = "all-MiniLM-L6-v2",
                device: str = None):
        """Initialize the search engine."""
        # Set fields to search and filter on
        if text_fields is None:
            self.text_fields = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility_criteria']
        else:
            self.text_fields = text_fields
            
        if filter_fields is None:
            self.filter_fields = ['condition', 'intervention_name', 'phase', 'overall_status']
        else:
            self.filter_fields = filter_fields
        
        # Load data
        if isinstance(data, str):
            self.preprocessor = TrialPreprocessor()
            self.data = self.preprocessor.load_data(data)
        else:
            self.data = data.copy()
        
        # Preprocess the data
        self._preprocess_data()
        
        # Set up semantic search if requested
        self.use_semantic_search = use_semantic_search
        if use_semantic_search:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers package is required for semantic search. Install it with `pip install transformers`.")
            
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            self.model_name = model_name
            self._initialize_semantic_search()
    
    def _preprocess_data(self):
        """Preprocess the trial data for search."""
        # Ensure all text fields are string type
        for field in self.text_fields:
            if field in self.data.columns:
                self.data[field] = self.data[field].fillna('').astype(str)
        
        # Create combined text field for easier searching
        self.data['combined_text'] = self.data[self.text_fields].apply(
            lambda row: ' '.join([str(text) for text in row if text not in [None, 'none', '']]).lower(), 
            axis=1
        )
        
        # Prepare filter fields
        for field in self.filter_fields:
            if field in self.data.columns:
                self.data[field] = self.data[field].fillna('').astype(str)
        
        # Add a field for all available filter values
        self.filter_values = {}
        for field in self.filter_fields:
            if field in self.data.columns:
                # For multi-value fields (comma-separated), split and get unique values
                all_values = []
                for val in self.data[field]:
                    if ',' in val:
                        all_values.extend([v.strip() for v in val.split(',')])
                    else:
                        all_values.append(val.strip())
                
                self.filter_values[field] = sorted(list(set([v for v in all_values if v])))
    
    def _initialize_semantic_search(self):
        """Initialize the transformer model for semantic search."""
        print(f"Initializing semantic search with model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        # Pre-compute embeddings for all trials
        self._compute_trial_embeddings()
    
    def _compute_trial_embeddings(self):
        """Compute embeddings for all trials."""
        print("Computing embeddings for all trials...")
        
        # Function to compute embeddings for a batch of text
        def get_embeddings(texts):
            # Tokenize and get model outputs
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Use mean pooling to get sentence embeddings
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output[0]  # First element contains token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return (sum_embeddings / sum_mask).cpu().numpy()
        
        # Process in batches
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(self.data), batch_size), desc="Computing embeddings"):
            batch_texts = self.data['combined_text'].iloc[i:i+batch_size].tolist()
            batch_embeddings = get_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        self.embeddings = np.vstack(all_embeddings)
    
    def search(self, 
              query: str = None,
              filters: Dict[str, Union[str, List[str]]] = None,
              top_k: int = 10,
              min_similarity: float = 0.3) -> pd.DataFrame:
        """Search for trials matching the query and filters.
        
        Parameters
        ----------
        query : str, optional
            Text query to search for
        filters : Dict[str, Union[str, List[str]]], optional
            Dictionary of field:value(s) filters
        top_k : int, optional
            Number of results to return
        min_similarity : float, optional
            Minimum similarity score for semantic search results (0 to 1)
            
        Returns
        -------
        pd.DataFrame
            Matching trials
        """
        # Start with all trials
        matches = np.ones(len(self.data), dtype=bool)
        
        # Apply filters if provided
        if filters:
            for field, values in filters.items():
                if field in self.data.columns:
                    if isinstance(values, str):
                        values = [values]
                    
                    # Create a filter for this field
                    field_match = np.zeros(len(self.data), dtype=bool)
                    for value in values:
                        # Handle comma-separated values in the data
                        field_match |= self.data[field].str.contains(value, case=False, regex=False, na=False)
                    
                    # Apply the filter
                    matches &= field_match
        
        # Apply text search if query provided
        similarity_scores = None
        if query:
            query = query.lower()
            
            if self.use_semantic_search:
                # Get query embedding
                encoded_query = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt')
                encoded_query = {k: v.to(self.device) for k, v in encoded_query.items()}
                
                with torch.no_grad():
                    model_output = self.model(**encoded_query)
                
                # Use mean pooling to get query embedding
                attention_mask = encoded_query['attention_mask']
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                query_embedding = (sum_embeddings / sum_mask).cpu().numpy()
                
                # Compute similarity between query and all trials
                similarity_scores = cosine_similarity(query_embedding, self.embeddings)[0]
                
                # Filter by similarity score
                semantic_matches = similarity_scores >= min_similarity
                matches &= semantic_matches
            else:
                # Simple text search in combined text
                text_matches = self.data['combined_text'].str.contains(query, case=False, regex=False, na=False)
                matches &= text_matches
        
        # Get matching trials
        results = self.data[matches].copy()
        
        # Sort by similarity if available, otherwise by NCT ID
        if similarity_scores is not None and len(results) > 0:
            # Add similarity scores to results
            similarity_indices = np.where(matches)[0]
            results['similarity_score'] = similarity_scores[similarity_indices]
            
            # Sort by similarity
            results = results.sort_values('similarity_score', ascending=False)
        
        # Return top k results
        return results.head(top_k)
    
    def get_filter_values(self, field: str) -> List[str]:
        """Get all possible values for a filter field.
        
        Parameters
        ----------
        field : str
            The filter field
            
        Returns
        -------
        List[str]
            All possible values for the field
        """
        if field in self.filter_values:
            return self.filter_values[field]
        return []
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Get query suggestions based on partial input.
        
        Parameters
        ----------
        partial_query : str
            Partial query to suggest completions for
        max_suggestions : int, optional
            Maximum number of suggestions to return
            
        Returns
        -------
        List[str]
            Query suggestions
        """
        partial_query = partial_query.lower()
        
        # Get words from titles that match the partial query
        suggestions = []
        
        # Look in brief titles
        if 'brief_title' in self.data.columns:
            # Extract words from titles
            title_words = ' '.join(self.data['brief_title'].fillna('').astype(str)).lower()
            title_words = re.findall(r'\b\w+\b', title_words)
            
            # Find matching words
            matching_words = [word for word in set(title_words) if word.startswith(partial_query)]
            matching_words.sort(key=len)
            
            suggestions.extend(matching_words[:max_suggestions])
        
        # Look in conditions
        if 'condition' in self.data.columns:
            # Extract conditions
            conditions = self.data['condition'].fillna('').astype(str).str.lower()
            conditions = conditions.str.split(',').explode().str.strip().unique()
            
            # Find matching conditions
            matching_conditions = [cond for cond in conditions if partial_query in cond]
            matching_conditions.sort(key=len)
            
            suggestions.extend(matching_conditions[:max_suggestions])
        
        # Deduplicate and limit
        suggestions = list(dict.fromkeys(suggestions))[:max_suggestions]
        
        return suggestions
    
    def search_by_similarity(self, trial_id: str, top_k: int = 10) -> pd.DataFrame:
        """Find trials similar to a specific trial.
        
        Parameters
        ----------
        trial_id : str
            ID of the reference trial
        top_k : int, optional
            Number of similar trials to return
            
        Returns
        -------
        pd.DataFrame
            Similar trials
        """
        if not self.use_semantic_search:
            raise ValueError("Semantic search must be enabled to search by similarity")
        
        # Find the trial
        trial_idx = self.data.index[self.data['nct_id'] == trial_id].tolist()
        if not trial_idx:
            raise ValueError(f"Trial with ID {trial_id} not found")
        
        # Get the trial's embedding
        trial_embedding = self.embeddings[trial_idx[0]].reshape(1, -1)
        
        # Compute similarity to all other trials
        similarity_scores = cosine_similarity(trial_embedding, self.embeddings)[0]
        
        # Create a DataFrame with similarity scores
        results = self.data.copy()
        results['similarity_score'] = similarity_scores
        
        # Exclude the reference trial
        results = results[results['nct_id'] != trial_id]
        
        # Sort by similarity and return top k
        return results.sort_values('similarity_score', ascending=False).head(top_k)


class TrialSearchCollator:
    """Collator for batching trial data for transformer models.
    
    Parameters
    ----------
    model_name : str
        Name of the transformer model
    max_seq_length : int
        Maximum sequence length for tokenization
    text_fields : List[str]
        Fields to tokenize
    device : str
        Device for tensors
    """
    
    def __init__(self,
        model_name: str,
        max_seq_length: int = 512,
        text_fields: List[str] = None,
        device: str = None
        ) -> None:
        """Initialize the collator."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package is required for TrialSearchCollator")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_seq_length
        
        if text_fields is None:
            self.text_fields = ['combined_text']
        else:
            self.text_fields = text_fields
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def __call__(self, features):
        """Collate a batch of examples."""
        return_dict = defaultdict(list)
        batch_df = pd.DataFrame(features)
        batch_df.fillna('', inplace=True)

        # Tokenize text fields
        return_dict.update(self._batch_tokenize(batch_df=batch_df, fields=self.text_fields))
        
        # Move tensors to device
        return self.batch_to_device(return_dict)

    def _batch_tokenize(self, batch_df, fields):
        """Tokenize a batch of text."""
        return_dict = {}
        for field in fields:
            if field in batch_df.columns:
                texts = batch_df[field].tolist()
                tokenized = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
                return_dict[field] = tokenized
        return return_dict
    
    def batch_to_device(self, batch):
        """Move batch to device."""
        for key in batch:
            if isinstance(batch[key], dict):
                # Handle tokenizer outputs
                for subkey, tensor in batch[key].items():
                    if torch.is_tensor(tensor):
                        batch[key][subkey] = tensor.to(self.device)
            elif torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
        return batch


def create_trial_search_demo(
    data: Union[pd.DataFrame, str],
    use_semantic_search: bool = False,
    model_name: str = "all-MiniLM-L6-v2"
) -> Callable:
    """Create a simple search demo function.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        DataFrame containing trial data or path to trial data file
    use_semantic_search : bool, optional
        Whether to use semantic search
    model_name : str, optional
        Name of the transformer model for semantic search
        
    Returns
    -------
    Callable
        Function that performs the search
    """
    # Initialize search engine
    search_engine = TrialSearchEngine(
        data=data,
        use_semantic_search=use_semantic_search,
        model_name=model_name
    )
    
    def search_demo(query=None, filters=None, top_k=10):
        """Search for clinical trials.
        
        Parameters
        ----------
        query : str, optional
            Text query to search for
        filters : Dict[str, Union[str, List[str]]], optional
            Dictionary of field:value(s) filters
        top_k : int, optional
            Number of results to return
            
        Returns
        -------
        pd.DataFrame
            Matching trials
        """
        results = search_engine.search(query=query, filters=filters, top_k=top_k)
        return results
    
    return search_demo


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
    
    print("Creating trial search engine...")
    
    # Initialize search engine (without semantic search for this example)
    search_engine = TrialSearchEngine(
        data=trial_data,
        use_semantic_search=False
    )
    
    # Example 1: Search by text query
    print("\nExample 1: Search for 'diabetes'")
    results = search_engine.search(query='diabetes')
    print(results[['nct_id', 'brief_title', 'condition']])
    
    # Example 2: Search with filters
    print("\nExample 2: Search for Phase 2 trials with 'Completed' status")
    results = search_engine.search(
        filters={'phase': 'Phase 2', 'overall_status': 'Completed'}
    )
    print(results[['nct_id', 'brief_title', 'phase', 'overall_status']])
    
    # Example 3: Combine text search and filters
    print("\nExample 3: Search for 'drug' in Phase 3 trials")
    results = search_engine.search(
        query='drug',
        filters={'phase': 'Phase 3'}
    )
    print(results[['nct_id', 'brief_title', 'intervention_name', 'phase']])
    
    # Example 4: Get filter values
    print("\nExample 4: Get all available conditions")
    conditions = search_engine.get_filter_values('condition')
    print(conditions)
    
    # Example 5: Get query suggestions
    print("\nExample 5: Get query suggestions for 'dep'")
    suggestions = search_engine.get_query_suggestions('dep')
    print(suggestions)
    
    # Example of creating a search demo function
    print("\nCreating a search demo function...")
    search_demo = create_trial_search_demo(data=trial_data)
    
    # Use the demo function
    print("\nUsing the search demo function:")
    demo_results = search_demo(query='asthma')
    print(demo_results[['nct_id', 'brief_title', 'condition']])
