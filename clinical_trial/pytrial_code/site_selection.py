import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict
from typing import List, Dict, Union, Optional, Tuple

from preprocess_trial import TrialPreprocessor, TrialDataset

class SiteDataProcessor:
    """Process site data for site selection tasks.
    
    This class loads and preprocesses site-related data for trial site selection.
    """
    
    def __init__(self, required_fields: List[str] = None):
        """Initialize the site data processor.
        
        Parameters
        ----------
        required_fields : List[str], optional
            Required fields for site data
        """
        if required_fields is None:
            self.required_fields = [
                'site_id', 'location_city', 'location_state', 'location_country',
                'specialty', 'capacity', 'experience_years', 'enrollment_rate'
            ]
        else:
            self.required_fields = required_fields
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load site data from file.
        
        Parameters
        ----------
        file_path : str
            Path to site data file
            
        Returns
        -------
        pd.DataFrame
            Processed site data
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return self._validate_and_clean_df(df)
    
    def _validate_and_clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean site data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw site data
            
        Returns
        -------
        pd.DataFrame
            Cleaned site data
        """
        # Standardize column names
        df.columns = [col.lower().replace('.', '_') for col in df.columns]
        
        # Check required fields
        missing_fields = [field for field in self.required_fields if field not in df.columns]
        if missing_fields:
            print(f"Warning: Missing required site fields: {missing_fields}")
            # Add missing fields with empty values
            for field in missing_fields:
                df[field] = "none"
        
        # Fill NaN values
        df.fillna("none", inplace=True)
        
        # Ensure site_id column exists
        if 'site_id' not in df.columns and 'id' in df.columns:
            df['site_id'] = df['id']
        elif 'site_id' not in df.columns:
            df['site_id'] = [f"SITE{i:06d}" for i in range(len(df))]
        
        # Convert string columns to appropriate numeric types when possible
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        return df
    
    def process_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographics data for sites.
        
        Parameters
        ----------
        df : pd.DataFrame
            Site data with demographics information
            
        Returns
        -------
        pd.DataFrame
            Processed site data with demographics
        """
        # Convert demographic columns to appropriate format
        demographic_cols = [col for col in df.columns if any(demo in col for demo in 
                                                           ['gender', 'age', 'race', 'ethnicity'])]
        
        if not demographic_cols:
            print("Warning: No demographic columns found in site data")
            df['has_demographics'] = False
            return df
        
        # Convert to numeric if possible
        for col in demographic_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        df['has_demographics'] = True
        return df


class SiteBase(Dataset):
    """Base dataset for site data.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        DataFrame containing site data or path to data file
    """
    
    def __init__(self, data: Union[pd.DataFrame, str]):
        """Initialize the dataset."""
        self.processor = SiteDataProcessor()
        
        # Load data if a file path is provided
        if isinstance(data, str):
            self.df = self.processor.load_data(data)
        else:
            self.df = data.copy()
        
        # Convert to numeric features
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare features for model input."""
        # Identify numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # One-hot encode categorical variables
        one_hot_df = pd.get_dummies(self.df[categorical_cols], drop_first=False)
        
        # Normalize numeric variables
        numeric_df = self.df[numeric_cols].copy()
        for col in numeric_cols:
            if numeric_df[col].std() > 0:
                numeric_df[col] = (numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std()
        
        # Combine features
        self.features = pd.concat([numeric_df, one_hot_df], axis=1)
        
        # Store feature dimension
        self.feature_dim = self.features.shape[1]
    
    def __len__(self):
        """Return the number of sites."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a site by index."""
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            return self.features.iloc[idx].values
        return self.features.iloc[idx].values


class SiteBaseDemographics(SiteBase):
    """Dataset for site data with demographics information.
    
    Parameters
    ----------
    data : pd.DataFrame or str
        DataFrame containing site data with demographics or path to data file
    """
    
    def __init__(self, data: Union[pd.DataFrame, str]):
        """Initialize the dataset."""
        super().__init__(data)
        
        # Process demographics data
        self.df = self.processor.process_demographics(self.df)
        
        # Extract demographic labels
        self._extract_demographic_labels()
    
    def _extract_demographic_labels(self):
        """Extract demographic labels from site data."""
        demographic_cols = [col for col in self.df.columns if any(demo in col for demo in 
                                                               ['gender', 'age', 'race', 'ethnicity'])]
        
        if not demographic_cols:
            self.demographic_labels = None
            return
        
        self.demographic_labels = self.df[demographic_cols].values
    
    def get_label(self, site_idx):
        """Get demographic labels for a site.
        
        Parameters
        ----------
        site_idx : int or list
            Index of site(s)
            
        Returns
        -------
        np.ndarray
            Demographic labels
        """
        if self.demographic_labels is None:
            return np.zeros((1, 1))
        
        if isinstance(site_idx, list) or isinstance(site_idx, np.ndarray):
            return self.demographic_labels[site_idx]
        return self.demographic_labels[site_idx]


class TrialSiteDataset(Dataset):
    """Dataset for trial-site pairs for site selection.
    
    Parameters
    ----------
    trial_data : pd.DataFrame or str
        DataFrame containing trial data or path to trial data file
    site_data : pd.DataFrame or str
        DataFrame containing site data or path to site data file
    mapping_data : pd.DataFrame or str, optional
        DataFrame containing trial-site mappings or path to mapping file
    has_demographics : bool, optional
        Whether site data includes demographics
    """
    
    def __init__(self, 
                trial_data: Union[pd.DataFrame, str],
                site_data: Union[pd.DataFrame, str],
                mapping_data: Union[pd.DataFrame, str] = None,
                has_demographics: bool = False):
        """Initialize the dataset."""
        # Load trial data
        if isinstance(trial_data, str):
            self.trial_processor = TrialPreprocessor()
            self.trial_df = self.trial_processor.load_data(trial_data)
        else:
            self.trial_df = trial_data.copy()
        
        # Load site data
        if has_demographics:
            self.sites = SiteBaseDemographics(site_data)
        else:
            self.sites = SiteBase(site_data)
        
        # Load or create trial-site mappings
        if mapping_data is not None:
            if isinstance(mapping_data, str):
                self.mapping_df = pd.read_csv(mapping_data)
            else:
                self.mapping_df = mapping_data.copy()
            
            # Extract mappings and enrollment values
            self._process_mappings()
        else:
            # Create random mappings for demonstration
            self._create_example_mappings()
        
        # Process trial data for feature extraction
        self._process_trial_features()
    
    def _process_mappings(self):
        """Process trial-site mappings."""
        if 'trial_id' not in self.mapping_df.columns or 'site_id' not in self.mapping_df.columns:
            raise ValueError("Mapping data must contain 'trial_id' and 'site_id' columns")
        
        # Convert to lists of trial-site pairs
        self.mappings = []
        self.enrollments = []
        
        for trial_id in self.trial_df['nct_id'].unique():
            # Get sites for this trial
            trial_sites = self.mapping_df[self.mapping_df['trial_id'] == trial_id]
            
            # Extract site indices
            site_indices = []
            enrollment_values = []
            
            for _, row in trial_sites.iterrows():
                site_id = row['site_id']
                site_idx = self.sites.df[self.sites.df['site_id'] == site_id].index.tolist()
                
                if site_idx:
                    site_indices.append(site_idx[0])
                    
                    # Get enrollment value if available
                    if 'enrollment' in row:
                        enrollment_values.append(row['enrollment'])
                    else:
                        enrollment_values.append(1.0)  # Default enrollment
            
            if site_indices:
                self.mappings.append(site_indices)
                self.enrollments.append(enrollment_values)
    
    def _create_example_mappings(self):
        """Create example mappings for demonstration."""
        self.mappings = []
        self.enrollments = []
        
        num_sites = len(self.sites)
        
        for _ in range(len(self.trial_df)):
            # Handle case where num_sites is small
            if num_sites < 5:
                # If we have fewer than 5 sites, use all available sites
                num_selected = num_sites
                site_indices = list(range(num_sites))
            else:
                # Randomly select 5-15 sites for each trial, limited by available sites
                max_sites = min(15, num_sites)
                num_selected = np.random.randint(5, max_sites + 1)  # +1 because upper bound is exclusive
                site_indices = np.random.choice(num_sites, num_selected, replace=False).tolist()
            
            # Generate random enrollment values
            enrollment_values = np.random.randint(1, 20, size=num_selected).tolist()
            
            self.mappings.append(site_indices)
            self.enrollments.append(enrollment_values)
    
    def _process_trial_features(self):
        """Process trial features for model input."""
        # Get trial features
        trial_features = {}
        
        # Extract numeric and categorical columns
        numeric_cols = self.trial_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.trial_df.select_dtypes(include=['object']).columns.tolist()
        
        # Normalize numeric features
        for col in numeric_cols:
            if self.trial_df[col].std() > 0:
                trial_features[col] = (self.trial_df[col] - self.trial_df[col].mean()) / self.trial_df[col].std()
            else:
                trial_features[col] = self.trial_df[col]
        
        # One-hot encode categorical features
        for col in categorical_cols:
            dummies = pd.get_dummies(self.trial_df[col], prefix=col, drop_first=False)
            for dummy_col in dummies.columns:
                trial_features[dummy_col] = dummies[dummy_col]
        
        # Convert to DataFrame
        self.trial_features = pd.DataFrame(trial_features)
    
    def __len__(self):
        """Return the number of trial-site pairs."""
        return len(self.mappings)
    
    def __getitem__(self, idx):
        """Get a trial-site pair by index."""
        trial_features = self.trial_features.iloc[idx].values
        site_indices = self.mappings[idx]
        site_features = self.sites[site_indices]
        enrollment_values = self.enrollments[idx]
        
        # Get demographic labels if available
        if isinstance(self.sites, SiteBaseDemographics):
            demographic_labels = self.sites.get_label(site_indices)
        else:
            demographic_labels = None
        
        return {
            'trial': trial_features,
            'site': site_features,
            'label': enrollment_values,
            'eth_label': demographic_labels
        }


class TrialSiteCollator:
    """Collator for batching trial-site data.
    
    Parameters
    ----------
    has_demographics : bool
        Whether site data includes demographics
    """
    
    def __init__(self, has_demographics: bool = False):
        """Initialize the collator."""
        self.has_demographics = has_demographics
    
    def __call__(self, batch):
        """Collate a batch of examples."""
        return_data = {}
        
        # Collect trial features
        return_data['trial'] = torch.FloatTensor(np.array([item['trial'] for item in batch]))
        
        # Collect site features with padding
        site_lengths = [len(item['site']) for item in batch]
        max_sites = max(site_lengths)
        
        # Create padded tensor for site features
        site_dim = batch[0]['site'].shape[1] if len(batch[0]['site'].shape) > 1 else 1
        padded_sites = np.zeros((len(batch), max_sites, site_dim))
        
        for i, item in enumerate(batch):
            padded_sites[i, :len(item['site'])] = item['site']
        
        return_data['site'] = torch.FloatTensor(padded_sites)
        
        # Create padded tensor for labels
        padded_labels = np.zeros((len(batch), max_sites))
        for i, item in enumerate(batch):
            padded_labels[i, :len(item['label'])] = item['label']
        
        return_data['label'] = torch.FloatTensor(padded_labels)
        
        # Create site mask to handle variable number of sites
        site_mask = np.zeros((len(batch), max_sites))
        for i, length in enumerate(site_lengths):
            site_mask[i, :length] = 1
        
        return_data['site_mask'] = torch.BoolTensor(site_mask)
        
        # Handle demographic labels if available
        if self.has_demographics:
            demo_dim = batch[0]['eth_label'].shape[1] if batch[0]['eth_label'] is not None else 1
            padded_demo = np.zeros((len(batch), max_sites, demo_dim))
            
            for i, item in enumerate(batch):
                if item['eth_label'] is not None:
                    padded_demo[i, :len(item['eth_label'])] = item['eth_label']
            
            return_data['eth_label'] = torch.FloatTensor(padded_demo)
        
        return return_data


def create_site_selection_dataloader(
    trial_data: Union[pd.DataFrame, str],
    site_data: Union[pd.DataFrame, str],
    mapping_data: Union[pd.DataFrame, str] = None,
    batch_size: int = 16,
    has_demographics: bool = False,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for site selection.
    
    Parameters
    ----------
    trial_data : pd.DataFrame or str
        DataFrame containing trial data or path to trial data file
    site_data : pd.DataFrame or str
        DataFrame containing site data or path to site data file
    mapping_data : pd.DataFrame or str, optional
        DataFrame containing trial-site mappings or path to mapping file
    batch_size : int, optional
        Batch size
    has_demographics : bool, optional
        Whether site data includes demographics
    shuffle : bool, optional
        Whether to shuffle the data
        
    Returns
    -------
    DataLoader
        DataLoader for site selection
    """
    dataset = TrialSiteDataset(
        trial_data=trial_data,
        site_data=site_data,
        mapping_data=mapping_data,
        has_demographics=has_demographics
    )
    
    collator = TrialSiteCollator(has_demographics=has_demographics)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Create from sample data
    print("Creating from sample trial and site data...")
    
    # Sample trial data
    trial_data = pd.DataFrame({
        'nct_id': ['NCT001', 'NCT002', 'NCT003'],
        'brief_title': ['Study 1', 'Study 2', 'Study 3'],
        'phase': ['Phase 1', 'Phase 2', 'Phase 3'],
        'condition': ['Diabetes', 'Hypertension', 'Asthma'],
        'eligibility_criteria': [
            'Inclusion: Age > 18\nExclusion: Pregnant',
            'Inclusion: Healthy\nExclusion: Cancer',
            'Inclusion: Adult\nExclusion: Children'
        ]
    })
    
    # Sample site data
    site_data = pd.DataFrame({
        'site_id': ['SITE001', 'SITE002', 'SITE003', 'SITE004', 'SITE005'],
        'location_city': ['New York', 'Chicago', 'Los Angeles', 'Houston', 'Miami'],
        'location_state': ['NY', 'IL', 'CA', 'TX', 'FL'],
        'location_country': ['US', 'US', 'US', 'US', 'US'],
        'specialty': ['Cardiology', 'Endocrinology', 'Pulmonology', 'Oncology', 'Neurology'],
        'capacity': [100, 80, 120, 90, 70],
        'experience_years': [10, 5, 15, 8, 12],
        'enrollment_rate': [0.8, 0.6, 0.9, 0.7, 0.75]
    })
    
    # Sample site data with demographics
    site_data_demo = site_data.copy()
    site_data_demo['gender_male'] = [0.6, 0.5, 0.55, 0.45, 0.5]
    site_data_demo['gender_female'] = [0.4, 0.5, 0.45, 0.55, 0.5]
    site_data_demo['age_18_30'] = [0.2, 0.25, 0.15, 0.3, 0.2]
    site_data_demo['age_31_50'] = [0.4, 0.35, 0.45, 0.3, 0.4]
    site_data_demo['age_51_plus'] = [0.4, 0.4, 0.4, 0.4, 0.4]
    
    # Create dataset without demographics
    dataset = TrialSiteDataset(
        trial_data=trial_data,
        site_data=site_data
    )
    print(f"Dataset has {len(dataset)} trial-site pairs")
    
    # Access a trial-site pair
    pair = dataset[0]
    print(f"Trial feature dimension: {pair['trial'].shape}")
    print(f"Site feature dimension: {pair['site'].shape}")
    print(f"Number of sites for this trial: {len(pair['label'])}")
    
    # Create dataset with demographics
    dataset_demo = TrialSiteDataset(
        trial_data=trial_data,
        site_data=site_data_demo,
        has_demographics=True
    )
    
    # Create dataloader
    dataloader = create_site_selection_dataloader(
        trial_data=trial_data,
        site_data=site_data,
        batch_size=2
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    print("\nBatch keys:", batch.keys())
    print(f"Trial batch shape: {batch['trial'].shape}")
    print(f"Site batch shape: {batch['site'].shape}")
    print(f"Label batch shape: {batch['label'].shape}")
    print(f"Site mask shape: {batch['site_mask'].shape}")
    
    # Create dataloader with demographics
    dataloader_demo = create_site_selection_dataloader(
        trial_data=trial_data,
        site_data=site_data_demo,
        batch_size=2,
        has_demographics=True
    )
    
    # Get a batch with demographics
    batch_demo = next(iter(dataloader_demo))
    print("\nBatch with demographics keys:", batch_demo.keys())
    print(f"Demographic label shape: {batch_demo['eth_label'].shape}")
