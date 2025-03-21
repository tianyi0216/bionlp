# This file is used to download the clinical trial data from ClinicalTrials.gov. Most of the available clinical trial data is gotten from the ClinicalTrials.gov
# The data is downloaded in the form of a zip file, which is then extracted into the aact-raw directory.
# The data is then given the option to be processed into a csv, json, or xml file.

import os
import requests
import zipfile
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Union, Optional
import csv
from io import StringIO

class ClinicalTrialDownloader:
    """Utilities for downloading clinical trial data from ClinicalTrials.gov."""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/"
    JSON_FORMAT = "format=json"
    CSV_FORMAT = "format=csv"
    XML_FORMAT = "format=xml"
    RAW_TXT_DIR = './aact-raw'
    
    # some default fields, can be modified to anything included in the API (to be verified)
    DEFAULT_FIELDS = {
        'json': [
            'NCTId',                  
            'BriefTitle',             
            'BriefSummary',           
            'DetailedDescription',    
            'EligibilityCriteria',    
            'Condition',              
            'OverallStatus'           
        ],
        'csv': [
            'nct_id',                 
            'brief_title',            
            'brief_summary',          
            'detailed_description',   
            'eligibility_criteria',   
            'condition',              
            'overall_status'          
        ]
    }

    def __init__(self, api_key: str = None):
        """Initialize downloader.
        api_key : optional in case needed
        """
        self.api_key = api_key
        self.api_info = self._get_api_info()

    def _convert_field_format(self, fields, from_fmt, to_fmt):
        """Convert field names between formats (json/csv).
        fields: list of field names to convert
        from_fmt: source format ('json' or 'csv')
        to_fmt: target format ('json' or 'csv')
        returns: converted field names
        """
        if from_fmt == to_fmt:
            return fields
            
        result = []
        
        # dictionaries to map the field names between formats
        json_to_csv = {}
        for j, c in zip(self.DEFAULT_FIELDS['json'], self.DEFAULT_FIELDS['csv']):
            json_to_csv[j] = c
            
        csv_to_json = {}
        for c, j in zip(self.DEFAULT_FIELDS['csv'], self.DEFAULT_FIELDS['json']):
            csv_to_json[c] = j
            
        if from_fmt == 'json' and to_fmt == 'csv':
            for field in fields:
                if field in json_to_csv:
                    result.append(json_to_csv[field])
                else:
                    # convert to a common format
                    result.append(field.lower().replace('.', '_'))
        elif from_fmt == 'csv' and to_fmt == 'json':
            for field in fields:
                if field in csv_to_json:
                    result.append(csv_to_json[field])
                else:
                    # convert to a common format
                    parts = field.split('_')
                    result.append(''.join(part.capitalize() if i > 0 else part for i, part in enumerate(parts)))
                    
        return result

    def _get_headers(self):
        """Get headers for API requests."""
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def _get_api_info(self):
        try:
            #get version info
            version_url = f"{self.BASE_URL}version"
            response = requests.get(version_url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            api_version = data.get("apiVersion", "unknown")
            last_updated = data.get("dataTimestamp", "unknown")
            
            return api_version, last_updated
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ClinicalTrials.gov API: {e}")
            return "unknown", "unknown"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "unknown", "unknown"

    def _request_json(self, url):
        """Make a JSON request to the API."""
        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ClinicalTrials.gov API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    print(f"API Error response: {error_data}")
                except:
                    print(f"Response text: {e.response.text}")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}
    
    def _request_csv(self, url):
        """Make a CSV request to the API. Returns the list of csv rows(readers)"""
        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            decoded_content = response.content.decode("utf-8")
            if not decoded_content.strip():
                print("Warning: Empty response received")
                return []
                
            # potential malformed CSV
            try:
                csv_reader = csv.reader(StringIO(decoded_content))
                return list(csv_reader)
            except csv.Error as e:
                print(f"CSV parsing error: {e}")
                print(f"Response preview: {decoded_content[:200]}...")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ClinicalTrials.gov API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response text: {e.response.text[:200]}...")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    @property
    def study_fields(self):
        """Get all possible fields available from ClinicalTrials.gov.
        Pytrial's API call to get fields is not working, for now, it seems it has to be manually set or defined.
        """
        return self.DEFAULT_FIELDS

    def download_data(self, date = '20220501', output_dir = './clinical_trials_data'):
        """Download the full dataset from ClinicalTrials.gov as a zip file. 
        This takes a while as it is downloading the full dataset, shouldn't be used too often.
        date: the date of the database copy in YYYYMMDD format
        output_dir: the output directory for the downloaded data
        """
        url = f'https://aact.ctti-clinicaltrials.org/static/exported_files/daily/{date}_pipe-delimited-export.zip'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f'Downloading data from {url}...')
        filename = os.path.join(output_dir, 'clinical_trials.zip')
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output_dir, self.RAW_TXT_DIR))
        print(f'Data downloaded and extracted to {output_dir}')

    def query_studies(self,
        search_expr,
        fields = None,
        max_studies = 50,
        fmt = 'json'):
        """Query study content for specified fields from the ClinicalTrials.gov API.
        search_expr: search expression as specified in ClinicalTrials.gov API documentation
        fields: list of fields to retrieve. If None, uses default fields.
        max_studies: maximum number of studies to return (1-1000)
        fmt: output format: 'json' or 'csv'
        returns: query results in specified format
        """
        if fields is None:
            # Use a subset of fields by default
            if fmt == 'json':
                fields = ['NCTId', 'BriefTitle', 'Condition', 'InterventionName', 'OverallStatus']
            else:  # csv
                fields = ['nct_id', 'brief_title', 'condition', 'intervention_name', 'overall_status']
        
        if max_studies > 1000 or max_studies < 1:
            raise ValueError("The number of studies can only be between 1 and 1000")
        field_format = 'json' if fmt == 'json' else 'csv'
        
        #convert fields types if needed
        if len(fields) > 0:
            first_field = fields[0]
            if first_field.islower() and field_format == 'json':
                print(f"Converting field names from CSV to JSON format.")
                fields = self._convert_field_format(fields, 'csv', 'json')
            elif not first_field.islower() and field_format == 'csv':
                print(f"Converting field names from JSON to CSV format.")
                fields = self._convert_field_format(fields, 'json', 'csv')
        
        # check if fields are valid
        valid_fields = [field for field in fields if field in self.study_fields.get(field_format, [])]
        if len(valid_fields) < len(fields):
            print(f"Warning: Some fields were not valid for format '{fmt}'. Using valid fields only: {valid_fields}")
            
        # if no valid fields, use default ones
        if not valid_fields:
            valid_fields = self.study_fields.get(field_format, [])[:5]
            print(f"Using default fields: {valid_fields}")
            
        # use appropriate format parameter and fields
        format_param = self.JSON_FORMAT if fmt == 'json' else self.CSV_FORMAT
        fields = valid_fields
        
        # join fields with pipe separator as required by the API
        concat_fields = "|".join(fields)
        
        # build the URL
        url = f"{self.BASE_URL}studies?{format_param}&query.term={search_expr}&markupFormat=legacy&fields={concat_fields}&pageSize={max_studies}"
        print(f"API URL: {url}")
        
        # equest based on format
        if fmt == 'json':
            result = self._request_json(url)
            return result
        else:  # CSV
            result = self._request_csv(url)
            if len(result) > 1:
                return pd.DataFrame(result[1:], columns=result[0])
            else:
                return pd.DataFrame()

    def get_full_studies(self, search_expr, max_studies = 50, fmt = 'json'):
        """Returns all content for a maximum of 100 study records.
        search_expr: search expression as specified in ClinicalTrials.gov API documentation
        max_studies: maximum number of studies to return (1-100)
        fmt: output format: 'json' or 'csv'
            
        returns: full study data in the specified format
        """
        if max_studies > 100 or max_studies < 1:
            raise ValueError("The number of studies can only be between 1 and 100")
        
        # Use appropriate format parameter
        format_param = self.JSON_FORMAT if fmt == 'json' else self.CSV_FORMAT
        
        # Build the URL
        url = f"{self.BASE_URL}studies?{format_param}&markupFormat=legacy&query.term={search_expr}&pageSize={max_studies}"
        print(f"API URL: {url}")
        
        # Make the request based on format
        if fmt == 'json':
            try:
                return self._request_json(url)
            except Exception as e:
                print(f"Error making API request: {e}")
                return {}
        else:  # CSV
            try:
                result = self._request_csv(url)
                if len(result) > 1:
                    # convert the list of csv rows to a pandas dataframe
                    return pd.DataFrame(result[1:], columns=result[0])
                else:
                    return pd.DataFrame()
            except Exception as e:
                print(f"Error making API request: {e}")
                return pd.DataFrame()

    def get_study_count(self, search_expr):
        """Get the count of studies matching a search expression.
        search_expr: search expression as specified in ClinicalTrials.gov API documentation
            
        returns: number of studies matching the search expression
        """
        if not search_expr:
            raise ValueError("The search expression cannot be blank.")
        
        try:
            # Query with minimal fields and pageSize=1 to get just the count
            url = f"{self.BASE_URL}studies?{self.JSON_FORMAT}&query.term={search_expr}&pageSize=1"
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            return data.get("totalCount", 0)
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ClinicalTrials.gov API: {e}")
            return 0
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 0

    def process_to_csv(self, input_dir, output_file):
        """Save downloaded data to CSV format.
        
        input_dir: directory containing the downloaded data
        output_file: path to save the processed CSV file
        """
        df = pd.read_csv(os.path.join(input_dir, 'studies.txt'), sep='|')
        df.to_csv(output_file, index=False)
        print(f'Data processed to CSV at {output_file}')

    def process_to_json(self, input_dir, output_file):
        """Save downloaded data to JSON format.
        
        input_dir: directory containing the downloaded data
        output_file: path to save the processed JSON file
        """
        df = pd.read_csv(os.path.join(input_dir, 'studies.txt'), sep='|')
        df.to_json(output_file, orient='records', lines=True)
        print(f'Data processed to JSON at {output_file}')

    def process_to_xml(self, input_dir, output_file):
        """Save downloaded data to XML format.
        
        input_dir: directory containing the downloaded data
        output_file : str
            Path to save the processed XML file
        """
        df = pd.read_csv(os.path.join(input_dir, 'studies.txt'), sep='|')
        root = ET.Element("ClinicalTrials")
        for _, row in df.iterrows():
            trial = ET.SubElement(root, "Trial")
            for col in df.columns:
                ET.SubElement(trial, col).text = str(row[col])
        tree = ET.ElementTree(root)
        tree.write(output_file)
        print(f'Data processed to XML at {output_file}')

    def __repr__(self):
        """String representation of the downloader."""
        return f"ClinicalTrials.gov client v{self.api_info[0]}, database last updated {self.api_info[1]}"

if __name__ == "__main__":
    # test the downloader
    downloader = ClinicalTrialDownloader()
    print(f"\nClinicalTrials.gov Downloader: {downloader}")
    
    # print the available fields
    print("\nAvailable fields:")
    print("JSON fields (first 10):", downloader.study_fields.get("json", [])[:10])
    print("CSV fields (first 10):", downloader.study_fields.get("csv", [])[:10])
    
    try:
        # Query example
        print("\nQuerying COVID-19 studies (JSON format):")
        covid_studies_json = downloader.query_studies(
            search_expr='COVID-19',
            fields=['NCTId', 'BriefTitle', 'OverallStatus'],
            max_studies=5,
            fmt='json'
        )
        print(f"Results found: {len(covid_studies_json.get('studies', []))}")
        
        # Try CSV format
        print("\nQuerying COVID-19 studies (CSV format):")
        covid_studies_csv = downloader.query_studies(
            search_expr='COVID-19',
            fields=['nct_id', 'brief_title', 'overall_status'],
            max_studies=5,
            fmt='csv'
        )
        print(f"Results shape: {covid_studies_csv.shape}")
        
        # Get study count
        print("\nGetting study count:")
        count = downloader.get_study_count('COVID-19')
        print(f"Number of COVID-19 studies: {count}")
        
        # Get full studies
        print("\nGetting full studies:")
        full_studies = downloader.get_full_studies(
            search_expr='COVID-19',
            max_studies=5
        )
        print(f"Full studies retrieved: {len(full_studies.get('studies', []))}")
        
    except Exception as e:
        print(f"Error in API queries: {e}")
    
    # test the download and processing of the full dataset
    print("\nDo you want to download and process the full dataset? (y/n)")
    answer = input()
    if answer.lower() == 'y':
        print("\nDownloading and processing data:")
        downloader.download_data() # this takes very long as it is downloading the full dataset, shouldn't be used too often
        downloader.process_to_csv('./clinical_trials_data/aact-raw', './clinical_trials_data/clinical_trials.csv')
        downloader.process_to_json('./clinical_trials_data/aact-raw', './clinical_trials_data/clinical_trials.json')
        downloader.process_to_xml('./clinical_trials_data/aact-raw', './clinical_trials_data/clinical_trials.xml')
    else:
        print("Skipping data download and processing.")