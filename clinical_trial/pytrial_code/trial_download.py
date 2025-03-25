# This file is used to download the clinical trial data from ClinicalTrials.gov. Most of the available clinical trial data is gotten from the ClinicalTrials.gov
# The data is downloaded in the form of a zip file, which is then extracted into the aact-raw directory.
# The data is then given the option to be processed into a csv, json, or xml file.
# There is also another code to download the data from the ClinicalTrials.gov API, in a different directory.

import os
import requests
import zipfile
import pandas as pd
import xml.etree.ElementTree as ET
import json


class ClinicalTrialDownloader:
    """Utilities for downloading clinical trial data from ClinicalTrials.gov."""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/"
    RAW_TXT_DIR = './aact-raw'
    
    # Default fields available from the API, this can be changed to include more fields and customized depending on the needs
    DEFAULT_FIELDS = [
        'NCTId',                  
        'BriefTitle',             
        'BriefSummary',           
        'DetailedDescription',    
        'EligibilityCriteria',    
        'Condition',              
        'OverallStatus'           
    ]

    def __init__(self, api_key = None):
        """Initialize downloader.
        api_key : optional in case needed
        """
        self.api_key = api_key
        self.api_info = self._get_api_info()

    def _get_headers(self):
        """
        Get headers for API requests.
        """
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        # in case of using the API key, uncomment the following line and make relevant changes
        # if self.api_key:
        #     headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def _get_api_info(self):
        """
        Get API version and last update information.
        """
        try:
            version_url = f"{self.BASE_URL}version"
            response = requests.get(version_url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            api_version = data.get("apiVersion", "unknown")
            last_updated = data.get("dataTimestamp", "unknown")
            
            return api_version, last_updated
        except Exception as e:
            print(f"Error getting API info: {e}")
            return "unknown", "unknown"

    def _request_json(self, url):
        """Make a JSON request to the API.
        """
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

    @property
    def study_fields(self):
        """Get all possible fields available from ClinicalTrials.gov.
        Pytrial's API call to get fields is not working, for now, it seems it has to be manually set or defined.
        """
        return self.DEFAULT_FIELDS

    def download_data(self, date = '20220501', output_dir = './clinical_trials_data'):
        """Download the full dataset from ClinicalTrials.gov as a zip file.
        
        date : Date of the database copy in YYYYMMDD format
        output_dir : Output directory for the downloaded data
        """
        url = f'https://aact.ctti-clinicaltrials.org/static/exported_files/daily/{date}_pipe-delimited-export.zip'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f'Downloading data from {url}...')
        filename = os.path.join(output_dir, 'clinical_trials.zip')
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
                
            extract_dir = os.path.join(output_dir, self.RAW_TXT_DIR)
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            print(f'Data downloaded and extracted to {extract_dir}')
        except Exception as e:
            print(f"Error downloading data: {e}")

    def query_studies(self, 
                     search_expr,
                     fields = None,
                     max_studies = 50,
                     return_fmt = 'json'):
        """Query study content for specified fields from the ClinicalTrials.gov API.
        
        search_expr : Search expression as specified in ClinicalTrials.gov API documentation
        fields : List of fields to retrieve
        max_studies : Maximum number of studies to return (1-1000)
        return_fmt : Return format ('json' or 'csv')
            
        return: Query results in specified format
        """
        if max_studies > 1000 or max_studies < 1:
            raise ValueError("The number of studies can only be between 1 and 1000")
            
        # Use default fields if none provided
        if fields is None:
            fields = ['NCTId', 'BriefTitle', 'Condition', 'InterventionName', 'OverallStatus']
        
        # converting snake_case field names to CamelCase if needed
        camel_case_fields = []
        for field in fields:
            if field.islower():
                # converting snake_case to CamelCase
                parts = field.split('_')
                camel_field = parts[0] + ''.join(p.capitalize() for p in parts[1:])
                camel_case_fields.append(camel_field)
            else:
                camel_case_fields.append(field)
        
        # join fields with pipe separator
        fields_param = "|".join(camel_case_fields)
        
        # building the URL
        url = f"{self.BASE_URL}studies?format=json&query.term={search_expr}&markupFormat=legacy&fields={fields_param}&pageSize={max_studies}"
        print(f"API URL: {url}")
        
        # Make request
        result = self._request_json(url)
        
        # Convert to DataFrame if CSV format requested
        if return_fmt.lower() == 'csv' and 'studies' in result:
            studies = result.get('studies', [])
            if studies:
                return pd.json_normalize(studies)
            return pd.DataFrame()
        
        return result

    def get_full_studies(self, 
                        search_expr,
                        max_studies = 50,
                        return_fmt = 'json'):
        """Get full study data for a maximum of 100 study records.
        
        search_expr : Search expression as specified in ClinicalTrials.gov API documentation
        max_studies : Maximum number of studies to return (1-100)
        return_fmt : Return format ('json' or 'csv')
            
        return: Full study data in the specified format
        """
        if max_studies > 100 or max_studies < 1:
            raise ValueError("The number of studies can only be between 1 and 100")
        
        # Build URL - always request JSON
        url = f"{self.BASE_URL}studies?format=json&query.term={search_expr}&pageSize={max_studies}"
        print(f"API URL: {url}")
        
        # Make request
        result = self._request_json(url)
        
        # Convert to DataFrame if CSV format requested
        if return_fmt.lower() == 'csv' and 'studies' in result:
            studies = result.get('studies', [])
            if studies:
                return pd.json_normalize(studies)
            return pd.DataFrame()
            
        return result

    #Seems like totalCount is not available in the API response, so we need to use a different approach.
    def get_study_count(self, search_expr):
        """Get the count of studies matching a search expression.
        search_expr : Search expression as specified in ClinicalTrials.gov API documentation
            
        return: Number of studies matching the search expression
        """
        if not search_expr:
            raise ValueError("Search expression cannot be blank.")
        
        try:
            # try to get totalCount directly from API response
            url = f"{self.BASE_URL}studies?format=json&query.term={search_expr}&pageSize=1"
            data = self._request_json(url)
            
            # check if totalCount is available
            if "totalCount" in data and data["totalCount"] > 0:
                return data["totalCount"]
            
            # If totalCount is not available or is zero, check if we still have results
            if "studies" in data and len(data["studies"]) > 0:
                # if we have studies but no totalCount, we need to use a different approach
                url_large = f"{self.BASE_URL}studies?format=json&query.term={search_expr}&pageSize=100"
                data_large = self._request_json(url_large)
                studies_returned = len(data_large.get("studies", []))
                
                # if we have a nextPageToken, there are more results
                if "nextPageToken" in data_large and data_large["nextPageToken"]:
                    print(f"Warning: More than {studies_returned} studies match the query. Returning a partial count.")
                    return studies_returned + 1  # +1 to indicate there are more
                
                return studies_returned
            
            return 0  # No studies found
        except Exception as e:
            print(f"Error getting study count: {e}")
            return 0

    def process_to_csv(self, input_dir, output_file):
        """Save downloaded data to CSV format.
        
        input_dir : Directory containing the downloaded data
        output_file : Path to save the processed CSV file
        """
        try:
            df = pd.read_csv(os.path.join(input_dir, 'studies.txt'), sep='|')
            df.to_csv(output_file, index=False)
            print(f'Data processed to CSV at {output_file}')
        except Exception as e:
            print(f"Error processing to CSV: {e}")

    def process_to_json(self, input_dir, output_file):
        """Save downloaded data to JSON format.
        
        input_dir : Directory containing the downloaded data
        output_file : Path to save the processed JSON file
        """
        try:
            df = pd.read_csv(os.path.join(input_dir, 'studies.txt'), sep='|')
            df.to_json(output_file, orient='records', lines=True)
            print(f'Data processed to JSON at {output_file}')
        except Exception as e:
            print(f"Error processing to JSON: {e}")

    def process_to_xml(self, input_dir, output_file):
        """Save downloaded data to XML format.
        
        input_dir : Directory containing the downloaded data
        output_file : Path to save the processed XML file
        """
        try:
            df = pd.read_csv(os.path.join(input_dir, 'studies.txt'), sep='|')
            root = ET.Element("ClinicalTrials")
            
            for _, row in df.iterrows():
                trial = ET.SubElement(root, "Trial")
                for col in df.columns:
                    ET.SubElement(trial, col).text = str(row[col])
                    
            tree = ET.ElementTree(root)
            tree.write(output_file)
            print(f'Data processed to XML at {output_file}')
        except Exception as e:
            print(f"Error processing to XML: {e}")

    def convert_json_to_csv(self, json_data):
        """Convert JSON data to CSV format.
        
        json_data : JSON data to convert
            
        return: CSV data as DataFrame
        """
        studies = json_data.get('studies', [])
        if studies:
            return pd.json_normalize(studies)
        return pd.DataFrame()

    def __repr__(self):
        """String representation of the downloader.
        return: String representation
        """
        return f"ClinicalTrials.gov client v{self.api_info[0]}, database last updated {self.api_info[1]}"


# Basic utilities for direct API requests
def request_ct(url: str) -> requests.Response:
    """Performs a get request that provides a useful error message.
    
    url : URL to request
        
    Returns
    -------
    requests.Response
        Response object
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.HTTPError as ex:
        raise ex
    except ImportError:
        raise ImportError(
            "Couldn't retrieve the data, check your search expression or try again later."
        )
    else:
        return response


def json_handler(url):
    """Returns request in JSON (dict) format
    
    url : URL to request
        
    return: JSON response
    """
    return request_ct(url).json()


def csv_handler(url):
    """Returns request in CSV (list of records) format
    
    url : URL to request
        
    return: CSV records
    """
    import csv
    from io import StringIO
    
    response = request_ct(url)
    decoded_content = response.content.decode("utf-8")

    cr = csv.reader(decoded_content.splitlines(), delimiter=",")
    records = list(cr)

    return records


if __name__ == "__main__":
    # Test the downloader
    downloader = ClinicalTrialDownloader()
    print(f"\nClinicalTrials.gov Downloader: {downloader}")
    
    # Print the available fields
    print("\nAvailable fields:")
    print("Available fields (first 10):", downloader.study_fields[:10])
    
    try:
        # Query example (JSON format)
        print("\nQuerying COVID-19 studies (JSON format):")
        covid_studies_json = downloader.query_studies(
            search_expr='COVID-19',
            fields=['NCTId', 'BriefTitle', 'OverallStatus'],
            max_studies=5,
            return_fmt='json'
        )
        print(f"Results found: {len(covid_studies_json.get('studies', []))}")
        
        # Query example (CSV format)
        print("\nQuerying COVID-19 studies (CSV format):")
        covid_studies_csv = downloader.query_studies(
            search_expr='COVID-19',
            fields=['NCTId', 'BriefTitle', 'OverallStatus'],
            max_studies=5,
            return_fmt='csv'
        )
        print(f"Results shape: {covid_studies_csv.shape}")
        
        # Get study count
        print("\nGetting study count:")
        count = downloader.get_study_count('COVID')
        print(f"Number of COVID-19 studies: {count}")
        
        # Get full studies
        print("\nGetting full studies:")
        full_studies = downloader.get_full_studies(
            search_expr='COVID-19',
            max_studies=5
        )
        print(f"Full studies retrieved: {len(full_studies.get('studies', []))}")
        
        # Test basic utility functions
        print("\nTesting basic utility functions:")
        basic_url = "https://clinicaltrials.gov/api/v2/studies?format=json&query.term=covid-19&pageSize=2"
        basic_json = json_handler(basic_url)
        print(f"Basic JSON handler result count: {len(basic_json.get('studies', []))}")
        
    except Exception as e:
        print(f"Error in API queries: {e}")
    
    # Test the download and processing of the full dataset (commented out)
    # print("\nDo you want to download and process the full dataset? (y/n)")
    # answer = input()
    # if answer.lower() == 'y':
    #     print("\nDownloading and processing data:")
    #     downloader.download_data()  # Takes a long time, use sparingly
    #     downloader.process_to_csv('./clinical_trials_data/aact-raw', './clinical_trials_data/clinical_trials.csv')
    #     downloader.process_to_json('./clinical_trials_data/aact-raw', './clinical_trials_data/clinical_trials.json')
    #     downloader.process_to_xml('./clinical_trials_data/aact-raw', './clinical_trials_data/clinical_trials.xml')
    # else:
    #     print("Skipping data download and processing.")