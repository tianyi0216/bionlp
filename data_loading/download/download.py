import os
import requests

class BaseDownloader:
    """Base class for downloading datasets."""
    
    def __init__(self, dataset_name, urls, save_dir="clinical_trials_data"):
        self.dataset_name = dataset_name
        self.urls = urls
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, f"{dataset_name}.zip")

    def download(self):
        """Download the dataset."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if os.path.exists(self.file_path):
            print(f"✅ {self.dataset_name} already downloaded. Skipping...")
            return self.file_path
        
        print(f"⬇️ Downloading {self.dataset_name} dataset...")

        # Download all urls
        success_urls = []
        failed_urls = []
        for url in self.urls:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                with open(self.file_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                
                print(f"✅ {self.dataset_name} downloaded successfully!")
                success_urls.append(url)
                
            except requests.RequestException as e:
                print(f"⚠️ Failed to download from {url}: {str(e)}")
                failed_urls.append(url)
                continue
        
        print(f"✅ {self.dataset_name} downloaded successfully from {len(success_urls)} URLs")
        print(f"❌ Failed to download from {len(failed_urls)} URLs")
        return success_urls, failed_urls
