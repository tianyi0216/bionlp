import os
import requests

class BaseDownloader:
    """Base class for downloading datasets."""
    
    def __init__(self, dataset_name, url, save_dir="clinical_trials_data"):
        self.dataset_name = dataset_name
        self.url = url
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

        response = requests.get(self.url, stream=True)
        with open(self.file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        print(f"✅ {self.dataset_name} downloaded successfully!")
        return self.file_path
