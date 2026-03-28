import os
import json
import urllib.request
import argparse
import random
from pathlib import Path

# Zenodo API endpoint for the OpenWorm Movement Database community
API_URL = "https://zenodo.org/api/records?q=communities%3Aopen-worm-movement-database&size={size}&page={page}"

def fetch_records(max_pages=5):
    """Fetch record metadata from Zenodo API."""
    print(f"Fetching record metadata from Zenodo...")
    records = []
    for page in range(1, max_pages + 1):
        try:
            url = API_URL.format(size=10, page=page)
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                hits = data.get('hits', {}).get('hits', [])
                if not hits:
                    break
                records.extend(hits)
                print(f"Fetched page {page} ({len(hits)} records)")
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
            
    return records

def categorize_records(records):
    """Split records into 'Healthy' (N2 Wild-Type) and 'Anomalous' (Mutants)."""
    healthy = []
    anomalous = []
    
    for r in records:
        title = r.get('metadata', {}).get('title', '')
        # N2 is the standard wild-type strain for C. elegans
        if title.startswith('N2 '):
            healthy.append(r)
        else:
            anomalous.append(r)
            
    return healthy, anomalous

def download_file(url, output_path):
    """Download a file with a simple text progress indicator."""
    if os.path.exists(output_path):
        print(f"  File {output_path} already exists. Skipping.")
        return
        
    print(f"  Downloading -> {output_path} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, output_path)
        print(" Done!")
    except Exception as e:
        print(f" Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download a curated subset of the OpenWorm Movement Database.")
    parser.add_argument('--healthy', type=int, default=5, help="Number of healthy (N2) records to download")
    parser.add_argument('--anomalous', type=int, default=5, help="Number of anomalous (mutant) records to download")
    parser.add_argument('--outdir', type=str, default="data/openworm", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    records = fetch_records(max_pages=20) # Fetch up to 2000 records to sample from
    healthy, anomalous = categorize_records(records)
    
    print(f"\nFound {len(healthy)} healthy records and {len(anomalous)} anomalous records.")
    
    # Randomly sample the requested counts
    sample_healthy = random.sample(healthy, min(args.healthy, len(healthy)))
    sample_anomalous = random.sample(anomalous, min(args.anomalous, len(anomalous)))
    
    def process_subset(subset, label):
        print(f"\n--- Downloading {len(subset)} {label} records ---")
        label_dir = Path(args.outdir) / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for i, rec in enumerate(subset, 1):
            title = rec.get('metadata', {}).get('title', 'Unknown')
            print(f"[{i}/{len(subset)}] {title}")
            
            # Find the main video HDF5 file (usually ends in .hdf5 and doesn't have 'features' in the name)
            files = rec.get('files', [])
            target_files = [f for f in files if f['key'].endswith('.hdf5') and 'features' not in f['key'].lower()]
            
            if target_files:
                target = target_files[0]
                url = target['links']['self'].replace(' ', '%20')
                filename = target['key']
                download_file(url, label_dir / filename)
            else:
                print("  No suitable main HDF5 video file found in this record.")

    process_subset(sample_healthy, "healthy")
    process_subset(sample_anomalous, "anomalous")
    
    print("\nDownload process complete!")
    print(f"Data saved to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
