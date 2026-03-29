import os
import json
import urllib.request
import argparse
import random
import time
from pathlib import Path

# Zenodo API endpoint for the OpenWorm Movement Database community
API_URL = "https://zenodo.org/api/records?q=communities%3Aopen-worm-movement-database&size={size}&page={page}"
ZENODO_TOKEN = "Ft0hCukTLFRQLv7oi0Pn860Goun7uXJKNOt3Xpwr77CKkrmtiyNOopkB3Wfd"

def fetch_records(max_pages=5):
    """Fetch record metadata from Zenodo API with backoff."""
    print(f"Fetching record metadata from Zenodo...")
    records = []
    page = 1
    retries = 0
    
    while page <= max_pages:
        try:
            url = API_URL.format(size=10, page=page)
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Bearer {ZENODO_TOKEN}')
            
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                hits = data.get('hits', {}).get('hits', [])
                if not hits:
                    break
                records.extend(hits)
                print(f"Fetched page {page} ({len(hits)} records)")
                page += 1
                retries = 0
                time.sleep(1) # Base pacing
                
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait_time = (2 ** retries) * 5
                print(f"HTTP {e.code} on page {page}. Sleeping for {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                if retries > 5:
                    print("Max retries exceeded.")
                    break
            else:
                print(f"Error fetching page {page}: {e}")
                break
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
        if title.startswith('N2 '):
            healthy.append(r)
        else:
            anomalous.append(r)
    return healthy, anomalous

def download_file(url, output_path):
    """Download a file with exponential backoff for 429s."""
    if os.path.exists(output_path):
        print(f"  File {output_path} already exists. Skipping.")
        return
        
    print(f"  Downloading -> {output_path} ...", flush=True)
    
    retries = 0
    while retries < 5:
        try:
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Bearer {ZENODO_TOKEN}')
            with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
                # Chunked download just in case
                while chunk := response.read(8192):
                    out_file.write(chunk)
            print("  Done!")
            time.sleep(1) # Pacing between files
            return
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait_time = (2 ** retries) * 10
                print(f"  HTTP {e.code}: Throttled. Waiting {wait_time}s...", flush=True)
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"  Failed HTTP {e.code}")
                return
        except Exception as e:
            print(f"  Error: {e}")
            return
            
    print("  Failed after max retries.")

def main():
    parser = argparse.ArgumentParser(description="Download OpenWorm Database with Token.")
    parser.add_argument('--healthy', type=int, default=200, help="Number of healthy (N2) records")
    parser.add_argument('--anomalous', type=int, default=200, help="Number of anomalous (mutant) records")
    parser.add_argument('--outdir', type=str, default="data/vjepa_openworm", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    records = fetch_records(max_pages=100)
    healthy, anomalous = categorize_records(records)
    
    print(f"\nFound {len(healthy)} healthy files and {len(anomalous)} anomalous files.")
    
    sample_healthy = random.sample(healthy, min(args.healthy, len(healthy)))
    sample_anomalous = random.sample(anomalous, min(args.anomalous, len(anomalous)))
    
    def process_subset(subset, label):
        print(f"\n--- Downloading {len(subset)} {label} records ---")
        label_dir = Path(args.outdir) / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for i, rec in enumerate(subset, 1):
            title = rec.get('metadata', {}).get('title', 'Unknown')
            print(f"[{i}/{len(subset)}] {title}")
            
            files = rec.get('files', [])
            target_files = [f for f in files if f['key'].endswith('.hdf5') and 'features' not in f['key'].lower()]
            
            if target_files:
                target = target_files[0]
                # Some zenodo file links need explicit token mapping or use the direct link
                url = target['links']['self'].replace(' ', '%20')
                filename = target['key']
                download_file(url, label_dir / filename)
            else:
                print("  No suitable main HDF5 video.")

    process_subset(sample_healthy, "healthy")
    process_subset(sample_anomalous, "anomalous")
    
    print("\nDownload process complete!")

if __name__ == "__main__":
    main()
