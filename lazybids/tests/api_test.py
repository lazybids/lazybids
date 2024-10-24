import os
import tempfile
import shutil
from pathlib import Path
import openneuro
from lazybids import Dataset
from lazybids.connection import Connection

def create_local_dataset():
    # Create a temporary directory for the local dataset
    # Create the directory ~/.lazybids-test-data if it doesn't exist
    test_data_dir = Path.home() / '.lazybids-test-data'
    dataset_dir = test_data_dir / 'ds005360'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not (dataset_dir / 'dataset_description.json').exists():
        # Download the dataset from OpenNeuro if it doesn't exist
        openneuro.download(dataset='ds005360', target_dir=str(dataset_dir))
    
    # Load the dataset using LazyBIDS
    dataset = Dataset.from_folder(str(dataset_dir))
    
    return dataset, str(dataset_dir)

def get_api_dataset():
    # Create a connection to the local server
    connection = Connection("http://localhost:8000")
    
    # Get the dataset from the API
    datasets = connection.list_datasets()
    print(datasets)
    ds005360 = next(ds for ds in datasets if ds['name'] == 'ds005360')
    print(ds005360)
    return connection.get_dataset(ds005360['id'])

def compare_datasets(local_ds, api_ds):
    # Compare basic dataset information
    assert local_ds.name == api_ds.name, f"Dataset names don't match: {local_ds.name} != {api_ds.name}"
    assert local_ds.bids_version == api_ds.bids_version, f"BIDS versions don't match: {local_ds.bids_version} != {api_ds.bids_version}"
    
    # Compare subjects
    local_subjects = set(local_ds.subjects.keys())
    api_subjects = set(api_ds.subjects.keys())
    assert local_subjects == api_subjects, f"Subject sets don't match: {local_subjects} != {api_subjects}"
    
    # Compare a sample subject
    sample_subject_id = next(iter(local_subjects))
    local_subject = local_ds.subjects[sample_subject_id]
    api_subject = api_ds.subjects[sample_subject_id]
    
    assert local_subject.participant_id == api_subject.participant_id, f"Participant IDs don't match: {local_subject.participant_id} != {api_subject.participant_id}"
    
    # Compare sessions (if any)
    local_sessions = set(local_subject.sessions.keys())
    api_sessions = set(api_subject.sessions.keys())
    assert local_sessions == api_sessions, f"Session sets don't match: {local_sessions} != {api_sessions}"
    
    if local_sessions:
        # Compare a sample session
        sample_session_id = next(iter(local_sessions))
        local_session = local_subject.sessions[sample_session_id]
        api_session = api_subject.sessions[sample_session_id]
        
        assert local_session.session_id == api_session.session_id, f"Session IDs don't match: {local_session.session_id} != {api_session.session_id}"
        
        # Compare scans
        local_scans = set(local_session.scans.keys())
        api_scans = set(api_session.scans.keys())
        assert local_scans == api_scans, f"Scan sets don't match: {local_scans} != {api_scans}"
        
        # Compare a sample scan
        sample_scan_id = next(iter(local_scans))
        local_scan = local_session.scans[sample_scan_id]
        api_scan = api_session.scans[sample_scan_id]
        
        assert local_scan.name == api_scan.name, f"Scan names don't match: {local_scan.name} != {api_scan.name}"
        assert len(local_scan.files) == len(api_scan.files), f"Number of files don't match: {len(local_scan.files)} != {len(api_scan.files)}"
        assert local_scan.fields == api_scan.fields, f"Scan fields don't match: {local_scan.fields} != {api_scan.fields}"
    
    # Compare subject-level scans (if any)
    local_subject_scans = set(local_subject.scans.keys())
    api_subject_scans = set(api_subject.scans.keys())
    assert local_subject_scans == api_subject_scans, f"Subject-level scan sets don't match: {local_subject_scans} != {api_subject_scans}"

def main():
    try:
        import rich
        print("Creating local dataset...")
        local_dataset, temp_dir = create_local_dataset()
        
        print("Getting API dataset...")
        api_dataset = get_api_dataset()
        print(api_dataset)
        print("Comparing datasets...")
        compare_datasets(local_dataset, api_dataset)
        
        print("All tests passed successfully!")
    
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'temp_dir' in locals():
            print("Cleaning up temporary directory...")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
