## Basic Usage

### Local Dataset

```python
import lazybids
dataset_dir = './templates/'
ds = lazybids.Dataset.from_folder(dataset_dir, load_scans_in_memory=False)
print(ds)
```
### Remote Dataset (via lazybids-ui server)

```python
import lazybids
connection = lazybids.Connection("http://localhost:8000")
ds = lazybids.connection.get_dataset('ds005360')
print(ds)
```

## Working with BIDS Objects

LazyBIDS represents BIDS data as a hierarchy of objects:

1. Dataset
2. Subject
3. Session
4. Scan

Each level can be accessed and iterated over using dictionary-like syntax.

### Subjects

```python
for subject in ds.subjects.values():
    print(subject)
```
### Sessions

```python
for ses in subject.sessions.values():
    print(ses)
```
### Scans

```python
for scan in ses.scans.values():
    print(scan)
```

## Accessing Scan Data

LazyBIDS provides multiple ways to access scan data:

```python
# As a SimpleITK Image
print(type(scan.data))
# As a numpy array
print(type(scan.numpy))
print(f"Shape: {scan.numpy.shape}")
```

## Memory Management

LazyBIDS allows fine-grained control over memory usage:

- Use `load_scans_in_memory=False` when creating a Dataset to avoid loading all scan data into memory
- Use `Dataset.load_scans()`, `Subject.load_scans()`, or `Session.load_scans()` to load scans selectively
- Use `Scan.load()` and `Scan.unload()` to manage individual scan data

## Metadata Access

All metadata for BIDS objects can be accessed via the `all_metadata` property, which combines information from filenames, JSON sidecars, and NIFTI/DICOM metadata.

## Working with Remote Datasets

LazyBIDS integrates seamlessly with the [lazybids-ui](https://github.com/lazybids/lazybids-ui) server, allowing you to work with large, online datasets as if they were local. This feature downloads only the parts of the dataset you need, making it efficient for working with extensive collections.

To use this feature:

1. Set up a lazybids-ui server
2. Create a Connection object with the server URL
3. Use the connection to retrieve datasets and interact with them as you would with local data

```python
connection = lazybids.Connection("http://your-lazybids-ui-server.com")
remote_ds = connection.get_dataset('dataset_name')
Work with remote_ds as if it were a local dataset
for subject in remote_ds.subjects.values():
    print(subject.participant_id)
```

