# LazyBIDS

Python package to (lazily) interact with BIDS datasets. Lazybids enables interaction with bids-datasets as python objects, similar to how [xnatpy](https://xnat.readthedocs.io/en/latest/static/tutorial.html) interacts with XNAT datasets, both locally and on a [lazybids-ui](https://github.com/lazybids/lazybids-ui) server.

Lazybids serves both as an io-library to interact with local bids datasets, as well as a client allowing you to interact with datasets with datasets stored on the [lazybids-ui](https://github.com/lazybids/lazybids-ui) server verry similar to how you would with a local dataset, without having to download the entire dataset. (See [Server-Client example](#server-client-example)) ([lazybids-ui](https://github.com/lazybids/lazybids-ui), in turn uses lazybids as it's core io-library.)

Install the latest version:
```bash
pip install lazybids
```

## Documentation

For detailed documentation, please visit our [GitHub Pages](https://lazybids.github.io/lazybids/).


### Example:
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lazybids/lazybids/blob/master/examples/bids_starter_kit_load.ipynb)

### Server-Client example:
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lazybids/lazybids/blob/master/examples/api_test.ipynb)

Please note that subjects, sessions and scans act as dictionaries with resp. the participant_id, session_id and scan name as key.

## Notable features:
- Access all metadata of a Dataset, Subject, Session or Scan using the all_metadata property. This combines variables from filenames, .json (sidecars) and nifti/dicom metadata.
- Access contents of scan/measurment level .tsv files using pandas from the Scan.table parameter
- All imaging formats supported by SimpleITK, including .nii, .nii.gz and DICOM-folders should work (*DICOM support not tested). As well as .tsv and .json sidecar/metadata formats.
MEG/Eeg support is limited at this time.
- You can control if scan's pixel/voxel data is cached in memory using the 'load_scans_in_memory' parameter on creation or using load_scans() function of a Dataset,Subject,or Sessions, or the Scan.load() and Scan.unload() functions.
- Scan meta-data is always loaded
- Access scan pixel/voxel data using SimpleITK images from the Scan.data property, or as numpy array using Scan.numpy
- Work with huge, online, datasets on a lazybids-ui server as if it was a local dataset, downloading only the parts you need.

## Roadmap
- Implement writing datasets to disk
- Improve capabilities of changing/updating existing datasets
- Add MEG/EEG support (e.g. MNE-python)

## Example usage
```python
import lazybids
local = True
if local:
    dataset_dir = './templates/'
    ds = lazybids.Dataset.from_folder(dataset_dir, load_scans_in_memory=False)
# OR Get the dataset from the API
else:
    connection = lazybids.Connection("http://localhost:8000")
    ds = lazybids.connection.get_dataset('ds005360')
# 
print(ds)
# Output:
# Dataset(
#     name='',
#     folder=WindowsPath('./lazyBIDS/examples/bids-starter-kit-template/templates'),
#     json_file=WindowsPath('./lazyBIDS/examples/bids-starter-kit-template/templates/dataset_description.json'),
#     participants_json=None,
#     bids_version='1.8.0',
#     HEDVersion=None,
#     authors=['', '', ''],
#     fields={},
#     description=None,
#     dataset_type='raw',
#     how_to_acknowledge='',
#     acknowledgements='',
#     funding=['', '', ''],
#     ethics_approvals=[''],
#     references_and_links=['', '', ''],
#     source_datasets=None,
#     license='',
#     dataset_doi='doi:',
#     subject_variables_metadata=None,
#     hed_version='8.2.0'
# )
```

```python
for subject in ds.subjects.values():
    print(subject)

# Subject(
#     participant_id='sub-01',
#     folder=WindowsPath('./lazyBIDS/examples/bids-starter-kit-template/templates/sub-01'),
#     scan_metadata={},
#     fields=None,
#     age=0,
#     sex='m',
#     handedness='l',
#     n_sessions=1,
#     n_scans=0
# )
# Subject(
#     participant_id='sub-epilepsy01',
#     folder=WindowsPath('./lazyBIDS/examples/bids-starter-kit-template/templates/sub-epilepsy01'),
#     scan_metadata={},
#     fields=None,
#     age=10,
#     sex='f',
#     handedness='r',
#     n_sessions=1,
#     n_scans=0
# )

```

```python
for ses in subject.sessions.values():
    print(ses)
# Session(
#     folder=WindowsPath('E:/git/lazyBIDS/examples/bids-starter-kit-template/templates/sub-epilepsy01/ses-01'),
#     scans={
#         'sub-epilepsy01_ses-01_electrodes': Scan(
#             name='sub-epilepsy01_ses-01_electrodes',
#             files=[],
#             metadata_files=[
#                 'E:\\git\\lazyBIDS\\examples\\bids-starter-kit-template\\templates\\sub-epilepsy01\\ses-01\\./ieeg\
# \.\\sub-epilepsy01_ses-01_electrodes.json',
#                 'E:\\git\\lazyBIDS\\examples\\bids-starter-kit-template\\templates\\sub-epilepsy01\\ses-01\\./ieeg\
# \.\\sub-epilepsy01_ses-01_electrodes.tsv'
#             ],
#             fields={
#                 'name': {'Description': 'REQUIRED. Name of the electrode contact point.'},
#                 'x': {
#                     'Description': 'REQUIRED. X position. The positions of the center of each electrode in xyz 
# space. Units are specified in space-<label>_coordsystem.json.'
#                 },
#                 'y': {'Description': 'REQUIRED. Y position.'},
#                 'z': {
#                     'Description': 'REQUIRED. Z position. If electrodes are in 2D space this should be a column of 
# n/a values.'
#                 },
#                 'size': {'Description': 'REQUIRED. Surface area of the electrode, units MUST be in mm^2.'},
#                 'seizure_zone': {
#                     'LongName': 'Seizure onset zone',
#                     'Description': 'final conclusion drawn by an epileptologist on the electrodes involved in the 
# seizures',
#                     'Levels': {
#                         'SOZ': 'Seizure Onset Zone, the region where the recorded clinical seizures originated 
# during the recording period.',
#                         'IrritativeZone': 'Region of cortex that generates interictal epileptiform discharges, but 
# not seizures',
#                         'EarlyPropagationZone': 'Region of cortex that generates the initial seizure symptoms. Not 
# seizure onset, but the propagation of seizure from SOZ into this region within first 3 seconds from seizure 
# onset.',
#                         'Resected': 'Region of cortex that was resected',
#                         'ResectedEdge': 'Region of cortex that is within 1 cm of the edge of the resected area.'
#                     }
#                 },
#                 'modality': 'ieeg'
#             },
#             table=  name  x  y  z  size   seizure_zone
# 0  TO1  0  0  0     5         NonSOZ
# 1  TO2  0  0  0     5  SOZ, Resected,
#             n_files=0
#         )
#     },
#     scan_metadata={},
#     fields=None,
#     participant_id='ses-01',
#     session_id='ses-01',
#     n_scans=1
# )

for scan in ses.scans.values():
    print(scan)

# Scan(
#     name='sub-epilepsy01_ses-01_electrodes',
#     files=[],
#     metadata_files=[
#         'E:\\git\\lazyBIDS\\examples\\bids-starter-kit-template\\templates\\sub-epilepsy01\\ses-01\\./ieeg\\.\\sub-
# epilepsy01_ses-01_electrodes.json',
#         'E:\\git\\lazyBIDS\\examples\\bids-starter-kit-template\\templates\\sub-epilepsy01\\ses-01\\./ieeg\\.\\sub-
# epilepsy01_ses-01_electrodes.tsv'
#     ],
#     fields={
#         'name': {'Description': 'REQUIRED. Name of the electrode contact point.'},
#         'x': {
#             'Description': 'REQUIRED. X position. The positions of the center of each electrode in xyz space. Units
# are specified in space-<label>_coordsystem.json.'
#         },
#         'y': {'Description': 'REQUIRED. Y position.'},
#         'z': {
#             'Description': 'REQUIRED. Z position. If electrodes are in 2D space this should be a column of n/a 
# values.'
#         },
#         'size': {'Description': 'REQUIRED. Surface area of the electrode, units MUST be in mm^2.'},
#         'seizure_zone': {
#             'LongName': 'Seizure onset zone',
#             'Description': 'final conclusion drawn by an epileptologist on the electrodes involved in the 
# seizures',
#             'Levels': {
#                 'SOZ': 'Seizure Onset Zone, the region where the recorded clinical seizures originated during the 
# recording period.',
#                 'IrritativeZone': 'Region of cortex that generates interictal epileptiform discharges, but not 
# seizures',
#                 'EarlyPropagationZone': 'Region of cortex that generates the initial seizure symptoms. Not seizure 
# onset, but the propagation of seizure from SOZ into this region within first 3 seconds from seizure onset.',
#                 'Resected': 'Region of cortex that was resected',
#                 'ResectedEdge': 'Region of cortex that is within 1 cm of the edge of the resected area.'
#             }
#         },
#         'modality': 'ieeg'
#     },
#     table=  name  x  y  z  size   seizure_zone
# 0  TO1  0  0  0     5         NonSOZ
# 1  TO2  0  0  0     5  SOZ, Resected,
#     n_files=0
# )
```

```python
print(type(scan.data))
print([type(scan.numpy), f'shape: {scan.numpy.shape}'])

# <class 'SimpleITK.SimpleITK.Image'>
# [<class 'numpy.ndarray'>, 'shape: (424, 640, 3)']

```
