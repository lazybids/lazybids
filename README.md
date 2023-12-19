# LazyBIDS

# This is a very early proof of concept, expect things to fail/break or change!!!
Python package to (lazily) interact with BIDS datasets.

Install the latest version:
```bash
pip install git+https://github.com/roelant001/lazybids.git
```
Or install a tagged release from [test pypi](https://test.pypi.org/project/lazybids/#files]):
```bash
pip install -i https://test.pypi.org/simple/ lazybids
```

Example:
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roelant001/lazybids/blob/master/examples/bids_starter_kit_load.ipynb)

Please note that subjects and experiments act as lists, whereas scans are dictionaries. This behaviour should probably be harmonized in a future release.

```python
import lazybids
dataset_dir = './templates/'
ds = lazybids.Dataset.from_folder(dataset_dir, load_scans_in_memory=False)
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
for subject in ds.subjects:
    print(subject)

# Subject(
#     participant_id='sub-01',
#     folder=WindowsPath('./lazyBIDS/examples/bids-starter-kit-template/templates/sub-01'),
#     scan_metadata={},
#     fields=None,
#     age=0,
#     sex='m',
#     handedness='l',
#     n_experiments=1,
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
#     n_experiments=1,
#     n_scans=0
# )

```

```python
for exp in subject.experiments:
    print(exp)
# Experiment(
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

for scan in exp.scans.values():
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
