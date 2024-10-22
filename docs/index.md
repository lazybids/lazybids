# LazyBIDS Documentation

LazyBIDS is a Python package designed to simplify interactions with BIDS (Brain Imaging Data Structure) datasets. It provides an intuitive object-oriented interface for working with BIDS data, both locally and remotely through the [lazybids-ui](https://github.com/lazybids/lazybids-ui) server.

## **Overview**

LazyBIDS serves two main purposes:

1. An I/O library for interacting with local BIDS datasets
2. A client for remote interaction with datasets stored on a [lazybids-ui](https://github.com/lazybids/lazybids-ui) server

This dual functionality allows users to work with BIDS data consistently, regardless of whether it's stored locally or on a remote server.  

## **Installation**

Install the latest version of LazyBIDS using pip:
```bash
pip install lazybids
```  

## **Key Features**

- Access all metadata of Dataset, Subject, Session, or Scan objects using the `all_metadata` property
- Retrieve contents of scan/measurement level .tsv files as pandas DataFrames via the `Scan.table` parameter
- Support for various imaging formats (via SimpleITK), including .nii, .nii.gz, and DICOM folders
- Limited support for MEG/EEG data
- Control memory usage by caching scan pixel/voxel data selectively
- Access scan pixel/voxel data as SimpleITK images or numpy arrays
- Seamless interaction with remote datasets on a lazybids-ui server, downloading only necessary data

