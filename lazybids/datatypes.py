from pkg_resources import resource_filename
from typing import Union, List, Dict, Optional, Any
from pathlib import Path
import json
import os
import glob
import yaml
import pandas as pd
import SimpleITK as sitk
import numpy as np
import shutil

from pydantic import BaseModel, computed_field, Field, Extra, ConfigDict
from pydantic.dataclasses import dataclass
from .utils import utils

# Setting up logging
from rich.logging import RichHandler
#from tqdm.rich import trange, tqdm
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


# Load YAML configurations
def load_yaml_config(file_path: str):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


config_paths = [
    resource_filename("lazybids", "./resources/schema/objects/modalities.yaml"),
    resource_filename("lazybids", "./resources/schema/objects/suffixes.yaml"),
    resource_filename("lazybids", "./resources/schema/objects/metadata.yaml"),
    resource_filename("lazybids", "./resources/schema/objects/extensions.yaml"),
    resource_filename("lazybids", "./resources/schema/objects/datatypes.yaml"),
    resource_filename("lazybids", "./resources/schema/objects/columns.yaml"),
]

modalities, suffix, metadata, extensions, datatypes, columns = [
    load_yaml_config(path) for path in config_paths
]
itk_supported_datatypes = json.load(
    open(resource_filename("lazybids", "./resources/itk_supported_extensions.json"), "r")
)


# Function to read image metadata
def read_image_meta_data(fname: Union[Path, str]) -> dict:
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    reader = (
        sitk.ImageFileReader() if os.path.isfile(fname) else sitk.ImageSeriesReader()
    )
    if isinstance(reader, sitk.ImageSeriesReader):
        dicom_names = reader.GetGDCMSeriesFileNames(str(fname))
        if not dicom_names:
            return None
        reader.SetFileNames(dicom_names)
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        return {k: image.GetMetaData(k) for k in image.GetMetaDataKeys()}
    else:
        reader.SetFileName(str(fname))
        reader.ReadImageInformation()
        return {k: reader.GetMetaData(k) for k in reader.GetMetaDataKeys()}


# Function to read an image
def read_image(files: List[Path]) -> sitk.Image:
    if not files:
        raise ValueError("No files provided for reading image")
    fname = files[0]
    reader = (
        sitk.ImageFileReader() if os.path.isfile(fname) else sitk.ImageSeriesReader()
    )
    if isinstance(reader, sitk.ImageSeriesReader):
        dicom_names = reader.GetGDCMSeriesFileNames(str(fname))
        reader.SetFileNames(dicom_names)
    else:
        reader.SetFileName(str(fname))
    return reader.Execute()


def load_scans(self):
    for modality in datatypes.keys():
        if os.path.isdir(os.path.join(self.folder, "./", modality)):
            logging.debug(f'loading modality folder: {os.path.join(self.folder, "./", modality)}')
            for file in glob.glob(os.path.join(self.folder, "./", modality, "./*")):
                basename, ext = utils.get_basename_extension(file)
                if ext == ".json":
                    if not (basename in self.scans.keys()):
                        self.scans[basename] = Scan().from_json(file, parent=self)
                    else:
                        self.scans[basename].from_json(file)
                elif ext == ".tsv":
                    if not (basename in self.scans.keys()):
                        self.scans[basename] = Scan().from_tsv(file, parent=self)
                    else:
                        self.scans[basename].from_tsv(file)
                elif ext.lower() in itk_supported_datatypes:
                    if not (basename in self.scans.keys()):
                        self.scans[basename] = Scan().from_file(file, parent=self)
                    else:
                        self.scans[basename].from_file(file)
                if basename in self.scans.keys():
                    self.scans[basename].fields['modality'] = modality
                else:
                    logging.warning(f"Scan file is not supported: {file}")



class Scan(BaseModel, extra=Extra.allow):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = ""
    parent: Any = Field(None, repr=False)
    # filetype: str = None
    files: List[Path] = []
    metadata_files: List[Path] = []
    fields: Optional[Union[dict,  None]] = {}
    table: Optional[Union[pd.DataFrame,  None]] = None
    _loaded: bool = False
    _cached_image: Optional[Union[sitk.Image,  None]] = None

    def from_file(self, file: Path, parent=None):
        if parent:
            self.parent = parent
        if type(file) == list:
            self.files = file
        else:
            self.files = [file]
        if not(self.name):

            self.name = utils.get_basename_extension(file)[0]
        new_fields = read_image_meta_data(file)
        for k,v in new_fields.items():
            if k in self.fields.keys():
                assert self.fields[k] == v, f"Field {k} is not the same in both files"
            else:
                self.fields[k] = v

        return self
    
    def from_json(self, file: Path, parent=None):
        if parent:
            self.parent = parent
        if not(self.name):
            self.name = utils.get_basename_extension(file)[0]
        for k,v in json.load(open(file, "r")).items():
            if k in self.fields.keys():
                assert self.fields[k] == v, f"Field {k} is not the same in both files"
            else:
                self.fields[k] = v
        self.metadata_files.append(file)
        return self

    def from_tsv(self, file: Path, parent=None):
        if parent:
            self.parent = parent
        if not(self.name):
            self.name = utils.get_basename_extension(file)[0]
        self.table = pd.read_csv(file, sep="\t")
        self.metadata_files.append(file)
        return self

    def write(self, folder):

        for file in self.files + self.metadata_files:
            new_file = os.path.join(folder, os.path.split(file)[1])
            shutil.copy(file,new_file)
            logging.info('copying {file} to {new_file}')
        if self.table is not None:
            self.table.to_csv(os.path.join(folder, self.name +'.tsv'),sep='\t', index=False)
        if self.fields:
            with open(os.path.join(folder,self.name + '.json'),'w') as outfile:
                json.dump(self.fields, outfile)
        
    @property
    def data(self) -> sitk.Image:
        if not (self._loaded):
            self.__cached_image = read_image(self.files)
            self._loaded = True
        return self.__cached_image
    
    @property
    def numpy(self) -> np.array:
        return sitk.GetArrayFromImage(self.data)

    
    def load(self) -> None:
        if self.files:
            self.data
    
    def unload(self) -> None:
        self.__cached_image = None
        self._loaded = False


    @computed_field
    @property
    def n_files(self) -> int:
        if not (self.files):
            return 0
        else:
            return len(self.files)

    @property
    def all_meta_data(self) -> dict:
        all_data = self.model_dump(exclude=['parent','data','numpy'])
        if self.fields:
            del all_data["fields"]
            all_data.update(self.fields)
        
        return all_data


class Session(BaseModel, extra=Extra.allow):
    folder: Path
    parent: Any = Field(None, repr=False)
    scans: Optional[Dict[str, Scan]] = Field(default_factory=dict, repr=False)
    scan_metadata: Optional[Union[dict,  None]] = {}
    fields: Optional[Union[dict,  None]] = None
    load_scans = load_scans
    path_vars_key: Optional[List[str]] = []

    def from_dict(exp_dict=None, dataset_folder="", parent=None):
        if os.path.isdir(
            os.path.join(
                dataset_folder,
                "./",
                exp_dict["participant_id"],
                "./" + exp_dict["session_id"],
            )
        ):
            exp_dict["folder"] = os.path.join(
                dataset_folder,
                "./",
                exp_dict["participant_id"],
                "./" + exp_dict["session_id"],
            )
        ses = Session(**exp_dict)
        if parent:
            ses.parent = parent
        if exp_dict["folder"]:
            ses.load_scans()
        return ses

    def from_folder(exp_dir="", parent=None):
        assert os.path.isdir(exp_dir), "Folder does not exist"
        pt_dict = {"participant_id": os.path.split(exp_dir)[1], "folder": exp_dir}
        path_vars_dict = utils.get_vars_from_path(exp_dir)
        
        for path_key, path_value in path_vars_dict.items():
            if path_key in pt_dict.keys():
                if not (path_value == pt_dict[path_key]):
                    logging.warning(
                        f"{path_key} does not correspond between .tsv and folder, folder value: {path_value}, tsv value: {pt_dict[path_key]}"
                    )
                    logging.warning(f"Saving folders {path_key} as folder_{path_key}")
                    pt_dict["folder_" + path_key] = path_value
            else:
                pt_dict[path_key] = path_value
        session = Session(**pt_dict)
        if parent:
            session.parent = parent
        session.path_vars_keys = path_vars_dict.keys()
        session.load_scans()

        return session

    def load_scans_in_memory(self):
        for scan in self.scans.values():
            scan.load()

    @computed_field
    @property
    def n_scans(self) -> int:
        if not (self.scans):
            return 0
        else:
            return len(self.scans)

    @property
    def all_meta_data(self):
        all_data = self.model_dump(exclude=['parent','scans'])
        if self.fields:
            del all_data["fields"]
            all_data.update(self.fields)
        return all_data


class Subject(BaseModel, extra=Extra.allow):
    participant_id: Optional[Union[str,  None]]
    parent: Any = Field(None, repr=False)
    folder: Optional[Union[Path,  None]] = None
    sessions: Optional[Dict[str, Session]] = Field(default_factory=dict, repr=False)
    scans: Optional[Dict[str, Scan]] = Field(default_factory=dict, repr=False)
    scan_metadata: Optional[Union[dict,  None]] = {}
    fields: Optional[Union[dict,  None]] = None
    path_vars_keys: Optional[List[str]] = []
    load_scans = load_scans

    def from_dict(pt_dict=None, dataset_folder="", parent=None):
        if os.path.isdir(os.path.join(dataset_folder, "./", pt_dict["participant_id"])):
            pt_dict["folder"] = os.path.join(
                dataset_folder, "./", pt_dict["participant_id"]
            )
        subject = Subject(**pt_dict)
        if parent:
            subject.parent = parent
        if pt_dict["folder"]:
            subject.load_sessions()
        subject.load_scans()
        return subject

    def from_folder(pt_dir="", parent=None):
        assert os.path.isdir(pt_dir), "Folder does not exist"
        pt_dict = {"participant_id": os.path.split(pt_dir)[1], "folder": pt_dir}
        path_vars_dict = utils.get_vars_from_path(pt_dir)
        
        for path_key, path_value in path_vars_dict.items():
            if path_key in pt_dict.keys():
                if not (path_value == pt_dict[path_key]):
                    logging.warning(
                        f"{path_key} does not correspond between .tsv and folder, folder value: {path_value}, tsv value: {pt_dict[path_key]}"
                    )
                    logging.warning(f"Saving folders {path_key} as folder_{path_key}")
                    pt_dict["folder_" + path_key] = path_value
            else:
                pt_dict[path_key] = path_value
        subject = Subject(**pt_dict)
        if parent:
            subject.parent = parent
        subject.path_vars_keys = path_vars_dict.keys()
        subject.load_sessions()
        subject.load_scans()
        return subject

    def load_sessions(self):
        assert self.folder, "Subject folder needs to be set to load sessions"
        logging.info(f'Loading sessions for participant {self.participant_id}')
        for exp_folder in tqdm(glob.glob(os.path.join(self.folder, "./ses-*"))):
            if os.path.isdir(exp_folder):
                ses = Session.from_folder(exp_folder)
                self.sessions[ses.session_id] = ses

    def load_scans_in_memory(self):
        for scan in self.scans.values():
            scan.load()
        for ses in self.sessions.values():
            ses.load_scans_in_memory()

    @computed_field
    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @computed_field
    @property
    def n_scans(self) -> int:
        if not (self.scans):
            return 0
        else:
            return len(self.scans)

    @property
    def all_meta_data(self):
        all_data = self.model_dump(exclude=['parent','scans', 'sessions'])
        if self.fields:
            del all_data["fields"]
            all_data.update(self.fields)
        return all_data


class Dataset(BaseModel, extra=Extra.allow):
    name: str
    folder: Path
    json_file: Union[Path, None] = None
    participants_json: Union[Path, None] = None

    bids_version: Union[str, None] = None
    HEDVersion: Union[str, None] = None
    authors: Union[List[str], None] = None
    fields: dict = {}
    description: Union[str, None] = None
    dataset_type: str = ""
    how_to_acknowledge: str = ""
    acknowledgements: str = ""
    funding: Union[List[str], None] = None
    ethics_approvals: Union[List[str], None] = None
    references_and_links: Union[List[str], None] = None
    source_datasets: Union[List[str], List[dict], None] = None
    license: str = ""
    dataset_doi: str = ""

    subjects: Dict[str, Subject] = Field(default_factory=dict, repr=False)
    subject_variables_metadata: Union[List[dict], None] = None


    def from_folder(folder_path, load_scans_in_memory=False):
        dataset_json = os.path.join(folder_path, "./dataset_description.json")
        assert os.path.isfile(
            dataset_json
        ), f"No dataset_description file found at expected location, this is required!: {os.path.join(folder_path, './dataset_description.json')}"
        return Dataset.from_json(dataset_json, load_scans_in_memory)

    def from_json(json_path, load_scans_in_memory=False):
        ds = json.load(open(json_path, "r"))
        ds = utils.dict_camel_to_snake(ds)
        ds["folder"] = os.path.split(json_path)[0]
        ds["json_file"] = json_path
        dataset = Dataset(**ds)
        if os.path.isfile(os.path.join(dataset.folder, "/participants.json")):
            dataset.participants_json = os.path.join(
                dataset.folder, "./participants.json"
            )
            dataset._subject_variables_metadata_from_json()
        dataset.load_subjects()
        if load_scans_in_memory:
            dataset.load_scans_in_memory()
        return dataset

    def load_scans_in_memory(self):
        for subject in self.subjects.values():
            subject.load_scans_in_memory()


    @property
    def all_meta_data(self):
        all_data = self.model_dump(exclude=['subjects'])
        if self.fields:
            del all_data["fields"]
            all_data.update(self.fields)
        return all_data

    def _subject_variables_metadata_from_json(self, json_path=None):
        if not (json_path):
            assert (
                self.participants_json
            ), "Dataset.participants_json or json_path needs to be set to load metadata"
            json_path = os.path.join(self.folder, "./participants.json")

        self.participants_json = json_path
        self.subject_variables_metadata = (
            pd.DataFrame.from_dict(json.load(open(json_path, "r")))
            .transpose()
            .to_dict("records")
        )

    def load_subjects(self):
        if not (os.path.isfile(os.path.join(self.folder, "./participants.tsv"))):
            logging.info(f'No participants.tsv found, loading subjects based on subdirectories')
            for pt_dir in tqdm(glob.glob(os.path.join(self.folder, "./sub-*"))):
                subject = Subject.from_folder(pt_dir=pt_dir, parent=self)
                self.subjects[subject.participant_id] = subject
        else:
            logging.info(f'Loading all subjects from {os.path.join(self.folder, "./participants.tsv")}')
            df = pd.read_csv(os.path.join(self.folder, "./participants.tsv"), sep="\t")
            participant_id_available = "participant_id" in df.columns.tolist()
            assert (
                participant_id_available
            ), f"participant_id missing in {os.path.join(self.folder, './participants.tsv')}"
            for i, pt in tqdm(df.iterrows(), total=len(df)):
                participant_id = pt["participant_id"]
                if not (
                    os.path.isdir(os.path.join(self.folder, f"./{participant_id}"))
                ):
                    logging.warning(
                        f"Missing subject folder for {participant_id}, continuing using only subject variables from {os.path.join(self.folder, './participants.tsv')} for this patient"
                    )
                else:
                    subject = Subject.from_dict(
                        pt_dict=pt.to_dict(),
                        dataset_folder=self.folder,
                        parent=self,
                    )
                    self.subjects[participant_id] = subject
