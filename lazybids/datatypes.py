from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connection import Connection

from pkg_resources import resource_filename
from typing import Union, List, Dict, Optional, Any
from pydantic import AnyHttpUrl
from pathlib import Path
import json
import os
import glob
import yaml
import pandas as pd
import SimpleITK as sitk
import numpy as np
import shutil
from urllib.parse import urljoin
import tempfile
from io import StringIO
from pydantic import BaseModel, computed_field, Field, Extra, ConfigDict
from .utils import get_vars_from_path, dict_camel_to_snake, get_basename_extension
from tqdm import tqdm


# Setting up logging
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)


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
    open(
        resource_filename("lazybids", "./resources/itk_supported_extensions.json"), "r"
    )
)


# Function to read image metadata
def read_image_meta_data(fname: Union[Path, str]) -> dict:
    if not os.path.exists(fname):
        raise Exception(f"File not found: {fname}")

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
def read_image(files: List[Union[Path, AnyHttpUrl]]) -> sitk.Image:
    if not files:
        raise ValueError("No files provided for reading image")
    if not isinstance(
        files[0], Path
    ):  # Check if the file is a URL, isinstance(f, AnyHttpUrl) errors..
        raise NotImplementedError(
            "Reading from URL not implemented, run scan.load() or scan.download() first"
        )
    else:
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
            logger.debug(
                f'loading modality folder: {os.path.join(self.folder, "./", modality)}'
            )
            for file in glob.glob(os.path.join(self.folder, "./", modality, "./*")):
                basename, ext = get_basename_extension(file)
                if ext == ".json":
                    if not (basename in self.scans.keys()):
                        self.scans[basename] = Scan().from_json(file)
                    else:
                        self.scans[basename].from_json(file)
                elif ext == ".tsv":
                    if not (basename in self.scans.keys()):
                        self.scans[basename] = Scan().from_tsv(file)
                    else:
                        self.scans[basename].from_tsv(file)
                elif ext.lower() in itk_supported_datatypes:
                    if not (basename in self.scans.keys()):
                        self.scans[basename] = Scan().from_file(file)
                    else:
                        self.scans[basename].from_file(file)
                if basename in self.scans.keys():
                    self.scans[basename].fields["modality"] = modality
                else:
                    logger.warning(f"Scan file is not supported: {file}")


class Scan(BaseModel, extra=Extra.allow):
    """
    Represents a single scan in a BIDS dataset.

    This class handles individual scan data, including metadata, file paths, and image data.

    Attributes:
        name (str): The name of the scan.
        files (List[Union[Path, AnyHttpUrl]]): List of file paths or URLs associated with the scan.
        metadata_files (List[Union[Path, AnyHttpUrl]]): List of metadata file paths or URLs.
        fields (Dict[str, Any]): Dictionary of additional fields associated with the scan.
        table (Optional[pd.DataFrame]): Tabular data associated with the scan, if any.
        participant_id (Optional[str]): ID of the participant this scan belongs to.
        session_id (Optional[str]): ID of the session this scan belongs to.
        data (Optional[sitk.Image]): The image data loaded into memory.
        numpy (Optional[np.ndarray]): The image data loaded into memory as a numpy array.

    Methods:
        from_file: Load scan data from a file.
        from_json: Load scan metadata from a JSON file.
        from_tsv: Load tabular data from a TSV file.
        from_api: Create a Scan object from data fetched from a lazybids-ui server.
        write: Write scan data to files.
        load: Load the image data into memory.
        unload: Unload the image data from memory.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={pd.DataFrame: lambda df: df.to_csv(sep="\t", index=False),
                       pd.core.frame.DataFrame: lambda df: df.to_csv(sep="\t", index=False)},
    )
    name: str = ""
    files: List[Union[Path, AnyHttpUrl]] = Field(default_factory=list)
    metadata_files: List[Union[Path, AnyHttpUrl]] = Field(default_factory=list)
    fields: Dict[str, Any] = Field(default_factory=dict)

    table: Optional[Any|None] = Field(
        
        description="A pandas DataFrame",
        example={"column1": [1, 2, 3], "column2": ["a", "b", "c"]},
        default_factory=pd.DataFrame,
    )
    _loaded: bool = False
    _cached_image: Optional[sitk.Image] = None
    participant_id: Optional[str] = None
    session_id: Optional[str] = None
    connection: Optional["Connection"] = None

    @computed_field(repr=False)
    @property    
    def _table_tsv(self) -> Optional[str]:
        if self.table is not None:
            if not self.table.empty:
                return self.table.to_csv(sep="\t", index=False)
            else:
                return None
        else:
            return None
    
    def from_file(self, file: Union[Path, str, AnyHttpUrl]) -> "Scan":
        if isinstance(file, (Path, str)):
            self.files = [Path(file)]
        else:
            self.files = [file]  # This is a URL

        if not self.name:
            self.name = get_basename_extension(str(file))[0]

        if isinstance(file, (Path, str)):
            try:
                new_fields = read_image_meta_data(file)
                for k, v in new_fields.items():
                    if k in self.fields:
                        if self.fields[k] != v:
                            raise ValueError(f"Field {k} is not the same in both files")
                    else:
                        self.fields[k] = v
            except Exception as e:
                raise Exception(f"Error reading image metadata: {str(e)}")

        return self

    def from_json(self, file: Path) -> "Scan":
        if not (self.name):
            self.name = get_basename_extension(file)[0]
        for k, v in json.load(open(file, "r")).items():
            if k in self.fields.keys():
                assert self.fields[k] == v, f"Field {k} is not the same in both files"
            else:
                self.fields[k] = v
        self.metadata_files.append(file)
        return self

    def from_tsv(self, file: Path) -> "Scan":
        if not (self.name):
            self.name = get_basename_extension(file)[0]
        self.table = pd.read_csv(file, sep="\t")
        self.metadata_files.append(file)
        return self

    def write(self, folder):

        for file in self.files + self.metadata_files:
            new_file = os.path.join(folder, os.path.split(file)[1])
            shutil.copy(file, new_file)
            logger.info("copying {file} to {new_file}")
        if self.table is not None:
            self.table.to_csv(
                os.path.join(folder, self.name + ".tsv"), sep="\t", index=False
            )
        if self.fields:
            with open(os.path.join(folder, self.name + ".json"), "w") as outfile:
                json.dump(self.fields, outfile)

    @property
    def data(self) -> sitk.Image:
        if not self._loaded:
            self.load()
        return self._cached_image

    @property
    def numpy(self) -> np.ndarray:
        self.load()
        return sitk.GetArrayFromImage(self.data)

    def load(self) -> None:
        if not self._loaded:
            try:
                if all(isinstance(f, (Path, str)) for f in self.files):
                    self._cached_image = read_image(self.files)
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        self.download(Path(temp_dir))
                        self._cached_image = read_image(
                            [
                                (
                                    Path(temp_dir) / Path(f.filename).name
                                    if not isinstance(f, Path)
                                    else f
                                )
                                for f in self.files
                            ]
                        )
                self._loaded = True
            except Exception as e:
                raise Exception(f"Error loading image: {str(e)}")

    def unload(self) -> None:
        self._cached_image = None
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
        all_data = self.model_dump(exclude=["data", "numpy"])
        if self.fields:
            del all_data["fields"]
            all_data.update(self.fields)

        return all_data

    @classmethod
    def from_api(
        cls,
        connection: "Connection",
        ds_id: int,
        sub_id: str,
        scan_id: str,
        ses_id: Optional[str] = None,
        scan_data: Optional[Dict] = None,
    ) -> "Scan":
        if ses_id:
            if scan_data is None:
                scan_data = connection.get(
                    f"/api/dataset/{ds_id}/subjects/{sub_id}/sessions/{ses_id}/scans/{scan_id}"
                )
            base_url = f"/api/dataset/{ds_id}/subject/{sub_id}/session/{ses_id}/scan/{scan_id}/files/"
        else:
            if scan_data is None:
                scan_data = connection.get(
                    f"/api/dataset/{ds_id}/subjects/{sub_id}/scans/{scan_id}"
                )
            base_url = f"/api/dataset/{ds_id}/subject/{sub_id}/scan/{scan_id}/files/"
        if "_table_tsv" in scan_data.keys() and not (scan_data["_table_tsv"] is None):
            scan_data["table"] = pd.read_csv(StringIO(scan_data["_table_tsv"]), sep="\t")
        scan = cls(**scan_data)
        scan.connection = connection
        scan.files = [
            AnyHttpUrl(url=urljoin(connection.base_url, f"{base_url}{Path(f).name}"))
            for f in scan.files
        ]
        scan.metadata_files = [
            AnyHttpUrl(url=urljoin(connection.base_url, f"{base_url}{Path(f).name}"))
            for f in scan.metadata_files
        ]

        return scan

    def download(self, destination: Path):
        new_files = []
        new_metadata_files = []
        for file_list in [self.files, self.metadata_files]:
            for file in file_list:
                if not isinstance(
                    file, Path
                ):  # Check if the file is a URL, isinstance(f, AnyHttpUrl) errors..
                    response = self.connection.session.get(file)
                    response.raise_for_status()
                    filename = Path(file.path).name
                    with open(destination / filename, "wb") as f:
                        f.write(response.content)
                    if file in self.files:
                        new_files.append(destination / filename)
                    if file in self.metadata_files:
                        new_metadata_files.append(destination / filename)
                else:
                    if file in self.files:
                        new_files.append(file)
                    if file in self.metadata_files:
                        new_metadata_files.append(file)

        self.files = new_files
        self.metadata_files = new_metadata_files


class Session(BaseModel):
    """
    Represents a session in a BIDS dataset.

    This class handles session-level data, including scans and metadata.

    Attributes:
        folder (Optional[Union[Path, None]]): Path to the session folder.
        scans (Optional[Dict[str, Scan]]): Dictionary of scans in this session.
        scan_metadata (Optional[Union[Dict[Any, Any], None]]): Additional metadata for scans.
        fields (Optional[Union[Dict[Any, Any], None]]): Additional fields for the session.
        session_id (Optional[str]): ID of the session.
        participant_id (Optional[str]): ID of the participant this session belongs to.

    Methods:
        from_dict: Create a Session object from a dictionary.
        from_folder: Create a Session object from a folder path.
        load_scans_in_memory: Load all scans in this session into memory.
        from_api: Create a Session object from data fetched from a lazybids-ui server.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    folder: Optional[Union[Path, None]] = None
    scans: Optional[Dict[str, Scan]] = Field(default_factory=dict, repr=False)
    scan_metadata: Optional[Union[Dict[Any, Any], None]] = Field(
        default_factory=dict, repr=False
    )
    fields: Optional[Union[Dict[Any, Any], None]] = None
    load_scans = load_scans
    path_vars_keys: Optional[List[str]] = Field(default_factory=list)
    session_id: Optional[str] = None
    participant_id: Optional[str] = None
    connection: Optional["Connection"] = None

    def from_dict(exp_dict=None, dataset_folder="") -> "Session":
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
        if exp_dict["folder"]:
            ses.load_scans()
        return ses

    @classmethod
    def from_folder(cls, exp_dir: str = "") -> "Session":
        assert os.path.isdir(exp_dir), "Folder does not exist"
        pt_dict = {"participant_id": os.path.split(exp_dir)[1], "folder": exp_dir}
        path_vars_dict = get_vars_from_path(exp_dir)

        for path_key, path_value in path_vars_dict.items():
            if path_key in pt_dict.keys():
                if not (path_value == pt_dict[path_key]):
                    logger.warning(
                        f"{path_key} does not correspond between .tsv and folder, folder value: {path_value}, tsv value: {pt_dict[path_key]}"
                    )
                    logger.warning(f"Saving folders {path_key} as folder_{path_key}")
                    pt_dict["folder_" + path_key] = path_value
            else:
                pt_dict[path_key] = path_value
        session = cls(**pt_dict)
        session.path_vars_keys = list(path_vars_dict.keys())
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
        all_data = self.model_dump(exclude=["scans"])
        if self.fields:
            del all_data["fields"]
            all_data.update(self.fields)
        return all_data

    @classmethod
    def from_api(
        cls, connection: "Connection", ds_id: int, sub_id: str, ses_id: str
    ) -> "Session":
        session_data = connection.get(
            f"/api/dataset/{ds_id}/subjects/{sub_id}/sessions/{ses_id}"
        )
        scans_data = connection.get(
            f"/api/dataset/{ds_id}/subjects/{sub_id}/sessions/{ses_id}/scans"
        )

        session = cls(**session_data)
        session.connection = connection
        for scan in scans_data.values():
            session.scans[scan["name"]] = Scan.from_api(
                connection,
                ds_id,
                sub_id,
                ses_id=ses_id,
                scan_id=scan["name"],
                scan_data=scan,
            )
        return session


class Subject(BaseModel):
    """
    Represents a subject in a BIDS dataset.

    This class handles subject-level data, including sessions, scans, and metadata.

    Attributes:
        participant_id (Optional[Union[str, None]]): ID of the subject.
        folder (Optional[Union[Path, None]]): Path to the subject folder.
        sessions (Optional[Dict[str, Session]]): Dictionary of sessions for this subject.
        scans (Optional[Dict[str, Scan]]): Dictionary of scans directly associated with this subject.
        scan_metadata (Optional[Union[Dict[Any, Any], None]]): Additional metadata for scans.
        fields (Optional[Union[Dict[Any, Any], None]]): Additional fields for the subject.

    Methods:
        from_dict: Create a Subject object from a dictionary.
        from_folder: Create a Subject object from a folder path.
        load_sessions: Load all sessions for this subject.
        load_scans_in_memory: Load all scans for this subject into memory.
        from_api: Create a Subject object from data fetched from a lazybids-ui server.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    participant_id: Optional[Union[str, None]]
    folder: Optional[Union[Path, None]] = None
    sessions: Optional[Dict[str, Session]] = Field(default_factory=dict, repr=False)
    scans: Optional[Dict[str, Scan]] = Field(default_factory=dict, repr=False)
    scan_metadata: Optional[Union[Dict[Any, Any], None]] = Field(
        default_factory=dict, repr=False
    )
    fields: Optional[Union[Dict[Any, Any], None]] = None
    path_vars_keys: Optional[List[str]] = Field(default_factory=list)
    load_scans = load_scans
    connection: Optional["Connection"] = None

    def from_dict(pt_dict=None, dataset_folder="") -> "Subject":
        if os.path.isdir(os.path.join(dataset_folder, "./", pt_dict["participant_id"])):
            pt_dict["folder"] = os.path.join(
                dataset_folder, "./", pt_dict["participant_id"]
            )
        subject = Subject(**pt_dict)
        if pt_dict["folder"]:
            subject.load_sessions()
        subject.load_scans()
        return subject

    @classmethod
    def from_folder(cls, pt_dir: str = "") -> "Subject":
        assert os.path.isdir(pt_dir), "Folder does not exist"
        pt_dict = {"participant_id": os.path.split(pt_dir)[1], "folder": pt_dir}
        path_vars_dict = get_vars_from_path(pt_dir)

        for path_key, path_value in path_vars_dict.items():
            if path_key in pt_dict.keys():
                if not (path_value == pt_dict[path_key]):
                    logger.warning(
                        f"{path_key} does not correspond between .tsv and folder, folder value: {path_value}, tsv value: {pt_dict[path_key]}"
                    )
                    logger.warning(f"Saving folders {path_key} as folder_{path_key}")
                    pt_dict["folder_" + path_key] = path_value
            else:
                pt_dict[path_key] = path_value
        subject = cls(**pt_dict)
        subject.path_vars_keys = list(path_vars_dict.keys())
        subject.load_sessions()
        subject.load_scans()
        return subject

    def load_sessions(self):
        assert self.folder, "Subject folder needs to be set to load sessions"
        logger.info(f"Loading sessions for participant {self.participant_id}")
        session_folders = glob.glob(os.path.join(self.folder, "./ses-*"))
        for exp_folder in tqdm(
            session_folders, desc="Loading sessions", unit="session"
        ):
            if os.path.isdir(exp_folder):
                ses = Session.from_folder(exp_folder)
                self.sessions[ses.session_id] = ses
                logger.debug(f"Loaded session: {ses.session_id}")

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
        all_data = self.model_dump(exclude=["scans", "sessions"])
        if self.fields:
            del all_data["fields"]
            all_data.update(self.fields)
        return all_data

    @classmethod
    def from_api(cls, connection: "Connection", ds_id: int, sub_id: str) -> "Subject":
        subject_data = connection.get(f"/api/dataset/{ds_id}/subjects/{sub_id}")
        sessions_data = connection.get(
            f"/api/dataset/{ds_id}/subjects/{sub_id}/sessions"
        )
        scans_data = connection.get(f"/api/dataset/{ds_id}/subjects/{sub_id}/scans")

        subject = cls(**subject_data)
        subject.connection = connection
        subject.sessions = {
            session["session_id"]: Session.from_api(
                connection, ds_id, sub_id, session["session_id"]
            )
            for session in sessions_data.values()
        }
        for scan in scans_data.values():
            subject.scans[scan["name"]] = Scan.from_api(
                connection, ds_id, sub_id, scan_id=scan["name"], scan_data=scan
            )
        return subject


class Dataset(BaseModel):
    """
    Represents a BIDS dataset.

    This class handles dataset-level information, including subjects, metadata, and dataset description.

    Attributes:
        name (str): Name of the dataset.
        folder (Path): Path to the dataset folder.
        json_file (Union[Path, None]): Path to the dataset description JSON file.
        participants_json (Union[Path, None]): Path to the participants JSON file.
        bids_version (Union[str, None]): BIDS version of the dataset.
        subjects (Dict[str, Subject]): Dictionary of subjects in the dataset.
        subject_variables_metadata (Union[List[Dict[Any, Any]], None]): Metadata for subject variables.

    Methods:
        from_folder: Create a Dataset object from a folder path.
        from_json: Create a Dataset object from a JSON file.
        load_scans_in_memory: Load all scans in the dataset into memory.
        load_subjects: Load all subjects in the dataset.
        from_api: Create a Dataset object from data fetched from a lazybids-ui server.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    folder: Path
    json_file: Union[Path, None] = None
    participants_json: Union[Path, None] = None

    bids_version: Union[str, None] = None
    HEDVersion: Union[str, None] = None
    authors: Union[List[str], None] = None
    fields: dict = Field(default_factory=dict)
    description: Union[str, None] = None
    dataset_type: str = ""
    how_to_acknowledge: str = ""
    acknowledgements: str = ""
    funding: Union[List[str], None] = None
    ethics_approvals: Union[List[str], None] = None
    references_and_links: Union[List[str], None] = None
    source_datasets: Union[List[str], List[Dict], None] = None
    license: str = ""
    dataset_doi: str = ""

    subjects: Dict[str, Subject] = Field(default_factory=dict, repr=False)
    subject_variables_metadata: Union[List[Dict[Any, Any]], None] = None

    connection: Optional["Connection"] = None

    @classmethod
    def from_folder(cls, folder_path, load_scans_in_memory=False) -> "Dataset":
        dataset_json = os.path.join(folder_path, "./dataset_description.json")
        assert os.path.isfile(
            dataset_json
        ), f"No dataset_description file found at expected location, this is required!: {os.path.join(folder_path, './dataset_description.json')}"
        return cls.from_json(dataset_json, load_scans_in_memory)

    @classmethod
    def from_json(cls, json_path, load_scans_in_memory=False) -> "Dataset":
        ds = json.load(open(json_path, "r"))
        ds = dict_camel_to_snake(ds)
        ds["folder"] = os.path.split(json_path)[0]
        ds["json_file"] = json_path
        dataset = cls(**ds)
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
        all_data = self.model_dump(exclude=["subjects"])
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
            logger.info(
                f"No participants.tsv found, loading subjects based on subdirectories"
            )
            subject_folders = glob.glob(os.path.join(self.folder, "./sub-*"))
            for pt_dir in tqdm(
                subject_folders, desc="Loading subjects", unit="subject"
            ):
                subject = Subject.from_folder(pt_dir=pt_dir)
                self.subjects[subject.participant_id] = subject
        else:
            logger.info(
                f'Loading all subjects from {os.path.join(self.folder, "./participants.tsv")}'
            )
            df = pd.read_csv(os.path.join(self.folder, "./participants.tsv"), sep="\t")
            participant_id_available = "participant_id" in df.columns.tolist()
            assert (
                participant_id_available
            ), f"participant_id missing in {os.path.join(self.folder, './participants.tsv')}"
            for i, pt in tqdm(
                df.iterrows(), total=len(df), desc="Loading subjects", unit="subject"
            ):
                participant_id = pt["participant_id"]
                if not (
                    os.path.isdir(os.path.join(self.folder, f"./{participant_id}"))
                ):
                    logger.warning(
                        f"Missing subject folder for {participant_id}, continuing using only subject variables from \
                            {os.path.join(self.folder, './participants.tsv')} for this patient"
                    )
                else:
                    subject = Subject.from_dict(
                        pt_dict=pt.to_dict(), dataset_folder=self.folder
                    )
                    self.subjects[participant_id] = subject

    @classmethod
    def from_api(cls, connection: "Connection", ds_id: int) -> "Dataset":
        dataset_data = connection.get(f"/api/dataset/{ds_id}")
        subjects_data = connection.get(f"/api/dataset/{ds_id}/subjects")

        dataset = cls(**dataset_data)
        dataset.connection = connection
        dataset.subjects = {
            subject["participant_id"]: Subject.from_api(
                connection, ds_id, subject["participant_id"]
            )
            for subject in subjects_data.values()
        }

        return dataset


from .connection import Connection

Dataset.model_rebuild()
Subject.model_rebuild()
Session.model_rebuild()
Scan.model_rebuild()






