from typing import Dict, List, Optional
from urllib.parse import urljoin
import requests
from pydantic import BaseModel, Field, ConfigDict
from .datatypes import Dataset, Subject, Session, Scan

class Connection(BaseModel):
    base_url: str
    token: Optional[str] = None
    session: requests.Session = Field(default_factory=requests.Session)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, base_url: str, token: Optional[str] = None, **data):
        super().__init__(base_url=base_url, token=token, **data)
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def get(self, endpoint: str) -> Dict:
        url = urljoin(self.base_url, endpoint)
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def list_datasets(self) -> List[Dict]:
        return self.get("/api/datasets")

    def get_dataset(self, ds_id: Optional[int] = None, ds_name: Optional[str] = None) -> Dataset:
        if ds_id:
            return Dataset.from_api(self, ds_id)
        elif ds_name:
            datasets = self.list_datasets()
            ds_id = next((ds['id'] for ds in datasets if ds['name'] == ds_name), None)
            if ds_id is None:
                raise ValueError(f"Dataset with name {ds_name} not found")
            return Dataset.from_api(self, ds_id)
        else:
            raise ValueError("Either ds_id or ds_name must be provided")

    def get_subject(self, ds_id: int, sub_id: str) -> Subject:
        return Subject.from_api(self, ds_id, sub_id)

    def get_session(self, ds_id: int, sub_id: str, ses_id: str) -> Session:
        return Session.from_api(self, ds_id, sub_id, ses_id)

    def get_scan(self, ds_id: int, sub_id: str, ses_id: str, scan_id: str) -> Scan:
        return Scan.from_api(self, ds_id, sub_id, ses_id, scan_id)
    
    @property
    def dataset(self, ds_id: int) -> Dataset:
        return Dataset.from_api(self, ds_id)
