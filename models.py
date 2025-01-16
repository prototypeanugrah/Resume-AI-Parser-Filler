from pydantic import BaseModel
from typing import List, Optional


class ResumeData(BaseModel):
    name: str
    email: str
    skills: List[str]
    experience: List[dict]
    education: List[dict]
    achievements: List[str]


class JobListing(BaseModel):
    title: str
    company: str
    location: str
    description: str
    url: str
    has_application_form: bool = False
    form_fields: Optional[dict] = None


class ApplicationForm(BaseModel):
    job: JobListing
    fields: dict
    status: str = "pending"
