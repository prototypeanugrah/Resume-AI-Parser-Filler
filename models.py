from pydantic import BaseModel, EmailStr
from typing import List, Optional


class ResumeData(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    skills: List[str]
    experience: List[dict]
    education: List[dict]
    achievements: List[str]
    linkedin: str
    github: str
    personal_portfolio: str


class JobListing(BaseModel):
    title: str
    company: str
    location: str
    company_overview: str
    job_description: str
    responsibilities: List[str]
    requirements: List[str]
    company_offers: List[str]
    salary_range: str
    url: str
    has_application_form: bool = False
    form_fields: Optional[dict] = None


class ApplicationForm(BaseModel):
    job: JobListing
    fields: dict
    status: str = "pending"
