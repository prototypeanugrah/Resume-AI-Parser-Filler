"""_summary_

Run the script using this:
uv run resume_analyzer.py -i ./Anugrah_Resume.pdf -m mistral
"""

from typing import List
import argparse
import json
import asyncio
import nest_asyncio
import os

from dotenv import load_dotenv
from llama_parse import LlamaParse
from models import ResumeData, JobListing
from ollama import Client

load_dotenv()
nest_asyncio.apply()

llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

if not llama_cloud_api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY not found in .env file")


class ResumeAnalyzer:
    def __init__(
        self,
        model: str,
    ):
        self.client = Client()
        self.model = model

        self.resume_schema = r"""
        You are an expert at analyzing and parsing resume and extracting helpful
        information from the provided resume. You are a helpful assistant that 
        extracts information from the resume. 
        
        You ONLY provide the information that is available in the provided 
        resume text and NO new information is generated. The resume text is 
        the ONLY correct information to be used. 
        
        Your response must be ONLY valid JSON with no additional text.
        
        Extract the following information from the given resume text and provide the output in JSON format:
        
        {
        "first_name": 'string', // First name of the individual.
        "last_name": 'string', // Last name of the individual.
        "email": "string", // Email address of the individual.
        "skills": ["string", ...], // List of skills mentioned in the resume. If there is a skills section in the resume, mention all of the mentioned skills. Provide the result as a List ONLY. Do not give a dictionary for this.
        "experience": [ 
            // List of professional experiences. Follow the guidelines mentioned for each field.
            // Note: if the experience includes a university experience like research assistant, researcher role, or similar, then the role is the job_title and company is the university.
            { "job_title": "string", // Job title of the individual (don't include the company name in the job title).
            "company": "string", // Company name.
            "start_date": "string" // Start of employment (e.g., "Jan 2020 - Dec 2023"). Here Jan 2020 is the start date}, ... 
            "end_date": "string" // End of employment (e.g., "Jan 2020 - Dec 2023"). Here Dec 2023 is the end date}, ... 
        ],
        "education": [ 
            // List of educational qualifications.
            { "degree": "string", // Degree earned (like Masters or M.S., Bachelors or B.E., etc. -- do not include the major).
            "major": "string", // Major field name (like Computer Science, Chemical, Physics, etc.)
            "institution": "string", // Name of the institution.
            "start_date": "string" // Start of education (if available). }, ... 
            "end_date": "string" // End of education (if available). }, ... 
        ],
        "achievements": ["string", ...] // List of achievements mentioned (only description) in the resume (if available). Do not use the project descriptions here. This could be achievements like winner of some prize, or mentorship experience, etc. Only give the description here.
        "linkedin": 'string' // LinkedIn url of the individual in the resume (if available)
        "github": 'string' // GitHub url of the individual in the resume (if available)
        "personal_website": 'string' // Personal website / portfolio url of the individual in the resume (if available)
        }
        """

    def extract_resume_data(
        self,
        resume_text: str,
    ) -> ResumeData:
        prompt = f"""
        You are an expert at analyzing and parsing resumes to extract structured information in JSON format. Your task is to extract data from the provided resume text strictly according to the following schema:

        "first_name": str // First name of the individual mentioned in the resume
        "last_name": str // Last name of the individual mentioned in the resume
        "email": EmailStr // Email of the individual mentioned in the resume
        "skills": List[str] // List of skills mentioned in the resume. Include only what is explicitly mentioned. Append all the skills mentioned in a single list.
        "experience": List[dict] // Professional experiences structured as follows:
            - "job_title": str, // The role or position held (e.g., "Software Engineer"). Do not include the company in this field.
            - "company": str, // The name of the organization or institution.
            - "start_date": str, // The start date of the role (e.g., "Jan 2020").
            - "end_date": str, // The end date of the role (e.g., "Dec 2023") or "Present" if ongoing.
        "education": List[dict] // Educational qualifications structured as follows:
            - "degree": str, // The degree earned (e.g., "Bachelor's").
            - "major": str, // The field of study (e.g., "Computer Science").
            - "institution": str, // The name of the institution (e.g., "MIT").
            - "start_date": str, // The start date (e.g., "Aug 2016").
            - "end_date": str, // The end date (e.g., "May 2020") or "Present" if ongoing.
        "achievements": List[str] // List of notable achievements mentioned in the resume (e.g., "Hackathon winner"). Do not include project descriptions here.
        "linkedin": str // LinkedIn profile URL (if available).
        "github": str // GitHub profile URL (if available).
        "personal_portfolio": str // Personal website or portfolio URL (if available).

        Extract the data based on this schema as accurately as possible from the resume text provided below:
        Resume text:
        {resume_text}
        
        Remember:
        1. Output ONLY valid JSON adhering to the above schema.
        2. Include all fields in the schema, even if they are null or empty arrays when not available in the resume.
        3. Extract all relevant information from the provided resume text exactly as written, with no assumptions or added data.
        4. Maintain the field order as specified in the schema.
        5. Do not include any additional text, commentary, or fields outside of the schema
        """

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
        )

        print(f"Model response:\n{response.response}")
        exit(0)

        resume_data = json.loads(response.response)
        return ResumeData(**resume_data)

    async def match_jobs(
        self,
        resume_data: ResumeData,
        job_listings: List[JobListing],
    ) -> List[JobListing]:
        matched_jobs = []

        for job in job_listings:
            prompt = f"""
            Compare this job posting with the candidate's resume and rate match 0-100:
            Job: {job.dict()}
            Resume: {resume_data.dict()}
            """

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
            )

            match_score = float(response.text)
            if match_score > 70:  # threshold for good matches
                matched_jobs.append(job)

        return matched_jobs


def main(args):

    analyzer = ResumeAnalyzer(model=args.model)

    llama_parser = LlamaParse(
        api_key=llama_cloud_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="text",  # "markdown" and "text" are available
        num_workers=1,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    document = llama_parser.load_data(args.input_path)

    resume_data = analyzer.extract_resume_data(document)

    # Search for jobs
    keywords = resume_data.skills[:20]  # Use top 5 skills as keywords
    location = args.location  # This could be configurable

    print(f"Top 5 skills:\n{keywords}")
    print(f"Location preference: {location}")
    # job_listings = await crawler.search_jobs(keywords, location)

    # # Match jobs with resume
    # matched_jobs = await analyzer.match_jobs(resume_data, job_listings)

    # # Fill forms for matched jobs
    # for job in matched_jobs:
    #     if job.has_application_form:
    #         form = ApplicationForm(job=job, fields=job.form_fields)
    #         success = await form_filler.fill_form(form, resume_data)
    #         if success:
    #             print(f"Successfully filled form for {job.title} at {job.company}")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Analyze the resume",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the input resume (PDF).",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Name of the model to be used for parsing the resume.",
    )
    parser.add_argument(
        "-l",
        "--location",
        default="Office",
        choices=["Remote", "Hybrid", "Office"],
        # required=True,
        type=str,
        help="Location preference for the candidate.",
    )

    args = parser.parse_args()

    main(args)
