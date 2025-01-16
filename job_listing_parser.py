from typing import List
import argparse
import json
import asyncio
import nest_asyncio
import os

from dotenv import load_dotenv
from llama_parse import LlamaParse
from models import JobListing
from ollama import Client

load_dotenv()
nest_asyncio.apply()

llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

if not llama_cloud_api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY not found in .env file")


class JobListingsParser:
    def __init__(
        self,
        file_path: str,
        model: str,
    ):
        self.file_path = file_path
        self.client = Client()
        self.model = model

        self.parser_schema = r"""
        Extract the following information from the given job description text and provide the output in JSON format:
        
        {
        "title": 'string', // Title of the job description.
        "company": 'string', // Company name in the job description.
        "location": "string", // Whether hybrid, remote, or in-office work model.
        "company_overview": "string", // Information about the company.
        "job_description": "string", // Job description.
        "responsibilities": ["string", ...], // List of key responsibilities mentioned in the job description.
        "requirements": ["string", ...], // List of minimum requirements mentioned in the job description.
        "company_offers": ["string", ...], // List of what the company offers (including benefits, work environment, if available).
        "salary_range": "string", // Salary range in the job description (if available).
        "url": "string", // Email address of the individual (if available).
        "has_application_form": "bool", // Email address of the individual (if available).
        "form_fields": "Optional[dict]", // Email address of the individual (if available).
        }
        """

    async def parse_job_listings(
        self,
        job_description_text: str,
    ) -> List[JobListing]:
        """Convert raw job descriptions into JobListing objects."""

        prompt = f"""
        Extract the following information from the given resume text and 
        provide the output in JSON format matching this schema:
        {self.resume_schema}

        Extract the data based on this schema as accurately as possible from 
        the resume text provided below:
        Resume text:
        {job_description_text}
        """

        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
        )

        resume_data = json.loads(response.text)
        return JobListing(**resume_data)


async def main(args):

    analyzer = JobListingsParser(model=args.model)

    llama_parser = LlamaParse(
        api_key=llama_cloud_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="text",  # "markdown" and "text" are available
        num_workers=1,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    for job_id, job_description in enumerate(args.input_path):
        job_description = llama_parser.load_data(job_description)
        job_description_dict = await analyzer.extract_job_description(job_description)

        print(f"Job description dict for job id {job_id}: {job_description_dict}")
        break

    # document = llama_parser.load_data(args.input_path)

    # resume_data = await analyzer.extract_resume_data(document)


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
        help="Path to the input job listings file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Name of the model to be used for parsing the resume.",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
