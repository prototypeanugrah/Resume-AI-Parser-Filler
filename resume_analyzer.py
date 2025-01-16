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

    async def extract_resume_data(
        self,
        resume_text: str,
    ) -> ResumeData:
        prompt = f"""
        Extract key information from this resume and format as JSON:
        {resume_text}
        Include: name, email, skills, experience, education, achievements
        """

        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
        )

        resume_data = json.loads(response.text)
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

            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
            )

            match_score = float(response.text)
            if match_score > 70:  # threshold for good matches
                matched_jobs.append(job)

        return matched_jobs


async def main(args):

    analyzer = ResumeAnalyzer(model=args.model)

    llama_parser = LlamaParse(
        api_key=llama_cloud_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="text",  # "markdown" and "text" are available
        num_workers=1,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    document = llama_parser.load_data(args.input_path)
    print(f"Document (preview): {document[:10]}")

    # # Load and analyze resume
    # with open(args.input_path, "r", encoding="utf-8") as f:
    #     resume_text = f.read()  # You'll need to implement PDF parsing

    resume_data = await analyzer.extract_resume_data(document)

    # Search for jobs
    keywords = resume_data.skills[:5]  # Use top 5 skills as keywords
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

    asyncio.run(main(args))
