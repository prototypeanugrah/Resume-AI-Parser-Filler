import argparse
import asyncio
import nest_asyncio
import os

from dotenv import load_dotenv
from llama_parse import LlamaParse

from form_filling_agent import FormFillingAgent
from job_listing_parser import JobListingParser
from models import ResumeData, ApplicationForm
from resume_analyzer import ResumeAnalyzer

# from job_search_crawler import JobSearchCrawler

load_dotenv()
nest_asyncio.apply()

llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

if not llama_cloud_api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY not found in .env file")


def main(args):

    llama_parser = LlamaParse(
        api_key=llama_cloud_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="text",  # "markdown" and "text" are available
        num_workers=1,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    # Initialize components
    resume_analyzer = ResumeAnalyzer(model=args.model)
    job_listing_parser = JobListingParser(model=args.model)
    form_filler = FormFillingAgent()

    # Load and analyze resume
    resume_text = llama_parser.load_data(args.input_path)
    resume_data = resume_analyzer.extract_resume_data(resume_text)

    # Read and parse job listings
    with open(args.job_listings_path, "r", encoding="utf-8") as file:
        try:
            job_descriptions = eval(file.read())
            if not isinstance(job_descriptions, list):
                raise ValueError("Input file must contain a list of job descriptions")
        except Exception as e:
            print(f"Error reading job descriptions: {e}")
            return

    # # Search for jobs
    # keywords = resume_data.skills  # [:5]  # Use top 5 skills as keywords
    # location = (
    #     resume_data.location if resume_data.location else args.location
    # )  # This could be configurable

    # Parse all job listings
    job_listings = []
    for job_description in job_descriptions:
        if not job_description.strip():  # Skip empty descriptions
            continue

        job_listing = job_listing_parser.extract_job_description(job_description)
        if job_listing:
            # Filter by location preference if specified
            preferred_location = (
                resume_data.location if resume_data.location else args.location
            )
            if preferred_location.lower() in job_listing.location.lower():
                job_listings.append(job_listing)

    # Match jobs with resume
    matched_jobs = resume_analyzer.match_jobs(resume_data, job_listings)

    # Fill forms for matched jobs
    for job in matched_jobs:
        if job.has_application_form:
            form = ApplicationForm(job=job, fields=job.form_fields)
            success = form_filler.fill_form(form, resume_data)
            if success:
                print(f"Successfully filled form for {job.title} at {job.company}")


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
        default="Hybrid",
        choices=["Remote", "Hybrid", "Office"],
        # required=True,
        type=str,
        help="Location preference for the candidate.",
    )

    args = parser.parse_args()

    main(args)
