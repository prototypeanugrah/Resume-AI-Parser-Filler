import argparse
import asyncio
import nest_asyncio
import os

from dotenv import load_dotenv
from llama_parse import LlamaParse

from resume_analyzer import ResumeAnalyzer
from job_search_crawler import JobSearchCrawler
from form_filling_agent import FormFillingAgent
from models import ResumeData, ApplicationForm

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
    analyzer = ResumeAnalyzer(model=args.model)
    crawler = JobSearchCrawler()
    form_filler = FormFillingAgent()

    # Load and analyze resume
    resume_text = llama_parser.load_data(args.input_path)

    resume_data = analyzer.extract_resume_data(resume_text)

    # Search for jobs
    keywords = resume_data.skills  # [:5]  # Use top 5 skills as keywords
    location = args.location  # This could be configurable
    job_listings = crawler.search_jobs(keywords, location)

    # Match jobs with resume
    matched_jobs = analyzer.match_jobs(resume_data, job_listings)

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
        default="Office",
        choices=["Remote", "Hybrid", "Office"],
        # required=True,
        type=str,
        help="Location preference for the candidate.",
    )

    args = parser.parse_args()

    main(args)
