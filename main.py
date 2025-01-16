import asyncio
import argparse
from resume_analyzer import ResumeAnalyzer
from job_search_crawler import JobSearchCrawler
from form_filling_agent import FormFillingAgent
from models import ResumeData, ApplicationForm


async def main():
    # Initialize components
    analyzer = ResumeAnalyzer(model="gemma2")
    crawler = JobSearchCrawler()
    form_filler = FormFillingAgent()

    # Load and analyze resume
    with open("resume.pdf", "r") as f:
        resume_text = f.read()  # You'll need to implement PDF parsing

    resume_data = await analyzer.extract_resume_data(resume_text)

    # Search for jobs
    keywords = resume_data.skills[:5]  # Use top 5 skills as keywords
    location = "Remote"  # This could be configurable
    job_listings = await crawler.search_jobs(keywords, location)

    # Match jobs with resume
    matched_jobs = await analyzer.match_jobs(resume_data, job_listings)

    # Fill forms for matched jobs
    for job in matched_jobs:
        if job.has_application_form:
            form = ApplicationForm(job=job, fields=job.form_fields)
            success = await form_filler.fill_form(form, resume_data)
            if success:
                print(f"Successfully filled form for {job.title} at {job.company}")


if __name__ == "__main__":
    asyncio.run(main())
