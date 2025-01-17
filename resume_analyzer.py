"""_summary_

Run the script using this:
uv run resume_analyzer.py -p prompts/resume_analyzer.txt -fp prompts/fill_empty_fields_resume_analyzer.txt -m gemma2:27b -i Anugrah_Resume.pdf -o resume_new -ro raw_resume_new --max_retries 3
"""

from typing import List
import argparse

# import asyncio
import json
import logging
import nest_asyncio
import os
import time

from dotenv import load_dotenv
from llama_parse import LlamaParse
from models import ResumeData, JobListing
from ollama import Client
from tqdm import tqdm

load_dotenv()
nest_asyncio.apply()

llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

if not llama_cloud_api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY not found in .env file")

# Create and configure logger
logging.basicConfig(
    filename="newfile.log",
    format="%(asctime)s %(message)s",
    filemode="w",
)

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


class ResumeAnalyzer:

    def __init__(
        self,
        model: str,
        raw_output_path: str,
        prompt_template: str,
        fill_prompt_template: str,
    ):
        self.client = Client()
        self.model = model
        self.raw_output_path = raw_output_path
        self.prompt_template = prompt_template
        self.fill_prompt_template = fill_prompt_template

    def normalize_resume_data(
        self,
        resume_data: dict,
    ) -> dict:
        """Normalize job data to match expected schema."""
        schema_fields = {
            "first_name": str,
            "last_name": str,
            "email": str,
            "location": str,
            "skills": str,
            "experience": List[str],
            "education": List[str],
            # "achievements": List[str],
            "linkedin": str,
            "github": str,
            "personal_portfolio": str,
        }

        # Create a normalized dictionary with all required fields
        normalized = {}

        # Map common field variations
        field_mappings = {
            "name": "first_name",
            "github_url": "github",
            "linkedin_url": "linkedin",
            "personal_website": "personal_portfolio",
            "work_experience": "experience",
        }

        # Normalize the data
        for schema_field, field_type in schema_fields.items():
            # Check if field exists directly
            if schema_field in resume_data:
                value = resume_data[schema_field]
            else:
                # Check mapped fields
                mapped_value = None
                for alt_field, correct_field in field_mappings.items():
                    if alt_field in resume_data and correct_field == schema_field:
                        mapped_value = resume_data[alt_field]
                        break
                value = mapped_value

            # Set default values if field is missing
            if value is None:
                if field_type == list:
                    value = []
                elif field_type == bool:
                    value = False
                elif field_type == str:
                    value = ""
                else:
                    value = None

            normalized[schema_field] = value

        return normalized

    def extract_resume_data(
        self,
        resume_text: str,
    ) -> ResumeData:
        prompt = f"""
        {self.prompt_template}
        
        Extract the data based on this schema as accurately as possible.
        You will be provided the extracted resume text from a candidate resume.
         from the resume text provided below:
        Resume text:
        {resume_text}
        """

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
        )

        try:
            # Clean the response to ensure we only get the JSON part
            response_text = response.response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            response_text = response_text.strip()

            # Parse the JSON
            resume_data = json.loads(response_text)

            with open(self.raw_output_path, "w", encoding="utf-8") as f:
                json.dump(resume_data, f, indent=2)
                logger.info("Saved the raw response at: %s", self.raw_output_path)

            resume_data = self.normalize_resume_data(resume_data)

            # Validate the required fields are present
            required_fields = [
                "first_name",
                "last_name",
                "email",
                "location",
                "skills",
                "experience",
                "education",
                # "achievements",
                "linkedin",
                "github",
                "personal_portfolio",
            ]

            for field in required_fields:
                if field not in resume_data:
                    raise ValueError(f"Missing required field: {field}")

            return ResumeData(**resume_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return None
        except ValueError as e:
            print(f"Validation error: {e}")
            return None

    def check_empty_fields(self, resume_data: dict) -> List[str]:
        """Check which first-level fields are empty."""
        empty_fields = []
        for key, value in resume_data.items():
            # Skip nested structures like lists
            if isinstance(value, (list, dict)):
                continue
            # Check if the value is empty string
            if isinstance(value, str) and not value.strip():
                empty_fields.append(key)
        return empty_fields

    def fill_empty_fields(
        self,
        resume_text: str,
        empty_fields: List[str],
    ) -> dict:
        """Fill empty fields using a specific prompt."""
        prompt = f"""
        {self.fill_prompt_template}
        
        The following fields are empty in the resume: {', '.join(empty_fields)}
        Please extract only these specific fields from the resume text:
        
        Resume text:
        {resume_text}
        """

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
        )

        try:
            response_text = response.response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            response_text = response_text.strip()

            filled_fields = json.loads(response_text)
            return filled_fields
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for filling fields: {e}")
            return {}

    async def match_jobs(
        self,
        resume_data: ResumeData,
        job_listings: List[JobListing],
    ) -> List[JobListing]:
        matched_jobs = []

        for job in job_listings:
            prompt = f"""
            Compare this job posting with the candidate's resume and rate match 0-100:
            Job: {job.model_dump()}
            Resume: {resume_data.model_dump()}
            """

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
            )

            match_score = float(response.response)
            if match_score > 70:  # threshold for good matches
                matched_jobs.append(job)

        return matched_jobs


def main(args):

    main_prompt_template = os.path.join(
        args.prompt_template,
        "resume_analyzer_main.txt",
    )
    fill_empty_prompt_template = os.path.join(
        args.prompt_template,
        "resume_analyzer_fill_empty.txt",
    )
    raw_output_path = os.path.join(args.output_dir, "raw_response.json")

    analyzer = ResumeAnalyzer(
        prompt_template=main_prompt_template,
        model=args.model,
        raw_output_path=raw_output_path,
        fill_prompt_template=fill_empty_prompt_template,
    )

    llama_parser = LlamaParse(
        api_key=llama_cloud_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="text",  # "markdown" and "text" are available
        num_workers=1,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

    document = llama_parser.load_data(args.input_path)
    retry_count = 0
    resume_data = None

    with open(
        os.path.join(args.output_dir, "raw_text.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for item in document:
            f.write(str(item) + "\n")

    # exit(0)

    for retry_count in tqdm(range(args.max_retries), desc="Parsing resume"):
        if resume_data is not None:
            break

        resume_data = analyzer.extract_resume_data(document)
        if resume_data is None:
            logger.warning(
                f"Attempt {retry_count + 1} failed to parse resume. Retrying..."
            )
            time.sleep(1)

    if resume_data is None:
        logger.error("Failed to parse resume after maximum retries")
        return

    resume_dump = resume_data.model_dump()

    # Check for empty fields
    empty_fields = analyzer.check_empty_fields(resume_dump)
    if empty_fields:
        logger.info(f"Found empty fields: {empty_fields}")

        # Read the fill prompt template
        with open(fill_empty_prompt_template, "r", encoding="utf-8") as f:
            fill_prompt_template = f.read()

        # Try to fill empty fields
        filled_fields = analyzer.fill_empty_fields(
            document,
            empty_fields,
        )

        # Update resume_dump with filled fields
        resume_dump.update(filled_fields)

    with open(
        os.path.join(args.output_dir, "final_response.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(resume_dump, f, indent=2)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Analyze the resume",
    )
    parser.add_argument(
        "-p",
        "--prompt_template",
        type=str,
        required=True,
        help="Path to the prompt template.",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the input resume (PDF).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to the output json.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Name of the model to be used for parsing the resume.",
    )
    parser.add_argument(
        "--max_retries",
        required=True,
        type=int,
        help="Number of max retries for the model to generate a response.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
