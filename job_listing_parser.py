"""_summary_

Run the script using this:
uv run job_listing_parser.py -i job_listings.txt -m mistral
"""

from typing import List
import argparse
import json
import os

from models import JobListing
from ollama import Client
from pprint import pprint
from tqdm import tqdm


class JobListingsParser:
    def __init__(
        self,
        model: str,
    ):
        self.model = model
        self.client = Client()

    def normalize_job_data(
        self,
        job_data: dict,
    ) -> dict:
        """Normalize job data to match expected schema."""
        schema_fields = {
            "title": str,
            "company": str,
            "location": str,
            "company_overview": str,
            "job_description": str,
            "responsibilities": list,
            "requirements": list,
            "company_offers": list,
            "salary_range": str,
            "url": str,
            "has_application_form": bool,
            "form_fields": type(None),
        }

        # Create a normalized dictionary with all required fields
        normalized = {}

        # Map common field variations
        field_mappings = {
            "job_title": "title",
            "description": "job_description",
            "salary": "salary_range",
            "benefits": "company_offers",
            "apply_instructions": "url",
        }

        # Normalize the data
        for schema_field, field_type in schema_fields.items():
            # Check if field exists directly
            if schema_field in job_data:
                value = job_data[schema_field]
            else:
                # Check mapped fields
                mapped_value = None
                for alt_field, correct_field in field_mappings.items():
                    if alt_field in job_data and correct_field == schema_field:
                        mapped_value = job_data[alt_field]
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

    def extract_job_description(
        self,
        job_description_text: str,
    ) -> List[JobListing]:
        """Convert raw job descriptions into JobListing objects."""

        prompt = f"""
        You are an expert at analyzing and parsing job descriptions to extract structured information in JSON format. Your task is to extract data from the provided job description text strictly according to the following schema:

        "schema": {{
            "title": str, // Title of the job description.
            "company": str, // Company name in the job description.
            "location": str, // Whether Hybrid, Remote, or In-Office work model.
            "company_overview": str, // Information about the company.
            "job_description": str, // Job description.
            "responsibilities": List[str], // List of key responsibilities mentioned in the job description.
            "requirements": List[str], // List of minimum requirements. IMPORTANT: If requirements are in paragraph form, split them into separate items. Each requirement should be a complete, standalone statement.
            "company_offers": List[str], // List of what the company offers (if available). What des the company value, opportunities for employess, etc.
            "salary_range": str, // Salary range in the job description (if available).
            "url": str, // Email address or application link (if available).
            "has_application_form": bool, // Whether the job requires filling out an application form.
            "form_fields": Optional[dict] // Dictionary of form fields if an application form is present (e.g., {{"name": str, "email": str, "resume": str}}).
        }}
        
        CRITICAL REQUIREMENTS:
        1. Output ONLY valid JSON adhering to the above schema.
        2. Include all fields in the schema, even if they are null or empty arrays when not available in the job description.
        3. Extract all relevant information from the provided job description exactly as written, with no assumptions or added data.
        4. Maintain the field order as specified in the schema.
        5. Do not include any additional text, commentary, or fields outside of the schema.
        6. Follow EXACTLY the same field names as shown above
        7. DO NOT change or rename ANY field names
        8. DO NOT add new fields
        9. DO NOT omit any fields
        10. ALL fields must be present even if null

        %%%%%%%%%%%%%%%%%%
        REFERENCE EXAMPLES:
        %%%%%%%%%%%%%%%%%%
        
        Example 1 -  Job description text:
        \"\"\"
        About the job
        Software Engineer 

        Remote 

        Tech Startup

        Are you an experienced software engineer who is eager to work on cutting-edge technologies? Join our growing team and help us build scalable, cloud-based solutions.

        Responsibilities:
        - Design and implement scalable systems.
        - Collaborate with cross-functional teams.

        Requirements:
        - Proficient in Python and cloud technologies.
        - Strong problem-solving skills.

        Salary: $80,000 - $120,000

        Apply at jobs@example.com.
        \"\"\"

        Example 1 output for the given job description is:
        {{
        "title": "Software Engineer",
        "company": "Tech Startup",
        "location": "Remote",
        "company_overview": "Join our growing team and help us build scalable, cloud-based solutions.",
        "job_description": "Are you an experienced software engineer who is eager to work on cutting-edge technologies?",
        "responsibilities": [
            "Design and implement scalable systems.",
            "Collaborate with cross-functional teams."
        ],
        "requirements": [
            "Proficient in Python and cloud technologies.",
            "Strong problem-solving skills."
        ],
        "company_offers": [],
        "salary_range": "$80,000 - $120,000",
        "url": "jobs@example.com",
        "has_application_form": false,
        "form_fields": null
        }}
        
        ------------------------------------
        
        Example 2 -  Job description text:
        \"\"\"
        About the job
        Software Engineer 

        Remote 

        Tech Startup

        Are you an experienced software engineer who is eager to work on cutting-edge technologies? Join our growing team and help us build scalable, cloud-based solutions.

        Responsibilities:
        - Design and implement scalable systems.
        - Collaborate with cross-functional teams.

        Requirements: Proficient in Python and cloud technologies. Strong problem-solving skills.

        Salary: $80,000 - $120,000

        Apply at jobs@example.com.
        \"\"\"

        Example 2 output for the given job description is:
        {{
        "title": "Software Engineer",
        "company": "Tech Startup",
        "location": "Remote",
        "company_overview": "Join our growing team and help us build scalable, cloud-based solutions.",
        "job_description": "Are you an experienced software engineer who is eager to work on cutting-edge technologies?",
        "responsibilities": [
            "Design and implement scalable systems.",
            "Collaborate with cross-functional teams."
        ],
        "requirements": [
            "Proficient in Python and cloud technologies.",
            "Strong problem-solving skills."
        ],
        "company_offers": [],
        "salary_range": "$80,000 - $120,000",
        "url": "jobs@example.com",
        "has_application_form": false,
        "form_fields": null
        }}
        
        ------------------------------------

        Now, extract the data based on this schema as accurately as possible from the job description provided below:
        {job_description_text}

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
            job_data = json.loads(response_text)
            job_data = self.normalize_job_data(job_data)

            # Validate the required fields are present
            required_fields = [
                "title",
                "company",
                "location",
                "company_overview",
                "job_description",
                "responsibilities",
                "requirements",
                "company_offers",
                "salary_range",
                "url",
                "has_application_form",
                "form_fields",
            ]

            for field in required_fields:
                if field not in job_data:
                    raise ValueError(f"Missing required field: {field}")

            return JobListing(**job_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            # pprint(f"{response_text}")
            return None
        except ValueError as e:
            print(f"Validation error: {e}")
            # pprint(job_data)
            return None


def main(args):

    analyzer = JobListingsParser(
        model=args.model,
    )

    # Read the text file and parse as Python list
    with open(args.input_path, "r", encoding="utf-8") as file:
        try:
            job_descriptions = eval(file.read())
            if not isinstance(job_descriptions, list):
                raise ValueError("Input file must contain a list of job descriptions")
        except Exception as e:
            print(f"Error reading job descriptions: {e}")
            return []

    parsed_jobs = []
    for job_id, job_description in tqdm(
        enumerate(job_descriptions, 1),
        desc="Processing Job",
        total=len(job_descriptions),
    ):
        if not job_description.strip():  # Skip empty descriptions
            continue

        job_description_dict = analyzer.extract_job_description(job_description)
        if job_description_dict:
            parsed_jobs.append(job_description_dict)

    print(
        f"\nSuccessfully parsed {len(parsed_jobs)} out of {len(job_descriptions)} jobs"
    )

    with open(args.output_path + ".json", "w", encoding="utf-8") as f:
        json.dump(parsed_jobs, f, indent=2)

    return parsed_jobs

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
        "-o",
        "--output_path",
        type=str,
        # required=True,
        help="Path to the output job listings file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Name of the model to be used for parsing the resume.",
    )

    args = parser.parse_args()

    main(args)
