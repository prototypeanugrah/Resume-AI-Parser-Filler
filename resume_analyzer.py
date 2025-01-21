"""_summary_

Run the script using this:
uv run resume_analyzer.py -p prompts/resume_analyzer.txt -fp prompts/fill_empty_fields_resume_analyzer.txt -m custom_gemma2 -i Anugrah_Resume.pdf -o resume_new -ro raw_resume_new --max_retries 3
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

        # Map common field variations
        self.field_mappings = {
            "name": "first_name",
            "candidate_name": "first_name",
            "github_url": "github",
            "github_profile": "github",
            "linkedin_url": "linkedin",
            "linkedin_profile": "linkedin",
            "personal_website": "personal_portfolio",
            "work_experience": "experience",
            "professional_experience": "experience",
        }

    def normalize_resume_data(self, resume_data: dict) -> dict:
        """Normalize resume data to match expected schema."""
        # Define required fields with their types
        required_fields = {
            "first_name": str,
            "last_name": str,
            "email": str,
            "location": str,
            "skills": str,
            "experience": list,
            "education": list,
            "linkedin": str,
            "github": str,
            "personal_portfolio": str,
        }

        try:
            # Create a new normalized dictionary
            normalized = {}

            # Process each required field
            for field, field_type in required_fields.items():
                # Check if field exists directly
                value = resume_data.get(field)

                # Check mapped fields if value not found
                if value is None:
                    for alt_field, correct_field in self.field_mappings.items():
                        if alt_field in resume_data and correct_field == field:
                            value = resume_data[alt_field]
                            break

                # Set default values if field is still missing
                if value is None:
                    if field_type == list:
                        value = []
                    elif field_type == str:
                        value = ""
                    else:
                        value = None

                normalized[field] = value

            # Remove any fields that aren't in required_fields
            extra_fields = set(resume_data.keys()) - set(required_fields.keys())
            if extra_fields:
                logger.info(f"Removing non-required fields: {extra_fields}")

            return normalized
        except Exception as e:
            logger.error(f"Error in normalize_resume_data: {e}")
            return None

    # def normalize_resume_data(
    #     self,
    #     resume_data: dict,
    # ) -> dict:
    #     """Normalize job data to match expected schema."""
    #     schema_fields = {
    #         "first_name": str,
    #         "last_name": str,
    #         "email": str,
    #         "location": str,
    #         "skills": str,
    #         "experience": List[str],
    #         "education": List[str],
    #         "linkedin": str,
    #         "github": str,
    #         "personal_portfolio": str,
    #     }

    #     # Create a normalized dictionary with all required fields
    #     normalized = {}

    #     # Normalize the data
    #     for schema_field, field_type in schema_fields.items():
    #         # Check if field exists directly
    #         if schema_field in resume_data:
    #             value = resume_data[schema_field]
    #         else:
    #             # Check mapped fields
    #             mapped_value = None
    #             for alt_field, correct_field in self.field_mappings.items():
    #                 if alt_field in resume_data and correct_field == schema_field:
    #                     mapped_value = resume_data[alt_field]
    #                     break
    #             value = mapped_value

    #         # Set default values if field is missing
    #         if value is None:
    #             if field_type == list:
    #                 value = []
    #             elif field_type == bool:
    #                 value = False
    #             elif field_type == str:
    #                 value = ""
    #             else:
    #                 value = None

    #         normalized[schema_field] = value

    #     return normalized

    def validate_name(self, resume_data: dict) -> dict:
        try:
            first_name, last_name = "", ""
            resume_data_keys = resume_data.keys()

            # Check for basic_information first
            if "basic_information" in resume_data_keys:
                basic_info = resume_data["basic_information"]
                if "name" in basic_info:
                    names = basic_info["name"].split()
                    if len(names) >= 2:
                        first_name, last_name = names[0], " ".join(names[1:])
                    else:
                        first_name = names[0]

                    # Copy other basic information to root level
                    for key, value in basic_info.items():
                        if key != "name":
                            resume_data[key] = value

                    # Delete the basic_information container
                    del resume_data["basic_information"]
                else:
                    raise ValueError("No name field found in basic_information")
            else:
                # Fallback to checking direct name fields
                name_keys = [key for key in resume_data_keys if "name" in key.lower()]

                if not name_keys:
                    raise ValueError("No name field found in resume data")

                # Try to find the most specific name field
                if "full_name" in name_keys:
                    names = resume_data["full_name"].split()
                elif "name" in name_keys:
                    names = resume_data["name"].split()
                else:
                    names = resume_data[name_keys[0]].split()

                if len(names) >= 2:
                    first_name = names[0]
                    last_name = " ".join(names[1:])
                else:
                    first_name = names[0]
                    last_name = ""

                # Remove old name fields
                for key in name_keys:
                    if key not in ["first_name", "last_name"]:
                        del resume_data[key]

            # Set the normalized name fields
            resume_data["first_name"] = first_name
            resume_data["last_name"] = last_name

            return resume_data
        except Exception as e:
            logger.error(f"Error in validate_name: {e}")
            return resume_data

    # def validate_name(
    #     self,
    #     resume_data: dict,
    # ) -> dict:
    #     try:
    #         first_name, last_name = "", ""
    #         resume_data_keys = resume_data.keys()
    #         name_key = [
    #             key for key in resume_data_keys if "name" in key or "information" in key
    #         ]
    #         if not name_key:
    #             raise ValueError("No 'name' key found in resume_dict")
    #         if len(name_key) == 1:
    #             name_key = name_key[0]
    #             names = resume_data.get(name_key).split()
    #             if len(names) >= 2:
    #                 first_name, last_name = names[0], " ".join(names[1:])
    #             else:
    #                 first_name, last_name = names[0], ""
    #         else:
    #             raise ValueError("Multiple 'name' key found in resume_dict")

    #         resume_data["first_name"] = first_name
    #         resume_data["last_name"] = last_name

    #         for key in name_keys:
    #             if key not in ["first_name", "last_name"]:
    #                 del resume_data[key]

    #         return resume_data
    #     except ValueError as e:
    #         print(f"Value error in validate_name: {e}")
    #         return None

    def validate_profile_links(
        self,
        resume_data: dict,
    ) -> dict:
        try:
            resume_data_keys = resume_data.keys()
            linkedin_key = [key for key in resume_data_keys if "linkedin" in key]
            github_key = [key for key in resume_data_keys if "github" in key]

            if not linkedin_key:
                resume_data["linkedin"] = ""
            if not github_key:
                resume_data["github"] = ""

            assert (
                len(linkedin_key) == 1
            ), f"Length of linkedin key: {len(linkedin_key)}"
            assert len(github_key) == 1, f"Length of github key: {len(github_key)}"

            linkedin_key = linkedin_key[0]
            github_key = github_key[0]

            resume_data["linkedin"] = resume_data.get(linkedin_key)
            resume_data["github"] = resume_data.get(github_key)

            if linkedin_key != "linkedin":
                del resume_data[linkedin_key]
            if github_key != "github":
                del resume_data[github_key]

            return resume_data
        except Exception as e:
            print(f"Exception: {e}")
            return resume_data

    def validate_education(self, resume_data: dict) -> dict:
        try:
            resume_data_keys = resume_data.keys()
            education_keys = [key for key in resume_data_keys if "education" in key]

            if not education_keys:
                resume_data["education"] = []
                return resume_data

            if len(education_keys) == 1:
                education_key = education_keys[0]

                if isinstance(resume_data.get(education_key), list):
                    resume_data["education"] = self.validate_education_sub(
                        resume_data.get(education_key)
                    )

                    # Delete old key if it's different from 'education'
                    if education_key != "education":
                        del resume_data[education_key]

                    return resume_data
                else:
                    raise ValueError(
                        f"Education section is of type: {type(resume_data.get(education_keys))}"
                    )
        except ValueError as e:
            print(f"Value error in validate_education: {e}")
            return resume_data

    def validate_education_sub(self, education_list: List) -> List:
        valid_fields = [
            "degree",
            "major",
            "university",
            "location",
            "start_date",
            "end_date",
        ]
        try:
            validated_education = []
            for sub_education in education_list:
                # Create new dict with only valid fields
                validated_entry = {}
                for field in valid_fields:
                    validated_entry[field] = sub_education.get(field, "")
                validated_education.append(validated_entry)
            return validated_education
        except Exception as e:
            print(f"Exception: {e}")
            return resume_data

    # def validate_skills(self, resume_data: dict) -> dict:
    #     try:
    #         resume_data_keys = resume_data.keys()
    #         skills_keys = [key for key in resume_data_keys if "skill" in key]

    #         if not skills_keys:
    #             resume_data["skills"] = ""
    #             return resume_data

    #         skills_str = ""
    #         if len(skills_keys) > 1:
    #             # Combine all skills into a single string
    #             all_skills = []
    #             for key in skills_keys:
    #                 if isinstance(resume_data[key], list):
    #                     all_skills.extend(resume_data[key])
    #                 elif isinstance(resume_data[key], str):
    #                     all_skills.append(resume_data[key])
    #             skills_str = ", ".join(all_skills)
    #             # Update resume data with combined skills
    #             resume_data["skills"] = skills_str
    #             # Remove old skill keys
    #             for key in skills_keys:
    #                 if key != "skills":
    #                     del resume_data[key]
    #         return resume_data
    #     except Exception as e:
    #         print(f"Error in validate_skills: {e}")
    #         return None

    def validate_skills(self, resume_data: dict) -> dict:
        try:
            resume_data_keys = resume_data.keys()
            skills_keys = [key for key in resume_data_keys if "skill" in key.lower()]

            if not skills_keys:
                resume_data["skills"] = ""
                return resume_data

            all_skills = []
            for key in skills_keys:
                skills_data = resume_data[key]
                if isinstance(skills_data, dict):
                    # Handle nested skills categories
                    for category_skills in skills_data.values():
                        if isinstance(category_skills, list):
                            all_skills.extend(category_skills)
                        elif isinstance(category_skills, str):
                            all_skills.append(category_skills)
                elif isinstance(skills_data, list):
                    all_skills.extend(skills_data)
                elif isinstance(skills_data, str):
                    all_skills.append(skills_data)

            # Join all skills into a single string
            skills_str = ", ".join(all_skills)

            # Update resume data with combined skills
            resume_data["skills"] = skills_str

            # Remove old skill keys
            for key in skills_keys:
                if key != "skills":
                    del resume_data[key]

            return resume_data
        except Exception as e:
            logger.error(f"Error in validate_skills: {e}")
            return resume_data

    def validate_experience(self, resume_data: dict) -> dict:
        try:
            # Find all experience-related keys
            resume_data_keys = resume_data.keys()
            exp_keys = [key for key in resume_data_keys if "experience" in key]

            if not exp_keys:
                resume_data["experience"] = []
                return resume_data

            # Define valid fields and field mappings
            valid_fields = ["job_title", "company", "start_date", "end_date"]
            field_mappings = {
                "title": "job_title",
                "position": "job_title",
                "employer": "company",
                "organization": "company",
                "institution": "company",
                "from_date": "start_date",
                "to_date": "end_date",
            }

            # Combine all experiences
            all_experiences = []
            for key in exp_keys:
                if isinstance(resume_data[key], list):
                    experiences = resume_data[key]
                    for exp in experiences:
                        normalized_exp = {}

                        # Map and validate fields
                        for field in valid_fields:
                            # Check direct field name
                            value = exp.get(field, "")

                            # Check mapped field names if value is empty
                            if not value:
                                for alt_field, correct_field in field_mappings.items():
                                    if alt_field in exp and correct_field == field:
                                        value = exp[alt_field]
                                        break

                            normalized_exp[field] = value

                        all_experiences.append(normalized_exp)

            # Update resume data with combined experiences
            resume_data["experience"] = all_experiences

            # Remove old experience keys
            for key in exp_keys:
                if key != "experience":
                    del resume_data[key]

            return resume_data

        except Exception as e:
            print(f"Error in validate_experience: {e}")
            return resume_data

    def validate_response_format(self, resume_data) -> dict:
        # Clean the response to ensure we only get the JSON part
        response_text = resume_data.strip()
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1]
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()

        # Parse the JSON
        resume_data = json.loads(response_text)

        return resume_data

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
        response_text = response.response

        try:
            resume_data = self.validate_response_format(response_text)

            with open(self.raw_output_path, "w", encoding="utf-8") as f:
                json.dump(resume_data, f, indent=2)
                logger.info("Saved the raw response at: %s", self.raw_output_path)

            resume_data = self.validate_name(resume_data)
            if resume_data is None:
                raise ValueError("Failed to validate response format")
            else:
                logger.info("Validate name invoked. Passed")

            resume_data = self.validate_experience(resume_data)
            if resume_data is None:
                raise ValueError("Failed to validate response format")
            else:
                logger.info("Validate experience invoked. Passed")

            resume_data = self.validate_education(resume_data)
            if resume_data is None:
                raise ValueError("Failed to validate response format")
            else:
                logger.info("Validate education invoked. Passed")

            resume_data = self.validate_skills(resume_data)
            if resume_data is None:
                raise ValueError("Failed to validate response format")
            else:
                logger.info("Validate skills invoked. Passed")

            resume_data = self.validate_profile_links(resume_data)
            if resume_data is None:
                raise ValueError("Failed to validate response format")
            else:
                logger.info("Validate profile invoked. Passed")

            resume_data = self.normalize_resume_data(resume_data)
                if resume_data is None:
                    raise ValueError("Failed to validate response format")
                else:
                    logger.info("Normalize resume invoked. Passed")

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

            # for field in resume_data.keys():
            #     if field not in required_fields:
            #         del resume_data[field]

            # for field in required_fields:
            #     if field not in resume_data:
            #         raise ValueError(f"Missing required field: {field}")

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
        type=int,
        default=3,
        help="Number of max retries for the model to generate a response.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
