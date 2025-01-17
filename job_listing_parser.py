"""_summary_

Run the script using this:
uv run job_listing_parser.py -i job_listings.txt -m mistral
"""

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
        model: str,
    ):
        self.model = model
        self.client = Client()

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

    def extract_job_description(
        self,
        job_description_text: str,
    ) -> List[JobListing]:
        """Convert raw job descriptions into JobListing objects."""

        # prompt = f"""
        # You are an expert at analyzing and parsing job descriptions to extract structured information in JSON format. Your task is to extract data from the provided job description text strictly according to the following schema:

        # "title": str, // Title of the job description.
        # "company": str, // Company name in the job description.
        # "location": str, // Whether hybrid, remote, or in-office work model.
        # "company_overview": str, // Information about the company.
        # "job_description": str, // Job description.
        # "responsibilities": List[str], // List of key responsibilities mentioned in the job description.
        # "requirements": List[str], // List of minimum requirements mentioned in the job description.
        # "company_offers": List[str], // List of what the company offers (including benefits, work environment, etc., if available).
        # "salary_range": str, // Salary range in the job description (if available).
        # "url": str, // Email address of the individual (if available).
        # "has_application_form": bool, // Email address of the individual (if available).
        # "form_fields": Optional[dict], // Email address of the individual (if available).

        # Extract the data based on this schema as accurately as possible from
        # the job description provided below:
        # Job description text:
        # {job_description_text}

        # Remember:
        # 1. Output ONLY valid JSON adhering to the above schema.
        # 2. Include all fields in the schema, even if they are null or empty arrays when not available in the resume.
        # 3. Extract all relevant information from the provided resume text exactly as written, with no assumptions or added data.
        # 4. Maintain the field order as specified in the schema.
        # 5. Do not include any additional text, commentary, or fields outside of the schema

        # %%%% EXAMPLE %%%%

        # Job Description text: ""
        # About the job
        # Machine Learning Engineer

        # Hybrid- NYC, (with travel up to once a month)

        # $110,000-$170,000

        # Tech Startup

        # Are you an experienced Machine Learning Engineer who is eager to work on the latest advancements in conversational AI and sensor technology? Harnham is partnering with a company that is leading the charge in transforming industries by bridging the gap between digital and physical worlds.

        # THE COMPANY

        # This company is revolutionizing the physical world by using sensor technology to connect AI agents with everyday objects. Their product has broad applications across many product-based industries. With a growing customer base, their products leverage real-time data to enhance business decision-making.

        # THE ROLE

        # As a Machine Learning Engineer, you will be working closely with the Head of AI to design and develop AI-driven virtual assistants and chatbots that interact with the company’s innovative sensor technology. You will be responsible for building multi-agent frameworks, optimizing LLMs, and implementing Retrieval-Augmented Generation systems to support dynamic, intelligent interactions between users and sensor devices. This is a pivotal role where you’ll contribute to deploying AI solutions in real-world applications, enhancing NLP and conversational AI capabilities that connect real-world environments with digital intelligence.

        # YOUR SKILLS AND EXPERIENCE

        # Experience in developing and deploying AI solutions in production environments, especially in cloud-based AI systems (AWS, GCP)
        # Expertise in building and optimizing LLMs
        # Proficient in natural language processing techniques, especially for chatbot and AI-powered agent development
        # Familiarity with multi-agent frameworks and experience implementing Retrieval-Augmented Generation (RAG) systems for enhanced AI-driven solutions.
        # Ability to integrate and manage data from diverse sources for context-aware, real-time AI responses
        # Solid understanding of machine learning engineering, deep learning frameworks (e.g., TensorFlow, PyTorch), and model optimization
        # Experience with edge computing solutions for real-time data processing and AI-driven interactions
        # Strong communication skills to collaborate with cross-functional teams, external stakeholders, and clients

        # HOW TO APPLY

        # Register your interest by sending your Resume to Virginia via the apply link on this page.

        # KEYWORDS

        # Machine Learning | LLM | Chatbot | GenAI | Artificial Intelligence | Multi-Agent Frameworks | Deploying | startup | NLP | RAG | Retrieval-Augmented Generation | Edge Computing | Real-Time AI | Machine Learning Models""

        # RESPONSE: "
        # {
        #     "title": "Machine Learning Engineer",
        #     "company": "Harnham",
        #     "location": "Hybrid - NYC, (with travel up to once a month)",
        #     "company_overview": "This company is revolutionizing the physical world by using sensor technology to connect AI agents with everyday objects. Their product has broad applications across many product-based industries. With a growing customer base, their products leverage real-time data to enhance business decision-making.",
        #     "job_description": "As a Machine Learning Engineer, you will be working closely with the Head of AI to design and develop AI-driven virtual assistants and chatbots that interact with the company’s innovative sensor technology. You will be responsible for building multi-agent frameworks, optimizing LLMs, and implementing Retrieval-Augmented Generation systems to support dynamic, intelligent interactions between users and sensor devices. This is a pivotal role where you’ll contribute to deploying AI solutions in real-world applications, enhancing NLP and conversational AI capabilities that connect real-world environments with digital intelligence.",
        #     "responsibilities": [
        #         "Design and develop AI-driven virtual assistants and chatbots that interact with sensor technology.",
        #         "Build multi-agent frameworks for dynamic, intelligent interactions.",
        #         "Optimize large language models (LLMs).",
        #         "Implement Retrieval-Augmented Generation (RAG) systems.",
        #         "Deploy AI solutions in real-world applications.",
        #         "Enhance NLP and conversational AI capabilities that connect real-world environments with digital intelligence."
        #     ],
        #     "requirements": [
        #         "Experience in developing and deploying AI solutions in production environments, especially in cloud-based AI systems (AWS, GCP).",
        #         "Expertise in building and optimizing LLMs.",
        #         "Proficient in natural language processing techniques, especially for chatbot and AI-powered agent development.",
        #         "Familiarity with multi-agent frameworks and experience implementing Retrieval-Augmented Generation (RAG) systems.",
        #         "Ability to integrate and manage data from diverse sources for context-aware, real-time AI responses.",
        #         "Solid understanding of machine learning engineering, deep learning frameworks (e.g., TensorFlow, PyTorch), and model optimization.",
        #         "Experience with edge computing solutions for real-time data processing and AI-driven interactions.",
        #         "Strong communication skills to collaborate with cross-functional teams, external stakeholders, and clients."
        #     ],
        #     "company_offers": [],
        #     "salary_range": "$110,000-$170,000",
        #     "url": "Virginia via the apply link on this page",
        #     "has_application_form": 'false',
        #     "form_fields": 'null'
        #     }
        # "
        # """

        prompt = f"""
        You are an expert at analyzing and parsing job descriptions to extract structured information in JSON format. Your task is to extract data from the provided job description text strictly according to the following schema:

        "schema": {{
            "title": str, // Title of the job description.
            "company": str, // Company name in the job description.
            "location": str, // Whether hybrid, remote, or in-office work model.
            "company_overview": str, // Information about the company.
            "job_description": str, // Job description.
            "responsibilities": List[str], // List of key responsibilities mentioned in the job description.
            "requirements": List[str], // List of minimum requirements mentioned in the job description.
            "company_offers": List[str], // List of what the company offers (including benefits, work environment, etc., if available).
            "salary_range": str, // Salary range in the job description (if available).
            "url": str, // Email address or application link (if available).
            "has_application_form": bool, // Whether the job requires filling out an application form.
            "form_fields": Optional[dict] // Dictionary of form fields if an application form is present (e.g., {{"name": str, "email": str, "resume": str}}).
        }}

        %%%%%%%%%%%%%%%%%%
        REFERENCE EXAMPLE:
        %%%%%%%%%%%%%%%%%%
        
        Job description text:
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

        Your output should be:
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

        Now, extract the data based on this schema as accurately as possible from the job description provided below:

        Job description text:
        {job_description_text}

        Remember:
        1. Output ONLY valid JSON adhering to the above schema.
        2. Include all fields in the schema, even if they are null or empty arrays when not available in the job description.
        3. Extract all relevant information from the provided job description exactly as written, with no assumptions or added data.
        4. Maintain the field order as specified in the schema.
        5. Do not include any additional text, commentary, or fields outside of the schema.
        """

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
        )

        resume_data = json.loads(response.response)
        return JobListing(**resume_data)


def main(args):

    analyzer = JobListingsParser(model=args.model)

    # Read the text file directly instead of using llama_parser
    with open(args.input_path, "r", encoding="utf-8") as file:
        job_descriptions = file.read().split(
            "\n\n"
        )  # Assuming job listings are separated by blank lines

    for job_id, job_description in enumerate(job_descriptions):
        if not job_description.strip():  # Skip empty descriptions
            continue
        job_description_dict = analyzer.extract_job_description(job_description)
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

    main(args)
