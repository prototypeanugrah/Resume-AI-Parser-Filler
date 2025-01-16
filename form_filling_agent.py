from models import ApplicationForm, ResumeData
from selenium import webdriver
from selenium.webdriver.common.by import By


class FormFillingAgent:
    def __init__(self):
        self.driver = webdriver.Chrome()  # or use undetected-chromedriver

    async def fill_form(self, form: ApplicationForm, resume_data: ResumeData):
        try:
            self.driver.get(form.job.url)

            for field_name, field_selector in form.fields.items():
                element = self.driver.find_element(By.CSS_SELECTOR, field_selector)
                value = self._get_resume_value(field_name, resume_data)
                if value:
                    element.send_keys(value)

            # Don't submit automatically - let user review
            return True
        except Exception as e:
            print(f"Error filling form: {e}")
            return False

    def _get_resume_value(self, field_name: str, resume_data: ResumeData) -> str:
        # Map form fields to resume data
        field_mapping = {
            "name": resume_data.name,
            "email": resume_data.email,
            # Add more mappings
        }
        return field_mapping.get(field_name, "")
