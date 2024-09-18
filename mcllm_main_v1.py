import os
import re
import logging
import csv
import yaml
import shutil
import time
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_community.document_loaders import PyMuPDFLoader

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Environment setup
load_dotenv()

def load_config(config_path: str = 'config.yaml') -> Dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# LLM setup based on config
llm_option = config['llm_option']

if llm_option == 'openai':
    from langchain_openai import ChatOpenAI
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
elif llm_option == 'azure':
    from langchain_openai import AzureChatOpenAI
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = config['azure']['openai_api_version']
    os.environ["OPENAI_API_BASE"] = config['azure']['api_base']
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
elif llm_option == 'gemini':
    from langchain_google_genai import ChatGoogleGenerativeAI
    import google.generativeai as genai
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
else:
    raise ValueError(f"Unsupported LLM option: {llm_option}")

class PDFTextProcessor:
    def __init__(self):
        self.full_text = ""
        self.current_filename = ""
        self.pdf_files = []

    def _process_filename(self, filename: str) -> str:
        words = filename.split()[:5]  # Take first 5 words
        processed_name = " ".join(words)
        processed_name = re.sub(r'[^\w\s-]', '', processed_name)
        processed_name = re.sub(r'\s+', '_', processed_name)
        return processed_name[:30]  # Limit to 30 characters

    def load_pdf_files(self, folder_path: str, num_files: int = None):
        self.pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        if num_files is not None:
            self.pdf_files = self.pdf_files[:num_files]
        logger.info(f"Loaded {len(self.pdf_files)} PDF files: {', '.join(self.pdf_files)}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            return " ".join(doc.page_content for doc in documents)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise

    def process_next_pdf(self, folder_path: str) -> str:
        if not self.pdf_files:
            logger.warning("No more PDF files to process.")
            return ""

        pdf_file = self.pdf_files.pop(0)
        pdf_path = os.path.join(folder_path, pdf_file)
        logger.info(f"Processing PDF: {pdf_path}")
        
        text = self._extract_text_from_pdf(pdf_path)
        self.full_text = text  # Replace full_text with the current PDF's text
        
        self.current_filename = self._process_filename(pdf_file)
        
        logger.info(f"Processed file: {pdf_file}")
        logger.info(f"Processed filename: {self.current_filename}")
        logger.info(f"Remaining files: {', '.join(self.pdf_files)}")
        
        return self.full_text

    def get_current_filename(self) -> str:
        return self.current_filename

def setup_llm():
    if llm_option == 'openai':
        return ChatOpenAI(
            model=config['openai']['llm']['model'],
            temperature=config['openai']['llm']['temperature']
        )
    elif llm_option == 'azure':
        return AzureChatOpenAI(
            deployment_name=config['azure']['llm']['azure_deployment'],
            temperature=config['azure']['llm']['temperature']
        )
    elif llm_option == 'gemini':
        return ChatGoogleGenerativeAI(
            model=config['gemini']['llm']['model_name'],
            temperature=config['gemini']['llm']['temperature']
        )

def create_prompt_template(prompt_config: Dict) -> PromptTemplate:
    return PromptTemplate(
        template=prompt_config['template'],
        input_variables=["journal_text"]
    )

def create_output_parser(model_name: str, fields: List[str]) -> PydanticOutputParser:
    field_dict = {field: (List[str], Field(...)) for field in fields}
    dynamic_model = create_model(f"{model_name.capitalize()}InfoOut", **field_dict)
    return PydanticOutputParser(pydantic_object=dynamic_model)

def save_to_csv(data: BaseModel, filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data.model_fields.keys())
        for row in zip(*data.model_dump().values()):
            writer.writerow(row)
    print(f"Data saved to {filename}")

def create_output_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created output folder: {folder_path}")
    else:
        print(f"Output folder already exists: {folder_path}")

def combine_csv_files(csv_files: List[str], output_file: str, prompt_order: List[str]):
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Item_No', 'Item', 'Found', 'Details', 'Source_Location_Section_Page_Sentence'])
        
        for item_no, prompt_name in enumerate(prompt_order, start=1):
            csv_file = next(file for file in csv_files if prompt_name in file)
            with open(csv_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow([item_no] + row)

def organize_files(output_folder: str, combined_filename: str, keep_combined_in_base=True):
    # Create a subfolder for individual CSV files
    individual_folder = os.path.join(output_folder, "individual_csvs")
    os.makedirs(individual_folder, exist_ok=True)

    # Move individual CSV files to the subfolder
    for filename in os.listdir(output_folder):
        if filename.endswith(".csv") and filename != combined_filename:
            if not filename.endswith("_combined_info.csv") or not keep_combined_in_base:
                shutil.move(os.path.join(output_folder, filename), os.path.join(individual_folder, filename))

def process_prompt(prompt_name: str, journal_full_text: str, llm, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            prompt_config = config['prompts'][prompt_name]
            prompt = create_prompt_template(prompt_config)
            output_parser = create_output_parser(prompt_name, config['output_models'][prompt_name])
            chain = prompt | llm | output_parser

            response = chain.invoke({"journal_text": journal_full_text})
            return response, None
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {prompt_name}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(config[llm_option]['retry_delay'])
            else:
                return None, str(e)

def main():
    create_output_folder(config['output_folder'])

    processor = PDFTextProcessor()
    num_files = config.get('number_of_pdf_files')
    processor.load_pdf_files(config['input_folder'], num_files)

    llm = setup_llm()

    prompt_order = ['llm_info', 'stochasticity', 'prompt_reporting', 'prompt_usage', 
                    'prompt_testing_optimization', 'test_dataset_independence']

    while processor.pdf_files or processor.current_filename:
        if processor.pdf_files:
            journal_full_text = processor.process_next_pdf(config['input_folder'])
        current_filename = processor.get_current_filename()

        csv_files = []
        for prompt_name in prompt_order:
            response, error = process_prompt(prompt_name, journal_full_text, llm)
            
            csv_filename = os.path.join(config['output_folder'], f"{current_filename}_{prompt_name}_info.csv")
            if response:
                save_to_csv(response, csv_filename)
                logger.info(f"Successfully processed and saved data for {prompt_name}")
            else:
                with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Error'])
                    writer.writerow([error])
                logger.error(f"Failed to process {prompt_name} after 3 attempts. Error saved to CSV.")
            
            csv_files.append(csv_filename)
            time.sleep(config[llm_option]['request_delay'])  # Add delay between requests

        # Combine CSV files for the current PDF
        if csv_files:
            try:
                combined_filename = f"{current_filename}_combined_info.csv"
                combined_file_path = os.path.join(config['output_folder'], combined_filename)
                combine_csv_files(csv_files, combined_file_path, prompt_order)
                logger.info(f"Combined CSV file created for {current_filename}: {combined_file_path}")

                # Organize files but keep combined_info in base folder
                organize_files(config['output_folder'], combined_filename, keep_combined_in_base=True)
                logger.info(f"Individual CSV files for {current_filename} moved to subfolder: {os.path.join(config['output_folder'], 'individual_csvs')}")
                logger.info(f"Combined CSV file {combined_filename} kept in base folder: {config['output_folder']}")
            except Exception as e:
                logger.error(f"Error combining CSV files for {current_filename}: {str(e)}")

        processor.current_filename = ""  # Reset current filename after processing

    # Final organization to ensure all combined files are in the base folder
    for filename in os.listdir(os.path.join(config['output_folder'], 'individual_csvs')):
        if filename.endswith("_combined_info.csv"):
            shutil.move(
                os.path.join(config['output_folder'], 'individual_csvs', filename),
                os.path.join(config['output_folder'], filename)
            )
    logger.info("All combined CSV files moved to base output folder.")

    logger.info("All PDF files have been processed.")

if __name__ == "__main__":
    main()