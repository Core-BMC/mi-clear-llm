import os
import re
from openai import OpenAI
import yaml
import time
import json
import fitz
import logging
import pandas as pd 
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from google import genai
from google.genai import types
from typing import ClassVar, List, Dict, Any, NamedTuple

# Ensure logs directory exists
logs_dir = "./logs"
os.makedirs(logs_dir, exist_ok=True)

# Logging setup
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(logs_dir, f"log_{current_time}.log")

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Create handlers
file_handler = logging.FileHandler(log_filename, mode='w')
console_handler = logging.StreamHandler()

# Set level for handlers
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_formatter)
console_handler.setFormatter(log_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Suppress logs from other libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Environment setup
load_dotenv()

def load_config(config_path: str = 'config_txt-mode.yaml') -> Dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_config()

class SearchItem(NamedTuple):
    """Class for defining search items"""
    name: str
    description: str
    search_locations: str
    example_phrases: List[str]

class IntegratedOutputParser(BaseOutputParser, BaseModel):
    """Enhanced integrated output parser class that handles all sections in a single response"""
    
    # Define the model fields explicitly
    all_items: List[str] = Field(default_factory=list)
    all_descriptions: List[str] = Field(default_factory=list)
    all_search_locations: List[str] = Field(default_factory=list)
    
    # Section definitions with search items as ClassVar
    sections: ClassVar[Dict[str, List[SearchItem]]] = {
        # From llm_info prompt
        'llm_info': [
            SearchItem(
                "LLM Name",
                "The specific name of the language model used in the study",
                "Check Methods section, Abstract, or Implementation Details",
                ["we used", "model employed", "leveraged", "utilized"]
            ),
            SearchItem(
                "Version",
                "The specific version number or identifier of the model",
                "Look in Methods section or Technical Details",
                ["version", "release", "variant", "v."]
            ),
            SearchItem(
                "Manufacturer",
                "The company or organization that developed the model",
                "Usually mentioned in Methods or Introduction sections",
                ["developed by", "created by", "from", "provided by"]
            ),
            SearchItem(
                "Training Data Cutoff Date",
                "The latest date of data included in model training",
                "Check Methods, Limitations, or Model Specifications",
                ["training cutoff", "knowledge cutoff", "trained until", "data up to"]
            ),
            SearchItem(
                "RAG Access",
                "Whether and how the model accessed external data",
                "Look in Methods, Implementation, or System Architecture sections",
                ["retrieval-augmented", "external knowledge", "real-time access", "data retrieval"]
            ),
            SearchItem(
                "Query Date",
                "The date when the model was actually queried for the study",
                "Check Methods, Experimental Setup, or Results sections",
                ["conducted on", "queried on", "experiments performed", "data collection period"]
            ),
            SearchItem(
                "Access Method",
                "How the model was accessed (API, public interface, institution-specific)",
                "Check Methods, Implementation, or Technical Details sections",
                ["API access", "API key", "public interface", "ChatGPT interface", "institutional access", "custom deployment"]
            ),
            SearchItem(
                "Access Details",
                "Specific details about the access method including API version or interface type",
                "Look in Methods, Implementation, or Technical Details sections",
                ["API version", "interface version", "access token", "authentication method", "deployment details"]
            ),
            SearchItem(
                "Access Restrictions",
                "Any limitations or restrictions on model access",
                "Check Methods, Limitations, or Technical Details sections",
                ["access limitations", "usage restrictions", "rate limits", "institutional constraints", "deployment constraints"]
            )
        ],
        # From stochasticity prompt
        'stochasticity': [
            SearchItem(
                "Number of Querying Attempts",
                "Number of times each query was attempted",
                "Check Methods and Experimental Setup sections",
                ["number of attempts", "repeated queries", "multiple runs"]
            ),
            SearchItem(
                "Synthesis of Multiple Results",
                "How results from multiple attempts were combined",
                "Look in Methods and Results sections",
                ["synthesis", "aggregation", "combining results", "result combination"]
            ),
            SearchItem(
                "Reliability Analysis",
                "Analysis of reliability across attempts",
                "Check Methods and Results sections",
                ["reliability", "consistency", "variance analysis", "statistical analysis"]
            ),
            SearchItem(
                "Temperature Settings",
                "Temperature parameter value and rationale",
                "Find in Methods or Experimental Setup sections",
                ["temperature", "sampling parameter", "randomness setting"]
            )
        ],
        # From prompt_reporting prompt
        'prompt_reporting': [
            SearchItem(
                "Precise Spellings",
                "Exact words and their spellings used in prompts",
                "Check Methods, Appendix, or Supplementary Materials",
                ["spelling", "exact wording", "prompt text", "verbatim"]
            ),
            SearchItem(
                "Symbols Used",
                "Special characters or symbols in prompts",
                "Look in Methods and Appendix sections",
                ["symbols", "special characters", "syntax", "formatting"]
            ),
            SearchItem(
                "Punctuation",
                "Specific punctuation marks used in prompts",
                "Check Methods and Appendix sections",
                ["punctuation", "marks", "formatting", "syntax"]
            ),
            SearchItem(
                "Spaces",
                "Use of spaces and line breaks in prompts",
                "Look in Methods and Appendix sections",
                ["spacing", "line breaks", "whitespace", "format"]
            ),
            SearchItem(
                "Other Syntax",
                "Other syntactical elements in prompts",
                "Check Methods and Appendix sections",
                ["syntax", "structure", "format", "elements"]
            )
        ],
        # From prompt_usage prompt
        'prompt_usage': [
            SearchItem(
                "Chat Session Structure",
                "Whether queries were handled as individual or combined sessions",
                "Look in Methods and Implementation sections",
                ["chat session", "conversation structure", "dialogue format", "interaction method"]
            ),
            SearchItem(
                "Query Input Method",
                "How multiple queries were input to the system",
                "Check Methods and Implementation sections",
                ["input method", "query submission", "batch processing", "sequential input"]
            ),
            SearchItem(
                "Prompt Analysis Synthesis",
                "How the analysis of multiple prompts was synthesized",
                "Look in Methods and Results sections",
                ["analysis synthesis", "result combination", "aggregation method", "comparative analysis"]
            )
        ],
        # From prompt_testing_optimization prompt
        'prompt_testing_optimization': [
            SearchItem(
                "Steps to Create Prompts",
                "Process used to develop the prompts",
                "Check Methods and Prompt Engineering sections",
                ["prompt creation", "development process", "design workflow", "engineering steps"]
            ),
            SearchItem(
                "Rationale for Specific Wording",
                "Explanations for specific word choices in prompts",
                "Look in Methods and Design sections",
                ["word choice", "phrasing", "terminology selection", "language optimization"]
            ),
            SearchItem(
                "Testing and Optimization Methods",
                "How prompts were tested and refined",
                "Check Methods and Evaluation sections",
                ["prompt testing", "refinement process", "optimization method", "validation approach"]
            )
        ],
        # From test_dataset_independence prompt
        'test_dataset_independence': [
            SearchItem(
                "Independent Test Data",
                "Whether test data was independent from training and optimization",
                "Look in Methods, Data, and Evaluation sections",
                ["independent test set", "held-out data", "unseen data", "separate validation"]
            ),
            SearchItem(
                "Data Source from Internet",
                "URLs and sources of test data",
                "Check Data Collection and Methods sections",
                ["data source", "URL", "internet source", "online data"]
            )
        ]
    }

    def __init__(self, **data):
        super().__init__(**data)
        # Generate complete item list and descriptions
        for section_items in self.sections.values():
            for item in section_items:
                self.all_items.append(item.name)
                self.all_descriptions.append(item.description)
                self.all_search_locations.append(item.search_locations)

    def get_format_instructions(self) -> str:
        """Return enhanced output format instructions for all sections"""
        instructions = """
        Please provide your analysis in the following JSON format:
        {
            "findings": [
                {
                    "section_name": "name of the section (e.g., llm_info, stochasticity, etc.)",
                    "item_name": "name of the item being searched",
                    "present": "Y/N",
                    "location": {
                        "section": "section name where found",
                        "page": "page number or identifier",
                        "content": "relevant quote or content from the text"
                    },
                    "details": "any additional context or details",
                    "confidence": "HIGH/MEDIUM/LOW - based on clarity and directness of evidence"
                }
            ]
        }

        Please analyze all of the following items:
        """
        
        # Add section and item-specific guidance
        for section_name, section_items in self.sections.items():
            instructions += f"\n\nSection: {section_name}\n"
            for item in section_items:
                instructions += f"\n- {item.name}:"
                instructions += f"\n  Description: {item.description}"
                instructions += f"\n  Where to look: {item.search_locations}"
        
        return instructions
    
    def parse(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse the response into a structured format"""
        try:
            parsed = json.loads(text)
            
            if "findings" not in parsed:
                raise ValueError("Missing required 'findings' key")
            
            # Organize findings by section
            results = {section: [] for section in self.sections.keys()}
            
            for finding in parsed["findings"]:
                section_name = finding.get("section_name")
                if section_name not in results:
                    continue
                
                processed_finding = {
                    "item": finding.get("item_name"),
                    "present": finding.get("present", "N"),
                    "location": {
                        "section": finding.get("location", {}).get("section", "Not specified"),
                        "page": finding.get("location", {}).get("page", "Not specified"),
                        "content": finding.get("location", {}).get("content", "Not found")
                    },
                    "details": finding.get("details", "Not provided"),
                    "confidence": finding.get("confidence", "LOW")
                }
                
                results[section_name].append(processed_finding)
            
            return results
            
        except json.JSONDecodeError:
            raise ValueError("Failed to parse output as JSON")
        except Exception as e:
            raise ValueError(f"Error parsing output: {str(e)}")

def create_output_parser() -> IntegratedOutputParser:
    """Factory function to create a new output parser instance"""
    return IntegratedOutputParser()

def create_integrated_prompt_template() -> PromptTemplate:
    """Create enhanced integrated prompt template for all sections"""
    template = """You are an AI assistant specializing in extracting detailed information about LLM implementation from research papers. Please analyze the following text carefully and provide information for ALL sections in a single response:

Journal text: {journal_text}

Your task is to extract specific information about how the LLM was implemented and evaluated in this research. For each item in ALL sections, please provide:

1. The section name (e.g., llm_info, stochasticity, etc.)
2. The specific item name
3. Confirm presence (Y/N)
4. Identify exact location (section and page)
5. Quote relevant content directly
6. Provide additional context
7. Assess confidence in finding (HIGH/MEDIUM/LOW)

Important guidelines:
- Begin with Methods and Implementation sections
- Examine tables and figures for technical details
- Verify all quotes by re-reading
- Note any ambiguity or unclear information
- Consider both explicit statements and implicit evidence
- Ensure you cover ALL sections and items in a single response
- All sections and items must be fully analyzed and printed without summarizing or omitting content, including phrases like '... (rest of the sections and items follow the same structure).' 

{format_instructions}

Before submitting your response, please verify:
1. You have analyzed ALL sections and their items
2. Each finding has ALL required fields
3. NO null values in your response
4. All section names match the expected values exactly
"""
    
    return PromptTemplate(
        template=template,
        input_variables=["journal_text", "format_instructions"]
    )


def setup_llm():
    llm_option = config['llm_option']

    if llm_option == 'openai':
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY_SS")
        return ChatOpenAI(
            model=config['openai']['llm']['model'],
            temperature=config['openai']['llm']['temperature']
        )
    
    elif llm_option == 'openai_o1':
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY_SS")
        return OpenAI() 
    
    elif llm_option == 'azure':
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = config['azure']['openai_api_version']
        os.environ["OPENAI_API_BASE"] = config['azure']['api_base']
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY_WUS")
        return AzureChatOpenAI(
            deployment_name=config['azure']['llm']['azure_deployment'],
            temperature=config['azure']['llm']['temperature']
        )
        
    elif llm_option == 'gemini':
        # Google Gemini setup using google.genai
        from google import genai
        from google.genai import types
        
        # Create client instance with API key
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_WS"))
        
        # Create generation config
        generation_config = types.GenerateContentConfig(
            max_output_tokens=config['gemini']['llm']['max_output_tokens'],
            temperature=config['gemini']['llm']['temperature'],
            top_p=config['gemini']['llm']['top_p']
        )
        
        return {
            'client': client,
            'model': config['gemini']['llm']['model_name'],
            'config': generation_config
        }
    else:
        raise ValueError(f"Unsupported LLM option: {llm_option}")

# LLM setup
llm = setup_llm()

class PDFTextProcessor:
    def __init__(self):
        self.current_filename = ""
        self.pdf_files = []
        self.full_text = ""

    def _process_filename(self, filename: str) -> str:
        words = filename.split()[:5]
        processed_name = " ".join(words)
        processed_name = re.sub(r'[^\w\s-]', '', processed_name)
        processed_name = re.sub(r'\s+', '_', processed_name)
        return processed_name[:30]

    def get_current_filename(self) -> str:
        return self.current_filename

    def load_pdf_files(self, folder_path: str, num_files: int = None):
        self.pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        if num_files is not None:
            self.pdf_files = self.pdf_files[:num_files]
        logger.info(f"Loaded {len(self.pdf_files)} PDF files: {', '.join(self.pdf_files)}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
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
        
        self.full_text = self._extract_text_from_pdf(pdf_path)
        self.current_filename = self._process_filename(pdf_file)
        
        logger.info(f"Processed file: {pdf_file}")
        logger.info(f"Processed filename: {self.current_filename}")
        logger.info(f"Remaining files: {', '.join(self.pdf_files)}")
        
        return self.full_text

def create_output_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Created output folder: {folder_path}")
    else:
        logger.info(f"Output folder already exists: {folder_path}")


def process_prompt(journal_text: str, llm, max_retries: int = 5) -> tuple[dict | None, str | None]:
    last_error = None
    output_parser = create_output_parser()

    for attempt in range(max_retries):
        if attempt > 0:
            retry_delay = config[config['llm_option']]['retry_delay']
            logger.info(f"Retry attempt {attempt + 1}. Waiting {retry_delay} seconds...")
            time.sleep(retry_delay)
        
        try:
            prompt = create_integrated_prompt_template()
            format_instructions = output_parser.get_format_instructions()
            
            query = prompt.format(
                journal_text=journal_text,
                format_instructions=format_instructions
            )

            if config['llm_option'] == 'openai_o1':
                # Special handling for o1 model
                response = llm.chat.completions.create(
                    model=config['openai_o1']['llm']['model'],
                    messages=[{"role": "user", "content": query}],
                    response_format={"type": "json_object"},
                    reasoning_effort="medium"
                )
                content = response.choices[0].message.content
                
            elif config['llm_option'] == 'gemini':
                # Updated Gemini handling using google.genai
                response = llm['client'].models.generate_content(
                    model=llm['model'],
                    contents=query,
                    config=llm['config']
                )
                content = response.text
                
            else:  # OpenAI/Azure
                response = llm.invoke(query)
                content = response.content

            if config['llm_option'] == 'openai_o1':
                # For o1, the response is already in JSON format
                try:
                    cleaned_json = {
                        k: ["Not Found" if v is None else v for v in values]
                        for k, values in json.loads(content).items()
                    }
                    parsed_response = output_parser.parse(json.dumps(cleaned_json))
                    logger.info(f"Successfully processed on attempt {attempt + 1}")
                    return [parsed_response], None
                except Exception as e:
                    last_error = f"JSON parsing error: {str(e)}"
                    logger.warning(f"Attempt {attempt + 1} failed with parsing error: {last_error}")
                    if attempt == max_retries - 1:
                        return create_error_response(), last_error
            else:
                # Existing logic for other models
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        parsed_json = json.loads(json_str)
                        cleaned_json = {
                            k: ["Not Found" if v is None else v for v in values]
                            for k, values in parsed_json.items()
                        }
                        
                        parsed_response = output_parser.parse(json.dumps(cleaned_json))
                        logger.info(f"Successfully processed on attempt {attempt + 1}")
                        return [parsed_response], None
                        
                    except Exception as e:
                        last_error = f"JSON parsing error: {str(e)}"
                        logger.warning(f"Attempt {attempt + 1} failed with parsing error: {last_error}")
                        if attempt == max_retries - 1:
                            return create_error_response(), last_error
                else:
                    last_error = "No JSON found in response"
                    logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    if attempt == max_retries - 1:
                        return create_error_response(), last_error
                    
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1} failed with error: {last_error}")
            if attempt == max_retries - 1:
                return create_error_response(), last_error
    
    return create_error_response(), f"Maximum retries exceeded: {last_error}"


def create_error_response() -> Dict[str, List[Dict[str, Any]]]:
    parser = create_output_parser()
    return {
        section: [{
            "item": item.name,
            "present": "N",
            "location": {
                "section": "Not Found",
                "page": "Not Found",
                "content": "Not Found"
            },
            "details": "Not Found",
            "confidence": "LOW"
        } for item in items]
        for section, items in parser.sections.items()
    }

def save_to_csv(data: List[Dict], filename: str):
    try:
        rows = []
        for section_name, section_data in data[0].items(): 
            for finding in section_data:
                row = {
                    "Section": section_name,
                    "Item": finding["item"],
                    "Present": finding["present"],
                    "Location": f"{finding['location']['section']} (p.{finding['location']['page']})",
                    "Content": finding["location"]["content"],
                    "Additional Info": finding["details"],
                    "Confidence": finding.get("confidence", "Not specified")
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving to CSV {filename}: {str(e)}")
        backup_filename = f"{filename}.error.json"
        with open(backup_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Backup data saved to {backup_filename}")

def main():
    create_output_folder(config['output_folder'])
    processor = PDFTextProcessor()
    num_files = config.get('number_of_pdf_files')
    processor.load_pdf_files(config['input_folder'], num_files)

    with tqdm(total=len(processor.pdf_files), desc="Processing PDFs") as pbar:
        while processor.pdf_files or processor.current_filename:
            if processor.pdf_files:
                journal_text = processor.process_next_pdf(config['input_folder'])
            current_filename = processor.get_current_filename()

            responses, error = process_prompt(journal_text, llm)
            
            if not responses:
                error_filename = os.path.join(
                    config['output_folder'], 
                    f"{current_filename}_analysis_error.txt"
                )
                with open(error_filename, 'w', encoding='utf-8') as f:
                    f.write(error)
                logger.error(f"Failed to process analysis. Error saved to: {error_filename}")
            else:
                output_filename = os.path.join(
                    config['output_folder'],
                    f"{current_filename}_analysis.csv"
                )
                save_to_csv(responses, output_filename)
                logger.info(f"Successfully processed and saved analysis for {current_filename}")
            
            pbar.update(1)
            processor.current_filename = ""

    logger.info("All PDF files have been processed.")

if __name__ == "__main__":
    main()