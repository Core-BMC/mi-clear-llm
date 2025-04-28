import os
import re
import io
import csv
import time
from openai import OpenAI
import yaml
import json
import fitz
import base64
import shutil
import logging
import pandas as pd
from tqdm import tqdm
from PIL import Image
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
logger.setLevel(logging.DEBUG)

# Clear any existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Create handlers
file_handler = logging.FileHandler(log_filename, mode='w')
console_handler = logging.StreamHandler()

# Set level for handlers
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.DEBUG)

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

def load_config(config_path: str = 'config_img-mode.yaml') -> Dict:
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
    """Enhanced integrated output parser class"""
    
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
    """Create enhanced integrated prompt template"""
    template = """You are an AI assistant specializing in extracting detailed information about LLM implementation from research papers. Please analyze the following images (journal pages) carefully.

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
        input_variables=["format_instructions"]
    )    

def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

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
        client = genai.Client(api_key=os.getenv("GOOGLE_CLOUD_API_KEY"))
        
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

class PDFImageProcessor:
    def __init__(self):
        self.current_filename = ""
        self.pdf_files = []

    def _process_filename(self, filename: str) -> str:
        words = filename.split()[:5]  # Take first 5 words
        processed_name = " ".join(words)
        processed_name = re.sub(r'[^\w\s-]', '', processed_name)
        processed_name = re.sub(r'\s+', '_', processed_name)
        return processed_name[:30]  # Limit to 30 characters

    def get_current_filename(self) -> str:
        return self.current_filename

    def load_pdf_files(self, folder_path: str, num_files: int = None):
        self.pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        if num_files is not None:
            self.pdf_files = self.pdf_files[:num_files]
        logger.info(f"Loaded {len(self.pdf_files)} PDF files: {', '.join(self.pdf_files)}")

    def process_next_pdf(self, folder_path: str) -> List[Image.Image]:
        if not self.pdf_files:
            logger.warning("No more PDF files to process.")
            return []

        pdf_file = self.pdf_files.pop(0)
        pdf_path = os.path.join(folder_path, pdf_file)
        logger.info(f"Processing PDF: {pdf_path}")
        
        images = pdf_to_images(pdf_path)
        
        self.current_filename = self._process_filename(pdf_file)
        
        logger.info(f"Processed file: {pdf_file}")
        logger.info(f"Processed filename: {self.current_filename}")
        logger.info(f"Remaining files: {', '.join(self.pdf_files)}")
        
        return images

def create_output_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Created output folder: {folder_path}")
    else:
        logger.info(f"Output folder already exists: {folder_path}")

def upload_images(images: List[Image.Image]) -> List[Any]:
    """Convert PIL images to format suitable for model input"""
    if config['llm_option'] == 'gemini':
        processed_images = []
        for img in images:
            # Convert image to RGB mode if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image if it's too large (optional)
            max_size = 1024  # Maximum size for either dimension
            if img.width > max_size or img.height > max_size:
                ratio = max_size / max(img.width, img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            processed_images.append(img)
        
        return processed_images
    else:
        return images  # For other models, keep original format

def clean_up_images():
    """Clean up temporary image files"""
    image_dirs = ['gemini_image', 'pdf_images']  # Add all temporary image directories here
    for image_dir in image_dirs:
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                file_path = os.path.join(image_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
            try:
                os.rmdir(image_dir)  # Try to remove the empty directory
                logger.info(f"Removed directory: {image_dir}")
            except Exception as e:
                logger.error(f"Error removing directory {image_dir}: {e}")

def process_prompt(images: List[Image.Image], llm, max_retries: int = 3) -> tuple[list[Any] | None, str | None]:
    """Process images with the specified LLM model"""
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
            query = prompt.format(format_instructions=format_instructions)

            if config['llm_option'] == 'gemini':
                # Create content list with text prompt first
                contents = [query]
                
                # Add processed images directly
                contents.extend(images)
                
                # Generate content with all images
                response = llm['client'].models.generate_content(
                    model=llm['model'],
                    contents=contents,
                    config=llm['config']
                )
                
                if hasattr(response, 'text'):
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        try:
                            json_str = json_match.group(0)
                            parsed_json = json.loads(json_str)
                            
                            if "findings" not in parsed_json:
                                raise ValueError("Missing 'findings' key in response")
                            
                            return [{"findings": parsed_json["findings"]}], None
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Failed to parse JSON: {str(e)}")
                    else:
                        raise ValueError("No JSON found in response")
                else:
                    raise ValueError("Response has no text attribute")

            elif isinstance(llm, OpenAI):  # O1
                # Prepare base message structure
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }]
                
                # Add images to the content array
                for img in images:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    })

                response = llm.chat.completions.create(
                    model=config['openai_o1']['llm']['model'],
                    messages=messages,
                    response_format={"type": "json_object"},
                    reasoning_effort="medium"
                )
                content = response.choices[0].message.content
                
                try:
                    parsed_json = json.loads(content)
                    if "findings" not in parsed_json:
                        raise ValueError("Missing 'findings' key in response")
                    
                    parsed_response = {"findings": parsed_json["findings"]}
                    return [parsed_response], None
                except Exception as e:
                    raise ValueError(f"Error processing O1 response: {str(e)}")

            else:  # ChatOpenAI/AzureChatOpenAI
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{query}\nIMPORTANT: Please format your entire response as a single JSON object with a 'findings' array. Do not include any text before or after the JSON. Do not use markdown code blocks."
                        }
                    ]
                }]
                
                # Add images to the content array
                for img in images:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    })
                
                try:
                    response = llm.invoke(messages)
                    content = response.content
                    logger.debug(f"Raw response content: {content}")
                    
                    # Clean up the content
                    content = content.strip()
                    
                    # Remove markdown code blocks if present
                    if content.startswith('```'):
                        content = re.sub(r'^```(?:json)?\s*', '', content)
                        content = re.sub(r'\s*```$', '', content)
                    
                    try:
                        # Try direct JSON parsing first
                        parsed_json = json.loads(content)
                    except json.JSONDecodeError:
                        # If direct parsing fails, try to clean the content further
                        content = re.sub(r'^[^{]*', '', content)  # Remove any text before first {
                        content = re.sub(r'[^}]*$', '', content)  # Remove any text after last }
                        parsed_json = json.loads(content)
                    
                    if "findings" not in parsed_json:
                        raise ValueError("Response JSON missing required 'findings' key")
                    
                    if not isinstance(parsed_json["findings"], list):
                        raise ValueError("'findings' must be an array")
                    
                    parsed_response = {"findings": parsed_json["findings"]}
                    logger.info(f"Successfully processed with {len(parsed_json['findings'])} findings")
                    return [parsed_response], None
                    
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing error: {str(je)}")
                    raise ValueError(f"JSON parsing error: {str(je)}")
                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
                    raise
                    
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1} failed with error: {last_error}")
            if attempt == max_retries - 1:
                return create_error_response(), last_error

    return create_error_response(), f"Maximum retries exceeded: {last_error}"
                    

def img_to_base64(img: Image.Image) -> str:
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode('utf-8')

def create_error_response() -> Dict[str, List[Dict[str, Any]]]:
    """Create error response in integrated format"""
    parser = create_output_parser()
    findings = []
    
    # Create findings for each section and item
    for section_name, items in parser.sections.items():
        for item in items:
            finding = {
                "section_name": section_name,
                "item_name": item.name,
                "present": "N",
                "location": {
                    "section": "Not Found",
                    "page": "Not Found",
                    "content": "Not Found"
                },
                "details": "Not Found",
                "confidence": "LOW"
            }
            findings.append(finding)
    
    return {"findings": findings}

def save_to_csv(data: List[Dict], filename: str):
    """Save analysis results to CSV without batch processing"""
    try:
        rows = []
        findings = data[0].get("findings", [])
        
        if not findings:
            logger.warning("No findings data to save")
            return None
            
        for finding in findings:
            row = {
                "Section": finding["section_name"],
                "Item": finding["item_name"],
                "Present": finding["present"],
                "Location": f"{finding['location']['section']} (p.{finding['location']['page']})",
                "Content": finding["location"]["content"],
                "Additional Info": finding["details"],
                "Confidence": finding.get("confidence", "Not specified")
            }
            rows.append(row)

        if not rows:
            logger.warning("No rows generated from findings")
            return None

        output_filename = f"{filename}.csv"
        df = pd.DataFrame(rows)
        
        if df.empty or len(df.columns) == 0:
            logger.error("Empty DataFrame generated")
            return None
            
        # Define section order from IntegratedOutputParser
        section_order = list(IntegratedOutputParser.sections.keys())
        df['Section'] = pd.Categorical(
            df['Section'],
            categories=section_order,
            ordered=True
        )

        # Create item order from all sections
        item_order = []
        for section in section_order:
            items = [item.name for item in IntegratedOutputParser.sections[section]]
            item_order.extend(items)
        
        df['Item'] = pd.Categorical(
            df['Item'],
            categories=item_order,
            ordered=True
        )
        
        # Sort by section and item
        df = df.sort_values(['Section', 'Item'])
        df.to_csv(output_filename, index=False)
        
        logger.info(f"Data saved to {output_filename}")
        return output_filename
        
    except Exception as e:
        logger.error(f"Error saving to CSV {filename}: {str(e)}")
        backup_filename = f"{filename}.error.json"
        with open(backup_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Backup data saved to {backup_filename}")
        return None

def main():
    """Main function to process PDF files and analyze them using the LLM model."""
    try:
        # Create output folder
        create_output_folder(config['output_folder'])
        
        # Initialize PDF processor
        processor = PDFImageProcessor()
        num_files = config.get('number_of_pdf_files')
        processor.load_pdf_files(config['input_folder'], num_files)

        total_pdfs = len(processor.pdf_files)
        if total_pdfs == 0:
            logger.warning("No PDF files found to process.")
            return

        logger.info(f"Starting to process {total_pdfs} PDF files")
        
        # Clean up any existing temporary images before starting
        clean_up_images()
        
        with tqdm(total=total_pdfs, desc="Processing PDFs") as pbar:
            while processor.pdf_files or processor.current_filename:
                try:
                    if processor.pdf_files:
                        # Clean up images from previous iteration
                        clean_up_images()
                        
                        # Get images from PDF
                        images = processor.process_next_pdf(config['input_folder'])
                        if not images:
                            logger.error("Failed to extract images from PDF")
                            continue
                        
                        # Log image details for debugging
                        logger.debug(f"Extracted {len(images)} images from PDF")
                        for i, img in enumerate(images):
                            logger.debug(f"Image {i}: Size={img.size}, Mode={img.mode}")
                        
                        # Process images for model input
                        processed_images = upload_images(images)
                        if not processed_images:
                            logger.error("Failed to process images")
                            continue
                        
                        logger.info(f"Successfully processed {len(processed_images)} images")
                    
                    current_filename = processor.get_current_filename()
                    if not current_filename:
                        logger.warning("No current filename available")
                        continue
                    
                    # Process with the model
                    responses, error = process_prompt(processed_images, llm)
                    
                    if error:
                        # Handle error case
                        error_filename = os.path.join(
                            config['output_folder'], 
                            f"{current_filename}_error.csv"
                        )
                        with open(error_filename, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['Error'])
                            writer.writerow([error])
                        logger.error(f"Failed to process {current_filename}. Error saved to: {error_filename}")
                    else:
                        # Process successful responses
                        output_filename = os.path.join(
                            config['output_folder'], 
                            f"{current_filename}_analysis"
                        )
                        result_file = save_to_csv(responses, output_filename)
                        
                        if result_file:
                            logger.info(f"Successfully processed {current_filename}")
                            logger.info(f"Results saved to: {result_file}")
                        else:
                            logger.error(f"Failed to save results for {current_filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
                    
                finally:
                    # Clean up and update progress
                    processor.current_filename = ""
                    clean_up_images()
                    pbar.update(1)
        
        # Final cleanup
        clean_up_images()
        logger.info("All PDF files have been processed.")
        
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Ensure all temporary resources are cleaned up
        clean_up_images()
        logger.info("Processing completed. All temporary resources cleaned up.")

if __name__ == "__main__":
    main()