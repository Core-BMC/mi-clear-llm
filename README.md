# MI-CLEAR-LLM Information Extractor

## Overview

This project implements an automated information extraction tool for the MI-CLEAR-LLM (The Minimum reporting Items for CLear Evaluation of Accuracy Reports of Large Language Models in healthcare) checklist. The MI-CLEAR-LLM checklist aims to provide a set of essential items for the transparent reporting of clinical studies that present the accuracy of Large Language Models (LLMs) in healthcare applications, promoting clearer evaluation of study findings.

## Purpose

The primary goal of this tool is to assist researchers and reviewers in extracting and analyzing key information from research papers that evaluate LLMs in healthcare contexts. By automating the extraction process, we aim to:

1. Ensure consistent evaluation of papers against the MI-CLEAR-LLM checklist items.
2. Facilitate faster and more accurate assessment of LLM studies in healthcare.
3. Promote transparency and reproducibility in LLM research within the medical field.

## Features

- Extracts information related to MI-CLEAR-LLM checklist items, including:
  - LLM details (name, version, manufacturer, etc.)
  - Stochasticity handling in the study
  - Prompt reporting and usage
  - Prompt testing and optimization methods
  - Test dataset independence
- Supports multiple LLM backends for information extraction: OpenAI, Azure OpenAI, and Google's Gemini
- Processes multiple PDF files of research papers
- Generates structured CSV outputs for easy analysis

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/mi-clear-llm.git
   cd mi-clear-llm
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

5. Configure the `config.yaml` file:
   - Set the `llm_option` to your preferred LLM backend ('openai', 'azure', or 'gemini')
   - Adjust the model settings, input/output folders, and other parameters as needed

## Usage

1. Place the PDF files of the research papers you want to analyze in the input folder specified in `config.yaml`.

2. Run the main script:
   ```
   python mcllm_main_v1.py
   ```

3. The script will process the PDF files and generate CSV outputs in the specified output folder. These outputs will contain the extracted information relevant to the MI-CLEAR-LLM checklist items.

## Configuration

You can modify the `config.yaml` file to adjust various settings:

- LLM backend and model settings
- Input and output folder paths
- Number of PDF files to process
- Prompt templates for information extraction (aligned with MI-CLEAR-LLM checklist items)
- Output model configurations

## Output

The script generates two types of CSV files for each processed PDF:

1. Individual CSV files for each MI-CLEAR-LLM checklist category
2. A combined CSV file with all extracted information

These files are organized in the output folder specified in the configuration, allowing for easy review and analysis of the extracted data.

## Contributing

Contributions to improve the project are welcome, especially those that enhance the tool's ability to extract MI-CLEAR-LLM relevant information. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a new Pull Request

## License

[MIT License](LICENSE)

## Acknowledgments

- This project is based on the MI-CLEAR-LLM checklist, which aims to improve the reporting quality of LLM studies in healthcare.
- We use various open-source libraries and AI models. Please refer to the `requirements.txt` file for a full list of dependencies.

## Citation

If you use this tool in your research, please cite both the original MI-CLEAR-LLM checklist paper and this repository:

[Include citation for MI-CLEAR-LLM paper when available]

```
@software{mi_clear_llm,
  author = {[Your Name]},
  title = {MI-CLEAR-LLM Information Extractor},
  year = {[Current Year]},
  url = {https://github.com/your-username/mi-clear-llm-extractor}
}
```

## Contact

For issues or contributions, please reach out to the repository maintainer:
- **Email**: heohwon@gmail.com
- **GitHub**: [Core-BMC](https://github.com/Core-BMC), [HwonHeo](https://github.com/hwonheo)


## Disclaimer

This tool is designed to assist in the extraction of information relevant to the MI-CLEAR-LLM checklist. It should be used as an aid in the review process and not as a replacement for human expert evaluation. Always verify the extracted information against the original research papers.