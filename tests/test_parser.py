import sys
import os
import json
import unittest
from pathlib import Path

# 프로젝트 루트 디렉토리를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcllm_txt_mode import IntegratedOutputParser, SearchItem


class TestIntegratedOutputParser(unittest.TestCase):
    """IntegratedOutputParser 클래스 테스트"""

    def setUp(self):
        self.parser = IntegratedOutputParser()

    def test_initialization(self):
        """출력 파서 초기화 테스트"""
        self.assertIsInstance(self.parser, IntegratedOutputParser)
        self.assertTrue(hasattr(self.parser, 'sections'))
        self.assertTrue(hasattr(self.parser, 'all_items'))
        self.assertTrue(hasattr(self.parser, 'all_descriptions'))
        self.assertTrue(hasattr(self.parser, 'all_search_locations'))

        # 모든 섹션이 초기화되었는지 확인
        expected_sections = [
            'llm_info', 'stochasticity', 'prompt_reporting', 
            'prompt_usage', 'prompt_testing_optimization', 
            'test_dataset_independence'
        ]
        self.assertEqual(list(self.parser.sections.keys()), expected_sections)

    def test_get_format_instructions(self):
        """형식 지침이 올바르게 생성되는지 테스트"""
        instructions = self.parser.get_format_instructions()
        
        # 지침에 모든 중요한 섹션과 항목이 포함되어 있는지 확인
        self.assertIn("Please provide your analysis in the following JSON format", instructions)
        self.assertIn("Section: llm_info", instructions)
        self.assertIn("LLM Name", instructions)
        self.assertIn("Section: stochasticity", instructions)
        self.assertIn("Temperature Settings", instructions)

    def test_parse_valid_json(self):
        """유효한 JSON 응답을 올바르게 파싱하는지 테스트"""
        # 유효한 JSON 응답 샘플
        sample_json = json.dumps({
            "findings": [
                {
                    "section_name": "llm_info",
                    "item_name": "LLM Name",
                    "present": "Y",
                    "location": {
                        "section": "Methods",
                        "page": "3",
                        "content": "We used GPT-4 for our analysis."
                    },
                    "details": "The paper explicitly mentions using GPT-4.",
                    "confidence": "HIGH"
                }
            ]
        })
        
        result = self.parser.parse(sample_json)
        
        # 결과가 예상대로 파싱되었는지 확인
        self.assertIn("llm_info", result)
        self.assertEqual(len(result["llm_info"]), 1)
        self.assertEqual(result["llm_info"][0]["item"], "LLM Name")
        self.assertEqual(result["llm_info"][0]["present"], "Y")
        self.assertEqual(result["llm_info"][0]["location"]["section"], "Methods")
        self.assertEqual(result["llm_info"][0]["confidence"], "HIGH")

    def test_parse_invalid_json(self):
        """잘못된 형식의 JSON에 대한 오류 처리 테스트"""
        invalid_json = "{ this is not valid json }"
        
        with self.assertRaises(ValueError) as context:
            self.parser.parse(invalid_json)
        
        self.assertIn("Failed to parse output as JSON", str(context.exception))

    def test_parse_missing_findings(self):
        """'findings' 키가 없는 JSON에 대한 오류 처리 테스트"""
        missing_findings_json = json.dumps({"results": []})
        
        with self.assertRaises(ValueError) as context:
            self.parser.parse(missing_findings_json)
        
        self.assertIn("Missing required 'findings' key", str(context.exception))


if __name__ == '__main__':
    unittest.main()