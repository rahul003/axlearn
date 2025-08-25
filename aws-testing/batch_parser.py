<<<<<<< HEAD
=======
#!/usr/bin/env python3
>>>>>>> efc2c09 (Add comprehensive documentation to batch_parser.py)
"""
Batch Test Result Parser for Model Testing

This module parses pytest XML output files from model tests,
It categorizes test failures, extracts model configurations, and uploads metrics to Scuba database
for monitoring and analysis.

Author: @yoamol
Created: 2025
Last Modified: 2025
Version: 1.0

Usage:
    python batch_parser.py --test_directory /path/to/tests --event_id VERSION_SET_ID

Dependencies:
    - xml.etree.ElementTree: XML parsing
    - kaena_scuba_data_api: Metrics upload to Scuba
    - Standard library modules: os, json, re, datetime, typing, dataclasses
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from kaena_scuba_data_api import ScubaHeartbeat

@dataclass
class TestResult:
    """Data class representing a single test result.
    
    Attributes:
        test_name: Name of the test method
        model_type: Type and configuration of the model being tested
        status: Test status ("PASS" or "FAIL")
        duration: Test execution time in seconds
        error_message: Error message if test failed
        error_category: Categorized error type
        error_categories: List of error categories for multiple errors
    """
    test_name: str
    model_type: str
    status: str
    duration: Optional[float]
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    error_categories: Optional[List[str]] = None

class BatchIntegLogParser:
    """Parser for batch processing of pytest XML test results.
    
    This class handles parsing XML test results from multiple test suites,
    categorizing failures, and uploading metrics to Scuba database.
    """
    
    def __init__(self, base_folder: str, event_id: str = "unknown", dry_run: bool = True):
        """Initialize the batch parser.
        
        Args:
            base_folder: Root directory containing test suite folders
            event_id: Version set identifier for tracking
            dry_run: If True, don't actually upload to Scuba
        """
        self.base_folder = base_folder
        self.event_id = event_id
        self.dry_run = dry_run
        self.run_timestamp = datetime.now().isoformat()
        try:
            self.scuba = ScubaHeartbeat(dry_run=dry_run)
            self.scuba_available = True
        except Exception as e:
            print(f"Warning: Scuba not available ({e}). Running in offline mode.")
            self.scuba = None
            self.scuba_available = False
        
    def find_test_suites(self) -> List[str]:
        """Find all test suite folders containing XML files.
        
        Scans the base folder for subdirectories that contain .xml files,
        which are assumed to be pytest result files.
        
        Returns:
            List of test suite folder names (sorted alphabetically)
        """
        test_suites = []
        for item in os.listdir(self.base_folder):
            suite_path = os.path.join(self.base_folder, item)
            if os.path.isdir(suite_path):
                # Check if folder has any .xml files
                xml_files = [f for f in os.listdir(suite_path) if f.endswith('.xml')]
                if xml_files:
                    test_suites.append(item)
        return sorted(test_suites)
    
    def parse_pytest_xml(self, xml_file):
        """Parse pytest XML file and extract test results.
        
        Parses a single pytest XML file to extract test statistics and failure information.
        
        Args:
            xml_file: Path to the XML file to parse
            
        Returns:
            Dict containing test statistics:
                - passed: Number of passed tests
                - failed: Number of failed tests
                - errors: Number of tests with errors
                - skipped: Number of skipped tests
                - failures: List of (test_name, error_message) tuples
                - all_tests: List of all test names
                - total: Total number of tests
            Returns None if parsing fails.
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing XML file: {e}")
            return None
        except FileNotFoundError:
            print(f"File not found: {xml_file}")
            return None
        
        # Initialize counters
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        failures = []
        all_tests = []
        
        # Find all testcase elements
        testcases = root.findall('.//testcase')
        
        for testcase in testcases:
            test_name = testcase.get('name', 'Unknown')
            classname = testcase.get('classname', '')
            full_name = f"{classname}::{test_name}" if classname else test_name

            skipped_elem = testcase.find('skipped')
            if skipped_elem is not None:
                skipped += 1
                continue

            all_tests.append(full_name)
            # Check for failure
            failure = testcase.find('failure')
            if failure is not None:
                failed += 1
                # Extract short error message (first line of failure message)
                failure_msg = failure.get('message', '')
                if not failure_msg and failure.text:
                    # If no message attribute, use first line of text content
                    failure_msg = failure.text.strip().split('\n')[0]
                failures.append((full_name, failure_msg))
                continue
            
            # Check for error
            error = testcase.find('error')
            if error is not None:
                errors += 1
                error_msg = error.get('message', '')
                if not error_msg and error.text:
                    error_msg = error.text.strip().split('\n')[0]
                failures.append((full_name, f"ERROR: {error_msg}"))
                continue

            # If no failure, error, or skip, it passed
            passed += 1
        
        return {
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'failures': failures,
            'all_tests': all_tests,
            'total': len(all_tests)
        }
    
    def parse_all_suites(self) -> Dict[str, List[TestResult]]:
        """Parse XML files from all test suites.
        
        Discovers all test suites and processes their XML files to create
        TestResult objects for each test.
        
        Returns:
            Dictionary mapping suite names to lists of TestResult objects
        """
        all_results = {}
        test_suites = self.find_test_suites()
        
        print(f"Found {len(test_suites)} test suites: {test_suites}")
        
        for suite in test_suites:
            suite_path = os.path.join(self.base_folder, suite)
            print(f"Parsing {suite}...")
            
            results = self._parse_xml_files(suite_path)
            
            # Add suite info to each result
            for result in results:
                result.model_type = f"{suite}_{result.model_type}"
            
            all_results[suite] = results
            print(f"  {len(results)} results from {suite}")
        
        return all_results
    
    def _parse_xml_files(self, suite_path: str) -> List[TestResult]:
        """Parse all XML files in a suite folder.
        
        Processes all .xml files in the given suite directory and converts
        them into TestResult objects with proper error categorization.
        
        Args:
            suite_path: Path to the test suite directory
            
        Returns:
            List of TestResult objects for all tests in the suite
        """
        results = []
        xml_files = [f for f in os.listdir(suite_path) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(suite_path, xml_file)
            parsed_data = self.parse_pytest_xml(xml_path)
            
            if not parsed_data:
                continue
                
            # Process passed tests
            for test_full_name in parsed_data['all_tests']:
                # Extract just the test method name
                test_name = test_full_name.split('::')[-1] if '::' in test_full_name else test_full_name
                
                # Skip skipped tests
                if 'skip' in test_name.lower():
                    continue
                    
                model_type = self._extract_model_type(test_name)
                
                # Check if this test failed
                failed_test = next((f for f in parsed_data['failures'] if f[0] == test_full_name), None)
                
                if failed_test:
                    error_message = failed_test[1]
                    error_category = self._categorize_single_error(error_message)
                    
                    results.append(TestResult(
                        test_name=test_name,
                        model_type=model_type,
                        status="FAIL",
                        duration=0.0,  # Duration not available in this format
                        error_message=error_message[:200] + "..." if len(error_message) > 200 else error_message,
                        error_category=error_category,
                        error_categories=[error_category]
                    ))
                else:
                    # Passed test
                    results.append(TestResult(
                        test_name=test_name,
                        model_type=model_type,
                        status="PASS",
                        duration=0.0,
                        error_message=None,
                        error_category=None,
                        error_categories=None
                    ))
        
        return results
    

    
    def _extract_model_type(self, test_name: str) -> str:
        """Extract model type and configuration from test name.
        
        Parses test names to identify MoE model configurations, extracting
        the number of experts and hidden size.
        
        Args:
            test_name: Name of the test method
            
        Returns:
            String describing model type (e.g., "MoE_e8_h2048")
        """
        if "MoE" in test_name:
            experts_match = re.search(r'e(\d+)', test_name)
            hidden_match = re.search(r'h(\d+)', test_name)
            
            experts = experts_match.group(1) if experts_match else "X"
            hidden = hidden_match.group(1) if hidden_match else "X"
            
            return f"MoE_e{experts}_h{hidden}"
        
        return "MoE_unknown"
    
    def _extract_test_name_from_xml(self, content: str, filename: str) -> str:
        """Extract test name from XML content or use filename.
        
        Attempts to find the actual test name in XML content, falling back
        to generating a name from the filename if not found.
        
        Args:
            content: Raw XML content as string
            filename: Name of the XML file
            
        Returns:
            Extracted or generated test name
        """
        # Try to find test name in XML content
        test_match = re.search(r'test_fwdbwd_blockwisev2_MoE_[^\s<>"]+', content)
        if test_match:
            return test_match.group(0)
        
        # Fallback to filename-based test name
        return f"test_from_{filename.replace('.xml', '')}"
    
    def _extract_error_lines(self, content: str) -> List[str]:
        """Extract failure messages from XML content.
        
        Parses XML content to find failure messages, using both XML attributes
        and text content parsing methods.
        
        Args:
            content: Raw XML content as string
            
        Returns:
            List of error message strings
        """
        import re
        
        # Extract failure messages from XML
        failure_matches = re.findall(r'<failure message="([^"]+)"', content)
        
        if failure_matches:
            # Return the failure message(s) as single error lines
            return failure_matches
        
        # Fallback to old method if no failure messages found
        lines = content.split('\n')
        error_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('E   ') or stripped.startswith('E\t'):
                error_text = stripped[4:].strip()
                if error_text and not error_text.startswith('actual=') and not error_text.startswith('desired='):
                    error_lines.append(error_text)
        
        return error_lines
    
    def _extract_error_categories(self, error_lines: List[str]) -> List[str]:
        """Extract error categories from error lines.
        
        Categorizes multiple error messages and returns unique categories.
        
        Args:
            error_lines: List of error message strings
            
        Returns:
            List of unique error categories
        """
        categories = []
        
        for error_line in error_lines:
            category = self._categorize_single_error(error_line)
            if category not in categories:
                categories.append(category)
        
        return categories
    
    def _categorize_single_error(self, error_message: str) -> str:
        """Categorize a single error message.
        
        Analyzes error message text to classify it into predefined categories
        for better error tracking and analysis.
        
        Args:
            error_message: The error message to categorize
            
        Returns:
            String representing the error category:
                - blocksize_support_error
                - tolerance_assertion_error
                - xla_runtime_error
                - nki_compilation_error
                - runtime_error
                - assertion_error
                - index_error
                - generic_exception
                - uncategorized_error
        """
        if not error_message:
            return "unknown"
        
        error_lower = error_message.lower()
        
        if "only support block_size" in error_lower:
            return "blocksize_support_error"
        elif "not equal to tolerance" in error_lower and ("rtol=" in error_lower or "atol=" in error_lower):
            return "tolerance_assertion_error"
        elif "not equal to tolerance" in error_lower:
            return "tolerance_error"
        elif "xlaruntimeerror" in error_lower:
            return "xla_runtime_error"
        elif "assertionerror" in error_lower:
            return "assertion_error"
        elif "neuronxcc" in error_lower or "nki" in error_lower:
            return "nki_compilation_error"
        elif "runtimeerror" in error_lower:
            return "runtime_error"
        elif "indexerror" in error_lower:
            return "index_error"
        elif "exception" in error_lower:
            return "generic_exception"
        else:
            return "uncategorized_error"
    
    def _categorize_error(self, error_message: str) -> str:
        """Legacy error categorization method.
        
        Alternative error categorization with slightly different logic.
        Consider using _categorize_single_error instead.
        
        Args:
            error_message: The error message to categorize
            
        Returns:
            String representing the error category
        """
        if not error_message:
            return "unknown"
        
        error_lower = error_message.lower()
        
        if "only support block_size" in error_lower:
            return "blocksize_support_error"
        elif "xlaruntimeerror" in error_lower:
            return "xla_runtime_error"
        elif "assertionerror:" in error_lower and len(error_message.strip()) < 20:
            return "empty_assertion_error"
        elif "assertionerror" in error_lower:
            return "assertion_error"
        elif "neuronxcc" in error_lower or "nki" in error_lower:
            return "nki_compilation_error"
        elif "runtimeerror" in error_lower:
            return "runtime_error"
        elif "indexerror" in error_lower:
            return "index_error"
        elif "exception" in error_lower:
            return "generic_exception"
        else:
            return "uncategorized_error"
    
    def aggregate_results(self, suite_results: Dict[str, List[TestResult]]) -> List[TestResult]:
        """Aggregate all results into single list.
        
        Flattens the suite-organized results into a single list for
        easier processing and analysis.
        
        Args:
            suite_results: Dictionary mapping suite names to TestResult lists
            
        Returns:
            Flattened list of all TestResult objects
        """
        aggregated = []
        for suite, results in suite_results.items():
            aggregated.extend(results)
        return aggregated
    
    def _parse_test_parameters(self, test_name: str) -> Dict:
        """Extract test parameters from test name.
        
        Uses regex patterns to extract various test configuration parameters
        from the test method name.
        
        Args:
            test_name: Name of the test method
            
        Returns:
            Dictionary containing extracted parameters:
                - input_size, hidden_size, num_experts, topk, groups
                - expert_capacity, block_size, batch_size, sequence_length
                - precision, test_type, implementation
                - mesh_strategy, tensor_parallel, data_parallel
        """
        params = {}
        
        # Extract parameters using regex
        patterns = {
            'input_size': r'i(\d+)',
            'hidden_size': r'h(\d+)',
            'num_experts': r'e(\d+)',
            'topk': r'topk(\d+)',
            'groups': r'g(\d+)',
            'expert_capacity': r'ec(\d+)',
            'block_size': r'blocksize(\d+)',
            'batch_size': r'b(\d+)',
            'sequence_length': r's(\d+)',
            'precision': r'(bf16|fp16|fp32)',
            'test_type': r'test_(\w+)_',
            'implementation': r'_(blockwise\w*|gather\w*)_'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, test_name)
            if match:
                params[key] = match.group(1)
        
        # Extract parallelization info
        mesh_match = re.search(r'mesh(\w+)-(\d+)tp(\d+)', test_name)
        if mesh_match:
            params['mesh_strategy'] = mesh_match.group(1)
            params['tensor_parallel'] = mesh_match.group(2)
            params['data_parallel'] = mesh_match.group(3)
        
        return params
    
    def _parse_model_info(self, model_type: str) -> Dict:
        """Parse model type string into components.
        
        Extracts model information from strings like '12B_MoE_e8_h7168' or
        'deepseek-v3_MoE_e128_h2048'.
        
        Args:
            model_type: Model type string to parse
            
        Returns:
            Dictionary containing:
                - model_type: Base model type (e.g., '12B', 'deepseek-v3')
                - num_experts: Number of experts in MoE model
                - hidden_size: Hidden layer size
        """
        info = {}
        
        # Extract model type (12B, deepseek-v3, etc.) - everything before _MoE
        type_match = re.search(r'^([^_]+)_MoE', model_type)
        if type_match:
            info['model_type'] = type_match.group(1)
        
        # Extract experts and hidden size
        experts_match = re.search(r'e(\d+)', model_type)
        if experts_match:
            info['num_experts'] = int(experts_match.group(1))
        
        hidden_match = re.search(r'h(\d+)', model_type)
        if hidden_match:
            info['hidden_size'] = int(hidden_match.group(1))
        
        return info
    
    def upload_all_results(self, aggregated_results: List[TestResult], use_dev_table: bool = True) -> Dict[str, int]:
        """Upload all aggregated results to Scuba database.
        
        Converts TestResult objects into structured data format and uploads
        to Scuba for monitoring and dashboard visualization.
        
        Args:
            aggregated_results: List of TestResult objects to upload
            use_dev_table: If True, upload to development Scuba table
            
        Returns:
            Dictionary with upload statistics:
                - success: Number of successful uploads
                - failed: Number of failed uploads
        """
        stats = {"success": 0, "failed": 0}
        
        for result in aggregated_results:
            try:
                # Parse test parameters and model info
                test_params = self._parse_test_parameters(result.test_name)
                model_info = self._parse_model_info(result.model_type)
                
                data = {
                    "cfg": {
                        "TestDef": {
                            "ModelName": result.model_type,
                            "Identifier": f"FS_MoE_Dashboard",
                            "TestName": result.test_name,
                            "ModelType": model_info.get('model_type', 'unknown'),
                            "InferenceOrTraining": "Training",
                            "SequenceLength": int(test_params.get('sequence_length', 0)),
                            "HiddenSize": model_info.get('hidden_size', 0),
                            "Custom": {
                                "AdditionalData": {
                                    "String": result.error_category or "none",
                                },
                                "ModelTypeCustom": model_info.get('model_type', 'unknown'),
                                "NumExperts": model_info.get('num_experts', 0),
                                
                                "InputSize": int(test_params.get('input_size', 0)),
                                "TopK": int(test_params.get('topk', 0)),
                                "Groups": int(test_params.get('groups', 0)),
                                "ExpertCapacity": int(test_params.get('expert_capacity', 0)),
                                
                                "BlockSizeCustom": int(test_params.get('block_size', 0)),
                                "PrecisionCustom": test_params.get('precision', 'unknown'),
                                "TestTypeCustom": test_params.get('test_type', 'unknown'),
                                "Implementation": test_params.get('implementation', 'unknown'),
                                "MeshStrategy": test_params.get('mesh_strategy', 'unknown'),
                                "TensorParallel": int(test_params.get('tensor_parallel', 0)),
                                "DataParallel": int(test_params.get('data_parallel', 0))
                            }
                        },
                        "TestOpt": {"BatchSize": int(test_params.get('batch_size', 1))},
                        "Hardware": {"InstanceType": "trn2.48xlarge"},
                        "Software": {"VersionSet": self.event_id}
                    },
                    "kpi": {
                        "E2ETime": result.duration or 0.0,
                        "TestPassFail": result.status,
                        "TestFailMessage": result.error_message or "",
                    },
                    "misc": {
                        "User": os.environ.get("USER", "unknown"),
                        "ModelCategory": result.model_type,
                        "Timestamp": self.run_timestamp,
                    },
                    "artifact": {}
                }
                
                if self.scuba_available:
                    success = self.scuba.post_metrics(data, use_dev_table=use_dev_table, dry_run=self.dry_run)
                    
                    if success:
                        stats["success"] += 1
                        print(f"✓ {result.test_name} ({result.status}) - {result.model_type}")
                    else:
                        stats["failed"] += 1
                        print(f"✗ Failed: {result.test_name}")
                else:
                    stats["success"] += 1
                    print(f"{result.test_name} ({result.status}) - {result.model_type} [OFFLINE MODE]")
                    
            except Exception as e:
                stats["failed"] += 1
                print(f"Error: {result.test_name} - {e}")
        
        return stats
    
    def generate_batch_summary(self, suite_results: Dict[str, List[TestResult]]) -> Dict:
        """Generate comprehensive summary across all test suites.
        
        Creates detailed statistics and analysis of test results across
        all suites, including pass rates, error categorization, and
        model-specific statistics.
        
        Args:
            suite_results: Dictionary mapping suite names to TestResult lists
            
        Returns:
            Dictionary containing comprehensive summary:
                - total_tests, passed_tests, failed_tests, pass_rate
                - suite_stats: Per-suite statistics
                - model_stats: Per-model statistics
                - error_categories: Error distribution
                - timestamp, base_folder: Metadata
        """
        aggregated = self.aggregate_results(suite_results)
        
        if not aggregated:
            return {}
        
        total_tests = len(aggregated)
        passed_tests = sum(1 for r in aggregated if r.status == "PASS")
        failed_tests = total_tests - passed_tests
        
        # Suite-wise statistics
        suite_stats = {}
        for suite, results in suite_results.items():
            suite_total = len(results)
            suite_passed = sum(1 for r in results if r.status == "PASS")
            suite_failed = suite_total - suite_passed
            
            suite_stats[suite] = {
                "total": suite_total,
                "pass": suite_passed,
                "fail": suite_failed,
                "pass_rate": suite_passed / suite_total if suite_total > 0 else 0
            }
        
        # Model-wise statistics (across all suites)
        model_stats = {}
        for result in aggregated:
            if result.model_type not in model_stats:
                model_stats[result.model_type] = {"pass": 0, "fail": 0, "total": 0}
            
            model_stats[result.model_type]["total"] += 1
            if result.status == "PASS":
                model_stats[result.model_type]["pass"] += 1
            else:
                model_stats[result.model_type]["fail"] += 1
        
        # Error categorization
        error_categories = {}
        for result in aggregated:
            if result.error_category:
                error_categories[result.error_category] = error_categories.get(result.error_category, 0) + 1
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "suite_stats": suite_stats,
            "model_stats": model_stats,
            "error_categories": error_categories,
            "timestamp": self.run_timestamp,
            "base_folder": self.base_folder
        }

def main():
    """Main entry point for the batch parser script.
    
    Handles command-line arguments, orchestrates the parsing process,
    generates summaries, uploads results to Scuba, and saves detailed
    output to JSON file.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch parse integ.log files and upload to Scuba')
    parser.add_argument('--test_directory', '--base-folder', required=True, help='Test directory containing test suites')
    parser.add_argument('--event_id', required=True, help='Event ID from VERSION_SET')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Run in dry-run mode')
    parser.add_argument('--use-dev-table', action='store_true', default=False, help='Use dev Scuba table')
    
    args = parser.parse_args()
    
    print(f"Starting batch analysis of {args.test_directory}")
    
    batch_parser = BatchIntegLogParser(args.test_directory, event_id=args.event_id, dry_run=args.dry_run)
    
    # Parse all test suites
    suite_results = batch_parser.parse_all_suites()
    
    if not suite_results:
        print("No test suites found with integ.log files")
        return
    
    # Aggregate results
    aggregated_results = batch_parser.aggregate_results(suite_results)
    print(f"\nTotal aggregated results: {len(aggregated_results)}")
    
    # Generate summary
    summary = batch_parser.generate_batch_summary(suite_results)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH SUMMARY")
    print("="*60)
    
    if not summary:
        print("No test results found to summarize")
    else:
        print(f"Total tests across all suites: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Overall pass rate: {summary['pass_rate']:.2%}")
        
        print("\nSuite-wise results:")
        for suite, stats in summary['suite_stats'].items():
            print(f"  {suite}: {stats['pass']}/{stats['total']} ({stats['pass_rate']:.1%})")
        
        if summary['error_categories']:
            print("\nError categories:")
            for category, count in summary['error_categories'].items():
                print(f"  {category}: {count}")
    
    # Upload results
    print("\n" + "="*60)
    print("UPLOADING TO SCUBA")
    print("="*60)
    
    if aggregated_results:
        upload_stats = batch_parser.upload_all_results(aggregated_results, use_dev_table=args.use_dev_table)
        print(f"Upload completed: {upload_stats['success']} success, {upload_stats['failed']} failed")
    else:
        upload_stats = {"success": 0, "failed": 0}
        print("No results to upload")

if __name__ == "__main__":
    main()