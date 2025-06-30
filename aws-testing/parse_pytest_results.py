#!/usr/bin/env python3
"""
Parse pytest XML output and display test results summary.
Usage: python pytest_parser.py <xml_file>
"""

import xml.etree.ElementTree as ET
import sys
import os
import argparse


def parse_pytest_xml(xml_file):
    """Parse pytest XML file and extract test results."""
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
    
    # Find all testcase elements
    testcases = root.findall('.//testcase')
    
    for testcase in testcases:
        test_name = testcase.get('name', 'Unknown')
        classname = testcase.get('classname', '')
        full_name = f"{classname}::{test_name}" if classname else test_name
        
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
        
        # Check for skipped
        skipped_elem = testcase.find('skipped')
        if skipped_elem is not None:
            skipped += 1
            continue
        
        # If no failure, error, or skip, it passed
        passed += 1
    
    return {
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'skipped': skipped,
        'failures': failures,
        'total': len(testcases)
    }


def print_results(results, fname, matches, job_killed=False):
    """Print formatted test results."""
    if results is None:
        return
    
    print("=" * 60)
    print(f"Results summary for {fname.upper()}")
    if job_killed:
        print(f"Some tests were KILLED")
    for m in matches:
        print(f"  {m}")
    
    print(f"Total tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Errors: {results['errors']}")
    print(f"Skipped: {results['skipped']}")
    print()
    
    if results['failures']:
        print("FAILURES:")
        print("-" * 40)
        for test_name, error_msg in results['failures']:
            print(f"• {test_name}")
            if error_msg:
                print(f"  {error_msg}")
            print()
    else:
        print("No failures! 🎉")

def was_job_killed(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            return "Killed" in content
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Parse pytest XML output')
    parser.add_argument('artifacts_dir', help='Path to pytest artifacts dir')
    
    args = parser.parse_args()
    suites = [d for d in os.listdir(args.artifacts_dir) if os.path.isdir(os.path.join(args.artifacts_dir, d))]
    import glob
    for suite in suites:
        
        matches = glob.glob(os.path.join(args.artifacts_dir, suite, "integ_*.xml"))
        log_file = os.path.join(args.artifacts_dir, suite, "integ.log")
        job_killed = was_job_killed(log_file)
        if matches:
            results = {}
            for m in matches:
                m_result = parse_pytest_xml(m)
                for k, v in m_result.items():
                    if k not in results:
                        results[k] = v
                    else:
                        results[k] += v
            print_results(results, suite, matches, job_killed=job_killed)
        else:
            print("=" * 60)
            print(f"Results summary for {suite.upper()}")
            print(f"MISSING XML FILE")
            if job_killed:
                print(f"Some tests were KILLED")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pytest_parser.py ARTIFACTS_DIR")
        print("Example: python pytest_parser.py test_results.xml")
        sys.exit(1)
    main()