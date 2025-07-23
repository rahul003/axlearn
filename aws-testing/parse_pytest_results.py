#!/usr/bin/env python3
"""
Parse pytest XML output and display test results summary.
Usage: python pytest_parser.py <xml_file>
"""

import xml.etree.ElementTree as ET
import sys
import os
import argparse
import json
import glob

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
    all_tests = []
    
    # Find all testcase elements
    testcases = root.findall('.//testcase')
    
    # total_filtered = 0
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

def compare_known_failures(args, cur_failures, suites):
    print("=" * 60)
    print("NEW FAILURES")
    have_new_failures = False
    with open(args.load_known_failures, 'r') as f:
        prev_failures = json.load(f)
    for suite in suites:
        if suite in prev_failures:
            prev_suite_failures = set(prev_failures[suite])
            cur_suite_failures = set(cur_failures.get(suite, []))
            new_failures = cur_suite_failures - prev_suite_failures
            have_new_failures = have_new_failures or bool(new_failures)
            if new_failures:
                print(f"New failures in {suite}:")
                for f in new_failures:
                    print(f"• {f}")
            else:
                print(f"No new failures in {suite}.")
        else:
            print(f"No previous failures recorded for {suite}.")
    if have_new_failures:
        print("=" * 60)
        print("There are new failures, please check the logs for details.")
        sys.exit(1)

def print_results(results, fname, matches, job_killed=False, print_exception=True):
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
            if error_msg and print_exception:
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
    parser.add_argument('--artifacts_dir', help='Path to pytest artifacts dir', required=True)
    parser.add_argument('--load_known_failures', help='Path to txt file containing known failures to compare and xfail', default=None)
    parser.add_argument('--print_exceptions', help='Path to txt file to save current failures for future runs', action='store_true')
    args = parser.parse_args()
    suites = [d for d in os.listdir(args.artifacts_dir) if os.path.isdir(os.path.join(args.artifacts_dir, d))]
    suites = sorted(suites)
    cur_failures = {}
    all_tests = {}
    some_suite_killed = False
    num_tests, num_passed, num_failed = 0, 0, 0
    for suite in suites:
        matches = glob.glob(os.path.join(args.artifacts_dir, suite, "integ_*.xml"))
        log_file = os.path.join(args.artifacts_dir, suite, "integ.log")
        job_killed = was_job_killed(log_file)
        some_suite_killed = some_suite_killed or job_killed
        if matches:
            results = {}
            for m in matches:
                m_result = parse_pytest_xml(m)
                num_tests += m_result['total']
                num_passed += m_result['passed']
                num_failed += m_result['failed']

                for k, v in m_result.items():
                    if k not in results:
                        results[k] = v
                    else:
                        results[k] += v
                if suite not in all_tests:
                    # TODO why were these duplicates in this list?, used set to get
                    # may no longer be an issue once we removed skipped tests from the list
                    all_tests[suite] = set(m_result['all_tests'])
                else:
                    all_tests[suite].update(set(m_result['all_tests']))

                for test_name, exception in m_result.get('failures', []):
                    if suite not in cur_failures:
                        cur_failures[suite] = [(test_name, exception)]
                    else:
                        cur_failures[suite].append((test_name, exception))
            cur_failures = {suite: sorted(tests) for suite, tests in cur_failures.items()}
            all_tests = {suite: sorted(list(tests)) for suite, tests in all_tests.items()}
            print_results(results, suite, matches, job_killed=job_killed, print_exception=args.print_exceptions)
        else:
            print("=" * 60)
            print(f"Results summary for {suite.upper()}")
            print(f"MISSING XML FILE")
            if job_killed:
                print(f"Some tests were KILLED")
    
    # total summary
    print("=" * 60)
    print("TOTAL summary")
    print(f"Total tests: {num_tests}")
    print(f"Passed: {num_passed}")
    print(f"Failed: {num_failed}")

    # Save current failures to file
    with open(os.path.join(args.artifacts_dir, 'failures_with_exceptions.txt'), 'w') as f:
        json.dump(cur_failures, f, indent=2)
    with open(os.path.join(args.artifacts_dir, 'failures.txt'), 'w') as f:
        cur_failures = {suite: [test[0] for test in tests] for suite, tests in cur_failures.items()}
        json.dump(cur_failures, f, indent=2)
    with open(os.path.join(args.artifacts_dir, 'all_tests.txt'), 'w') as f:
        json.dump(all_tests, f, indent=2)

    if args.load_known_failures:
        compare_known_failures(args, cur_failures, suites)
    if some_suite_killed:
        print("=" * 60)
        print("Some tests were KILLED, please check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()