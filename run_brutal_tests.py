#!/usr/bin/env python3
"""
Master Test Runner for Auto-NVIS Brutal Test Suite

Runs all unit tests and integration tests with comprehensive reporting.
Tracks CPU usage, memory usage, and test execution times.
"""

import subprocess
import sys
import time
import psutil
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_subheader(text):
    """Print subsection header"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{'-'*80}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{'-'*80}{Colors.ENDC}\n")


def run_test_file(test_file, test_name):
    """Run a single test file and return results"""
    print_subheader(f"Running: {test_name}")

    # Get initial CPU and memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.time()

    # Run pytest using python -m to ensure venv packages are used
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', str(test_file), '-v', '--tb=short', '-s'],
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    # Get final memory
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_delta = mem_after - mem_before

    # Parse results
    passed = result.stdout.count(' PASSED')
    failed = result.stdout.count(' FAILED')
    errors = result.stdout.count(' ERROR')

    # Print results
    if result.returncode == 0:
        print(f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}✗ FAILED{Colors.ENDC}")

    print(f"  Tests: {passed} passed, {failed} failed, {errors} errors")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory delta: {mem_delta:+.1f} MB")

    # Show output if there were failures
    if failed > 0 or errors > 0:
        print(f"\n{Colors.WARNING}Output:{Colors.ENDC}")
        print(result.stdout[-2000:])  # Last 2000 chars

    return {
        'name': test_name,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'time': elapsed,
        'mem_delta': mem_delta,
        'success': result.returncode == 0
    }


def main():
    """Main test runner"""
    print_header("AUTO-NVIS BRUTAL TEST SUITE")

    print(f"{Colors.BOLD}System Information:{Colors.ENDC}")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  Total RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"  Python: {sys.version}")

    # List of all test files
    test_suite = [
        # Unit tests
        ('tests/unit/test_config.py', 'Configuration'),
        ('tests/unit/test_geodesy.py', 'Geodesy'),
        ('tests/unit/test_state_vector.py', 'State Vector'),
        ('tests/unit/test_message_queue.py', 'Message Queue'),
        ('tests/unit/test_mode_controller.py', 'Mode Controller'),
        ('tests/unit/test_data_validator.py', 'Data Validator'),
        ('tests/unit/test_goes_xray.py', 'GOES X-ray Ingestion'),
        ('tests/unit/test_gnss_tec.py', 'GNSS-TEC Ingestion'),
        ('tests/unit/test_nvis_quality.py', 'NVIS Quality Assessment'),
        ('tests/unit/test_nvis_aggregation.py', 'NVIS Aggregation'),
        ('tests/unit/test_information_gain.py', 'Information Gain'),
        ('tests/unit/test_optimal_placement.py', 'Optimal Placement'),
        ('tests/unit/test_propagation_service.py', 'Propagation Service'),

        # Integration tests
        ('tests/integration/test_nvis_end_to_end.py', 'NVIS End-to-End'),
        ('tests/integration/test_nvis_performance.py', 'NVIS Performance'),
        ('tests/integration/test_nvis_validation.py', 'NVIS Validation'),
        ('tests/integration/test_brutal_system_integration.py', 'BRUTAL SYSTEM INTEGRATION'),
    ]

    results = []
    total_start = time.time()

    # Run unit tests
    print_header("UNIT TESTS")
    for test_file, test_name in test_suite[:13]:
        test_path = Path(test_file)
        if test_path.exists():
            result = run_test_file(test_path, test_name)
            results.append(result)
        else:
            print(f"{Colors.WARNING}⚠ SKIP: {test_name} (file not found){Colors.ENDC}")

    # Run integration tests
    print_header("INTEGRATION TESTS")
    for test_file, test_name in test_suite[13:]:
        test_path = Path(test_file)
        if test_path.exists():
            result = run_test_file(test_path, test_name)
            results.append(result)
        else:
            print(f"{Colors.WARNING}⚠ SKIP: {test_name} (file not found){Colors.ENDC}")

    total_elapsed = time.time() - total_start

    # Summary
    print_header("TEST SUITE SUMMARY")

    total_passed = sum(r['passed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    total_tests = total_passed + total_failed + total_errors

    successful_suites = sum(1 for r in results if r['success'])
    failed_suites = len(results) - successful_suites

    print(f"{Colors.BOLD}Overall Results:{Colors.ENDC}")
    print(f"  Test suites: {successful_suites} passed, {failed_suites} failed")
    print(f"  Total tests: {total_tests}")
    print(f"  {Colors.OKGREEN}✓ Passed: {total_passed}{Colors.ENDC}")
    print(f"  {Colors.FAIL}✗ Failed: {total_failed}{Colors.ENDC}")
    print(f"  {Colors.WARNING}⚠ Errors: {total_errors}{Colors.ENDC}")
    print(f"\n{Colors.BOLD}Performance:{Colors.ENDC}")
    print(f"  Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
    print(f"  Average per suite: {total_elapsed/len(results):.2f}s")

    # Slowest tests
    print(f"\n{Colors.BOLD}Slowest test suites:{Colors.ENDC}")
    sorted_results = sorted(results, key=lambda x: x['time'], reverse=True)
    for r in sorted_results[:5]:
        print(f"  {r['name']}: {r['time']:.2f}s")

    # CPU intensive tests (longest runtime = most CPU intensive)
    print(f"\n{Colors.BOLD}Most CPU intensive:{Colors.ENDC}")
    for r in sorted_results[:3]:
        print(f"  {r['name']}: {r['time']:.2f}s")

    # Memory intensive tests
    print(f"\n{Colors.BOLD}Most memory intensive:{Colors.ENDC}")
    sorted_by_mem = sorted(results, key=lambda x: x['mem_delta'], reverse=True)
    for r in sorted_by_mem[:3]:
        print(f"  {r['name']}: {r['mem_delta']:+.1f} MB")

    # Final verdict
    print()
    if total_failed == 0 and total_errors == 0:
        print(f"{Colors.OKGREEN}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}{'ALL TESTS PASSED!':^80}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.FAIL}{Colors.BOLD}{'SOME TESTS FAILED':^80}{Colors.ENDC}")
        print(f"{Colors.FAIL}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
