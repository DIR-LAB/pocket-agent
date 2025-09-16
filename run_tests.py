#!/usr/bin/env python3
"""
Utility script for running pocket agent tests with uv.

This script provides an easy way to run the pocket agent test suite
using uv, with options for different test configurations and reporting.

Usage:
    python run_tests.py [options]
    
Options:
    --quick         Run a quick subset of tests for development
    --verbose       Enable verbose test output
    --coverage      Run tests with coverage reporting
    --parallel      Run tests in parallel (experimental)
    --help          Show this help message

Examples:
    python run_tests.py --verbose
    python run_tests.py --coverage
    python run_tests.py --quick
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd: list[str], description: str = "") -> int:
    """Run a command and return the exit code."""
    if description:
        print(f"\n🔧 {description}")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure 'uv' is installed and available in your PATH")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Test run interrupted by user")
        return 1


def check_uv_available() -> bool:
    """Check if uv is available in the system."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def setup_test_environment():
    """Set up the test environment and install dependencies."""
    print("📦 Setting up test environment...")
    
    # Install test dependencies using uv
    cmd = ["uv", "sync", "--group", "test"]
    return run_command(cmd, "Installing test dependencies")


def run_tests(
    verbose: bool = False,
    coverage: bool = False,
    parallel: bool = False
) -> int:
    """Run all tests with the specified options."""
    
    # Base pytest command through uv
    cmd = ["uv", "run", "pytest"]
    
    # Run all tests in the tests/ directory
    cmd.extend(["tests/"])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage reporting
    if coverage:
        cmd.extend(["--cov=pocket_agent", "--cov-report=html", "--cov-report=term"])
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])  # Requires pytest-xdist
    
    # Add asyncio mode for async tests
    cmd.extend(["--asyncio-mode=auto"])
    
    # Add markers
    cmd.extend(["-m", "not slow"])  # Skip slow tests by default
    
    return run_command(cmd, "Running all tests")


def run_quick_tests() -> int:
    """Run a quick subset of tests for development."""
    print("🚀 Running quick test suite...")
    
    cmd = [
        "uv", "run", "pytest",
        # Quick unit tests
        "tests/test_agent.py::TestAgentConfig::test_agent_config_creation",
        "tests/test_agent.py::TestPocketAgent::test_agent_initialization",
        # Quick integration test
        "tests/test_integration.py::TestAgentIntegration::test_agent_initialization_end_to_end",
        "-v", "--asyncio-mode=auto"
    ]
    
    return run_command(cmd, "Quick test suite")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run pocket agent tests with uv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Test selection options
    parser.add_argument("--quick", action="store_true",
                       help="Run a quick subset of tests for development")
    
    # Test configuration options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose test output")
    parser.add_argument("--coverage", "-c", action="store_true",
                       help="Run tests with coverage reporting")
    parser.add_argument("--parallel", "-p", action="store_true",
                       help="Run tests in parallel (experimental)")
    
    args = parser.parse_args()
    
    # Check if uv is available
    if not check_uv_available():
        print("❌ Error: 'uv' is not available in your PATH")
        print("Please install uv: https://github.com/astral-sh/uv")
        return 1
    
    print("🧪 Pocket Agent Test Runner")
    print("=" * 40)
    
    # Setup test environment
    exit_code = setup_test_environment()
    if exit_code != 0:
        print("❌ Failed to set up test environment")
        return exit_code
    
    # Run tests based on selected options
    if args.quick:
        exit_code = run_quick_tests()
    else:  # Default: run all tests
        print("🎯 Running all tests...")
        exit_code = run_tests(args.verbose, args.coverage, args.parallel)
    
    # Report results
    if exit_code == 0:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        print("Check the output above for details")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
