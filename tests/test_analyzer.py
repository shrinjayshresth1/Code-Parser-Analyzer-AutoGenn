#!/usr/bin/env python3
"""
Test script for AutoGen + GenAI Enhanced Code Analyzer
Demonstrates the functionality without requiring API keys
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from code_analyzer import analyze_code, CodeAnalyzer

# Test code sample
test_code = '''
def bubble_sort(arr):
    """Sort array using bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def main():
    numbers = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", numbers)
    
    sorted_numbers = bubble_sort(numbers.copy())
    print("Sorted array:", sorted_numbers)

if __name__ == "__main__":
    main()
'''

print("=" * 80)
print("AutoGen + GenAI Enhanced Code Analyzer - Test Run")
print("=" * 80)
print("Testing with bubble sort algorithm...")
print("\nINPUT CODE:")
print("-" * 40)
print(test_code)
print("-" * 40)

print("\nRUNNING ANALYSIS (Basic Mode - No API Key Required)...")
print("=" * 80)

# Run analysis without AI (basic mode)
result = analyze_code(test_code)
print("STRUCTURED ANALYSIS RESULT:")
print(result)

print("\n" + "=" * 80)
print("Test completed successfully!")
print("\nTo test with AutoGen + AI enhancement:")
print("1. Get an API key from OpenRouter (https://openrouter.ai) or Groq (https://groq.com)")
print("2. Run: python code_analyzer.py")
print("3. Choose 'y' for AI enhancement and provide your API key")
print("=" * 80)
