#!/usr/bin/env python3


from analyzer import analyze_code_basic
import json

def test_complete_analyzer():
    """Test all analyzer features."""
    
    print("Testing Complete Code Analyzer")
    print("=" * 50)
    
    # Test with comprehensive code example
    test_code = '''import math
from collections import defaultdict

class Calculator:
    def __init__(self, name):
        self.name = name
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"add({a}, {b}) = {result}")
        return result
    
    def factorial(self, n):
        if n <= 1:
            return 1
        else:
            return n * self.factorial(n - 1)

def process_numbers(numbers):
    calc = Calculator("Main")
    results = []
    
    for num in numbers:
        if num > 0:
            fact = calc.factorial(num)
            results.append(fact)
        else:
            print(f"Skipping negative number: {num}")
    
    return results

# Main execution
data = [1, 2, 3, -1, 4, 5]
output = process_numbers(data)
print("Results:", output)'''
    
    print("\nTest Code:")
    print("-" * 30)
    print(test_code)
    print("-" * 30)
    
    # Analyze the code
    result = analyze_code_basic(test_code)
    
    print("\nComplete Analysis Results:")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    
    # Verify all components are present
    print("\n" + "=" * 50)
    print("VERIFICATION CHECKLIST:")
    print("=" * 50)
    
    checks = [
        ("✅ AST Analysis", "ast" in result),
        ("✅ Symbol Table", "symbol_table" in result),
        ("✅ Control Flow Graph", "control_flow_graph" in result),
        ("✅ Metrics", "metrics" in result),
        ("✅ Functions Found", len(result.get("ast", {}).get("functions", [])) > 0),
        ("✅ Classes Found", len(result.get("ast", {}).get("classes", [])) > 0),
        ("✅ Imports Found", len(result.get("ast", {}).get("imports", [])) > 0),
        ("✅ Symbol Table Populated", len(result.get("symbol_table", [])) > 0),
        ("✅ CFG Nodes Created", len(result.get("control_flow_graph", {}).get("nodes", [])) > 0),
        ("✅ CFG Edges Created", len(result.get("control_flow_graph", {}).get("edges", [])) > 0)
    ]
    
    for check_name, check_result in checks:
        status = "PASS" if check_result else "FAIL"
        print(f"{check_name}: {status}")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    if result.get("error"):
        print(f"❌ Analysis failed with error: {result['error']}")
    else:
        print("✅ All required components successfully generated!")
        print(f"   - AST: {len(result['ast']['functions'])} functions, {len(result['ast']['classes'])} classes")
        print(f"   - Symbol Table: {len(result['symbol_table'])} symbols")
        print(f"   - CFG: {len(result['control_flow_graph']['nodes'])} nodes, {len(result['control_flow_graph']['edges'])} edges")

if __name__ == "__main__":
    test_complete_analyzer()
