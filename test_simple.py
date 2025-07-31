#!/usr/bin/env python3
"""Test script for the simple analyzer."""

from analyzer import analyze_code_basic
import json

def test_analyzer():
    """Test the basic analyzer functionality."""
    
    print("Testing Simple Code Analyzer")
    print("=" * 40)
    
    # Test 1: Simple function
    print("\nTest 1 - Simple function:")
    code1 = """def hello(name):
    return "Hello " + name"""
    
    result1 = analyze_code_basic(code1)
    print(json.dumps(result1, indent=2))
    
    # Test 2: Variables and assignments
    print("\nTest 2 - Variables:")
    code2 = """x = 10
y = 20
z = x + y
name = "Python\""""
    
    result2 = analyze_code_basic(code2)
    print(json.dumps(result2, indent=2))
    
    # Test 3: Class with methods
    print("\nTest 3 - Class with methods:")
    code3 = """class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, a, b):
        return a * b"""
    
    result3 = analyze_code_basic(code3)
    print(json.dumps(result3, indent=2))
    
    # Test 4: Complex example
    print("\nTest 4 - Complex example:")
    code4 = """import math

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def circumference(self):
        return 2 * math.pi * self.radius

def create_circles(radii):
    circles = []
    for r in radii:
        circle = Circle(r)
        circles.append(circle)
    return circles

# Usage
radii_list = [1, 2, 3, 4, 5]
my_circles = create_circles(radii_list)"""
    
    result4 = analyze_code_basic(code4)
    print(json.dumps(result4, indent=2))
    
    # Test 5: Syntax error handling
    print("\nTest 5 - Syntax error handling:")
    code5 = "def broken_function(:"
    
    result5 = analyze_code_basic(code5)
    print(json.dumps(result5, indent=2))
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_analyzer()
