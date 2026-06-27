# Bad smells: God class, magic numbers, unused imports, mutable defaults, etc.
import math
import random
import datetime
import os
import sys
from typing import List, Dict, Any

# Global variables - bad smell
global_var = 42
global_list = []

# Class doing way too much (God class)
class DataProcessor:
    def __init__(self, data=[], config={}):  # Mutable default arguments - BAD!
        self.data = data
        self.config = config
        self.temp = None
        self.result = None
        self.counter = 0
        
    def process(self, x, y, z, flag):
        # Magic numbers everywhere
        if flag == 1:
            self.temp = x * 3.14159 + 42
        elif flag == 2:
            self.temp = y * 2.71828 - 7
        elif flag == 3:
            self.temp = z * 1.41421 / 13
        else:
            self.temp = 0
        
        # Duplicated code
        if self.temp > 50:
            print("Temperature high?")
            self.temp = self.temp * 0.9
            self.counter += 1
            global_list.append(self.temp)
        
        if self.temp < 0:
            print("Temperature negative!?")
            self.temp = self.temp * 1.1
            self.counter += 1
            global_list.append(self.temp)
        
        # Another duplicated block
        if self.temp > 200:
            self.temp = self.temp * 0.8
            self.counter += 1
            global_list.append(self.temp)
        
        # Inconsistent naming and style
        ReturnValue = self.temp  # Bad naming
        
        # Using global variable without need
        global global_var
        global_var = ReturnValue + global_var
        
        return ReturnValue
    
    # Long method with many responsibilities
    def calculate_and_validate_and_save_and_notify(self, a, b, c):
        # Way too many things in one method
        if a > b:
            result1 = a + b + c
            result2 = a - b
            result3 = result1 * result2
            for i in range(10):
                result3 = result3 + i
            # Lots of nested conditions
            if result3 > 0:
                if result3 < 100:
                    if a % 2 == 0:
                        result3 = result3 / 2
                    else:
                        result3 = result3 * 3 + 1
                elif result3 < 500:
                    result3 = result3 * 1.5
                else:
                    result3 = result3 * 0.5
            else:
                if result3 > -100:
                    result3 = abs(result3)
                else:
                    result3 = -result3
        
        # More duplication
        if b > c:
            result4 = b + c + a
            result5 = b - c
            result6 = result4 * result5
            for i in range(10):
                result6 = result6 + i
            if result6 > 0:
                if result6 < 100:
                    if b % 2 == 0:
                        result6 = result6 / 2
                    else:
                        result6 = result6 * 3 + 1
                elif result6 < 500:
                    result6 = result6 * 1.5
                else:
                    result6 = result6 * 0.5
            else:
                if result6 > -100:
                    result6 = abs(result6)
                else:
                    result6 = -result6
        
        # Saving to file (no error handling)
        with open("output.txt", "w") as f:
            f.write(str(result3))
            f.write(str(result6))
        
        # Notify (just prints)
        print("Done processing!")
        print("Result 1:", result3)
        print("Result 2:", result6)
        self.result = (result3, result6)
        return self.result

# Function with too many parameters
def process_data(x, y, z, a, b, c, d, e, f, g, flag=False, mode=1, verbose=True):
    # Using eval - extremely dangerous!
    expression = f"{x} + {y} * {z}"
    result = eval(expression)  # Bad smell: eval
    
    # Deep nesting
    if flag:
        if mode == 1:
            if verbose:
                if result > 0:
                    for i in range(10):
                        if i % 2 == 0:
                            if result < 100:
                                print("Even and small")
                            else:
                                print("Even and large")
                        else:
                            if result < 100:
                                print("Odd and small")
                            else:
                                print("Odd and large")
        elif mode == 2:
            # More nested logic
            if result < 50:
                print("Mode 2: low")
            else:
                print("Mode 2: high")
    else:
        # Inconsistent indentation (mixing spaces/tabs)
        print("Flag is False")
        print("This line is misaligned!")  # Bad indentation
    
    return result

# Not using exceptions properly - catching all
def risky_operation(data):
    try:
        # Something that might fail
        result = data / 0  # Will fail
    except:  # Bare except - BAD!
        # Silently fail
        pass  # Swallowing errors
    
    return None

# String concatenation in loop (inefficient)
def build_string(items):
    output = ""
    for item in items:
        output = output + item + ", "  # Bad: creates many string objects
    return output

# Unused variables and imports
unused_var = "I'm never used"
unused_list = [1, 2, 3, 4, 5]

# Type hints that are wrong
def get_name() -> int:  # Wrong return type hint
    return "John"  # Returns string, not int

# Class with public attributes (bad encapsulation)
class User:
    def __init__(self, name, age):
        self.name = name  # Public attribute
        self.age = age    # Public attribute
        self.__secret = "hidden"  # Not really private

# Inconsistent naming styles
user_name = "Alice"
USER_EMAIL = "alice@example.com"
userPhone = "555-1234"  # Mixed style

# Long line exceeding PEP 8
very_long_variable_name_that_is_unnecessarily_long = "This line is way too long and should be broken up but we're writing it anyway to demonstrate a bad smell in the code"

# Function with side effects
counter = 0
def increment_counter():
    global counter  # Using global
    counter += 1
    return counter

# Duplicated string literal
print("Error: Invalid input")
# ... later ...
print("Error: Invalid input")  # Duplicate string

# Dead code
def dead_function():
    # This function is never called
    return "I'm dead"

# Inconsistent return types
def inconsistent_return(x):
    if x > 0:
        return x
    elif x == 0:
        return 0
    # Returns None implicitly for negative values

# Deep inheritance chain (anti-pattern)
class A: pass
class B(A): pass
class C(B): pass
class D(C): pass
class E(D): pass
class F(E): pass
class G(F): pass
class H(G): pass
class I(H): pass
class J(I): pass  # Way too deep!

# Code smell: comments that lie
def add_numbers(a, b):
    # This multiplies two numbers (WRONG comment)
    return a + b

# Using 'is' for value comparison
x = 1000
if x is 1000:  # Should use ==, not 'is'
    print("x is 1000")

# Empty except block
try:
    risky_call()
except:
    pass  # Swallows all exceptions silently - BAD!

# Main execution with no guard
print("This runs even when imported!")
if __name__ == "__main__":
    # Code here...
    print("Running main...")

# Import inside function (unnecessary)
def lazy_import():
    import math
    return math.sqrt(16)