# Assignment 2 - Calculator Usage Guide

## Overview
This is a simple addition calculator that allows users to add multiple numbers together. The calculator takes user input for the number of values to add and then prompts for each number individually.

## Features
- Add multiple numbers (minimum 2)
- Input validation for numeric values
- Support for both integers and floating-point numbers
- User-friendly prompts and error messages

## Prerequisites
- Python 3.12 or higher
- No external dependencies required

## Usage

### Running the Calculator

#### Method 1: Using the provided run script (Recommended)
```bash
# Make the script executable
chmod +x run.sh

# Run the calculator
./run.sh
```

#### Method 2: Direct Python execution
```bash
# Navigate to the src directory
cd src

# Run with Python
python3 calculator.py
# or
python calculator.py
```

#### Method 3: Using uv
```bash
# From the assignment-2 directory
uv run python src/calculator.py
```

### Example Usage

1. **Start the calculator:**
   ```
   Welcome to the Addition calculator!
   ```

2. **Specify how many numbers to add:**
   ```
   How many numbers do you want to add?: 3
   ```

3. **Enter the numbers:**
   ```
   Enter a number (1): 10.5
   Enter a number (2): 25.3
   Enter a number (3): 14.2
   ```

4. **View the result:**
   ```
   The Addition is: 50.0
   ```

### Input Validation

The calculator includes robust input validation:

- **Number count validation:** Must be at least 2 numbers
- **Numeric validation:** Only accepts valid integers or floating-point numbers
- **Error handling:** Clear error messages for invalid inputs

#### Example of validation in action:
```
How many numbers do you want to add?: 1
Please enter a value greater than or equal to 2.
How many numbers do you want to add?: abc
Invalid input. Please enter a valid integer.
How many numbers do you want to add?: 3

Enter a number (1): abc
Invalid input. Please enter a number.
Enter a number (1): 10.5
Enter a number (2): 25
Enter a number (3): 30.5
The Addition is: 66.0
```
