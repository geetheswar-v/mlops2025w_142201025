# Given a number n and a prompt message
# Take multiple numeric inputs (n times) from the user
def num_input(n: int, prompt: str) -> list[float]:
    numbers = []
    for i in range(n):
        while True:
            s = input(f"{prompt} ({i + 1}): ")
            if is_number(s, float):
                numbers.append(float(s))
                break
            print("Invalid input. Please enter a number.")
    return numbers

# Take a single integer input with minimum threshold
def single_integer_input(prompt: str, min_value: int = 1) -> int:
    while True:
        s = input(f"{prompt}: ")
        if is_number(s, int):
            value = int(s)
            if value >= min_value:
                return value
            print(f"Please enter a value greater than or equal to {min_value}.")
        print("Invalid input. Please enter a valid integer.")

# Check if a string is a valid number (int, float)
def is_number(s: str, type: int | float) -> bool:
    try:
        if type == int:
            int(s)
        else:
            float(s)
        return True
    except ValueError:
        return False
