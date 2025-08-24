from utils import num_input, single_integer_input

def add(a: float, b: float) -> float:
    return a + b

def main() -> None:
    print("Welcome to the Addition calculator!")
    n = single_integer_input("How many numbers do you want to add?", 2)
    numbers = num_input(n, "Enter a number")
    result = sum(numbers)
    print(f"The Addition is: {result}")

if __name__ == "__main__":
    main()