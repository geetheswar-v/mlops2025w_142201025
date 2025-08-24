from utils import num_input

def add(a: float, b: float) -> float:
    return a + b

def main() -> None:
    print("Welcome to the Addition calculator!")
    numbers = num_input(2, "Enter a number")
    result = add(*numbers)
    print(f"The Addition is: {result}")

if __name__ == "__main__":
    main()