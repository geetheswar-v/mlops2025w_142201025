# Given a number n and a prompt message
# Take multiple numeric inputs (n times) from the user
def num_input(n: int, prompt: str) -> list[float]:
    numbers = []
    for i in range(n):
        while True:
            try:
                numbers.append(float(input(f'{prompt} ({i + 1}): ')))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    return numbers
