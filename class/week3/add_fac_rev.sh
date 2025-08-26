#!/bin/bash

# Adding Two Number and Factorial
echo "Given Two number A and B, Results SUM(A+B), Fac(A) and Fac(B)"

# Inputs
read -p "a: " a
read -p "b: " b

echo "Given, A = ${a} and B = ${b}"

# Sum of two numbers
sum=$((a + b))

# Factorial Function
factorial() {
  local n=$1
  if (( n < 0 )); then
    echo "negative number $n"
    return
  fi
  if (( n == 0 || n == 1 )); then
    echo 1
    return
  fi
  local fact=1
  for (( i=2; i<=n; i++ )); do
    fact=$((fact * i))
  done
  echo $fact
}

# Result Echo
echo "${a} + ${b} = ${sum}"
echo "Factorial of A (${a}) = $(factorial "$a")"
echo "Factorial of B (${b}) = $(factorial "$b")"
