#!/bin/bash

# Adding Two Number
echo "Given Two number A and B, Results SUM(A+B)"

# Inputs
read -p "a: " a
read -p "b: " b

echo "Given, A = ${a} and B = ${b}"

# Sum of two numbers
sum=$((a + b))

# Result Echo
echo "${a} + ${b} = ${sum}"
