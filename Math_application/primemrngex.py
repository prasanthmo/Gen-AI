# Function to check if a number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# Generate Set B (first 1000 odd numbers)
set_b = [num for num in range(1, 2000, 2)][:1000]  # First 1000 odd numbers

# Generate Set A (prime numbers up to the largest number in Set B)
upper_bound_b = max(set_b)
set_a = [num for num in range(2, upper_bound_b + 1) if is_prime(num)]

# Find the intersection of Set A and Set B
intersection_a_b = [num for num in set_b if num in set_a]

# Calculate P(A | B) = |A ∩ B| / |B|
probability_a_given_b = len(intersection_a_b) / len(set_b)

# Output the results

print(f"P(A | B) = {probability_a_given_b}")
