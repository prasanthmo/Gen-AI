def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes(limit):
    try:
        primes = []
        for num in range(1, limit + 1):
            if is_prime(num):
                primes.append(num)
        return primes
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    prime_numbers = find_primes(100)
    print("Prime numbers between 1 and 100:", prime_numbers)

