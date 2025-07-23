# Example collaborative coding project

def fibonacci(n):
    """Generate fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence

def main():
    print("🤖 Multi-Agent Collaborative Coding Demo")
    print("This file will be edited by multiple agents in real-time!")
    
    # Generate and display fibonacci sequence
    n = 10
    fib_sequence = fibonacci(n)
    print(f"\nFibonacci sequence ({n} terms): {fib_sequence}")

if __name__ == "__main__":
    main()
