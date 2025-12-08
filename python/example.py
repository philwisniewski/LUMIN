#!/usr/bin/env python3
"""
Example usage of LUMIN Python bindings
"""
import numpy as np
import lumin

def main():
    print("LUMIN Python Bindings Example")
    print("=" * 40)
    
    # Create matrices
    print("\n1. Creating matrices:")
    A = lumin.Matrix(3, 3)
    B = lumin.Matrix(3, 3)
    
    # Fill with values
    for i in range(3):
        for j in range(3):
            A[i, j] = i * 3 + j + 1
            B[i, j] = (i * 3 + j + 1) * 2
    
    print(f"A:\n{A}")
    print(f"\nB:\n{B}")
    
    # Matrix operations
    print("\n2. Matrix operations:")
    C = A + B
    print(f"A + B:\n{C}")
    
    D = A * B
    print(f"\nA * B:\n{D}")
    
    E = A.transpose()
    print(f"\nA.transpose():\n{E}")
    
    dot = A.dot(B)
    print(f"\nA.dot(B) = {dot}")
    
    # Scalar multiplication
    print("\n3. Scalar multiplication:")
    F = A * 2.5
    print(f"A * 2.5:\n{F}")
    
    G = 3.0 * A
    print(f"\n3.0 * A:\n{G}")
    
    # NumPy integration
    print("\n4. NumPy integration:")
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    print(f"NumPy array:\n{arr}")
    
    M = lumin.Matrix(arr)
    print(f"\nConverted to LUMIN Matrix:\n{M}")
    
    result = M.to_numpy()
    print(f"\nConverted back to NumPy:\n{result}")
    
    # Random matrix
    print("\n5. Random matrix:")
    R = lumin.Matrix.random_int(5, 5, max_value=100)
    print(f"Random 5x5 matrix:\n{R}")
    
    # Backend selection
    print("\n6. Backend selection:")
    try:
        lumin.set_backend("cpu")
        print("Set backend to CPU")
    except Exception as e:
        print(f"Error setting backend: {e}")
    
    try:
        lumin.set_backend("openmp")
        print("Set backend to OpenMP")
    except Exception as e:
        print(f"OpenMP not available: {e}")
    
    try:
        lumin.set_backend("cuda")
        print("Set backend to CUDA")
    except Exception as e:
        print(f"CUDA not available: {e}")

if __name__ == "__main__":
    main()

