#import modules
import numpy as np

#Simulation any random rectangular matrix A
A = np.random.rand(4,5)
print(f"\nRadom matrix A\n{'--'*30}")
print(A)

#Rank and Trace of A
range_A = np.linalg.matrix_rank (A)
trace_A = np.trace (A)
print(f"\nRank and trace of matrix A\n{'--'*30}")
print(f"Rank matrix: { range_A }")
print(f"Trace matrix: { trace_A }")

#Determinant of A
#det_A = np.linalg.det (A)
#print(f"Trace matrix: {det_A}")
print(f"\nDeterminant of matrix A\n{'--'*30}")
print("The determinants of a matrix only apply to square matrices, so this point cann't be calculated.")

#Can you invert A? How?
#inv_A = np.linalg.inv(A)
pinv_A = np.linalg.pinv(A)

print(f"\nInvert of matrix A\n{'--'*30}")
print(f"A invert:  { pinv_A }")
#print(f"A invert:  { inv_A }")
print("\n The matrix A cann't be inverted because it is not square and has no determinant. The pinv function computes a pseudo inverse with computacional loss.")

#How are eigenvalues and eigenvectors of A’A and AA’ related? 
#What interesting differences can you notice between both? 
A_transpose = np.transpose(A)       
eigen_valor_At_A, eigen_vector_At_A = np.linalg.eig (A_transpose @ A)
eigen_valor_A_At, eigen_vector_A_At = np.linalg.eig (A @ A_transpose)

print(f"\nDifference between eigenvalues of At.A and A.At\n{'--'*30}")
print ("Eigenvalue At.A: ", eigen_valor_At_A, "\n")
print ("Eigenvalue A.At: ", eigen_valor_A_At)
print("\nFor At.A it can be seen that there is a very small different additional value; this may be due to rounding errors. \n \
      They are also presented in a different order but are the same values.")


print(f"\nDifference between eigenvectors of At.A and A.At\n{'--'*30}")
print ("Eigenvectors At.A: ", eigen_vector_At_A, "\n")
print ("Eigenvectors A.At: ", eigen_vector_A_At)
print("\nEigenvectors have different magnitudes and directions. There is also a different quantity due to the different dimensions of the matrices.")