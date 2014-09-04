package com.kno10.svm.libmodernsvm.kernelfunction;

/**
 * Interface for kernel functions.
 * 
 * @param <T>
 *            Data type
 */
public interface KernelFunction<T> {
	/**
	 * Compute the kernel function for two objects.
	 * 
	 * @param x First object
	 * @param y Second object
	 * @return Similarity
	 */
	double similarity(T x, T y);
}
