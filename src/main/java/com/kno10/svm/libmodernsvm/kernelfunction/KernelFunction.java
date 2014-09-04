package com.kno10.svm.libmodernsvm.kernelfunction;

public interface KernelFunction<T> {
	double kernel_function(T i, T j);
}
