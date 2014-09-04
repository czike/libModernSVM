package com.kno10.svm.libmodernsvm;

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
abstract class QMatrix {
	abstract float[] get_Q(int column, int len);
	abstract double[] get_QD();
	abstract void swap_index(int i, int j);
}