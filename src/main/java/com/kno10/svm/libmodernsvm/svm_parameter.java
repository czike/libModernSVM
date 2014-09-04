package com.kno10.svm.libmodernsvm;

import com.kno10.svm.libmodernsvm.svm.AbstractSVM;
import com.kno10.svm.libmodernsvm.svm.SVM_C;
import com.kno10.svm.libmodernsvm.svm.SVM_Nu;
import com.kno10.svm.libmodernsvm.svm.SVM_OneClass;
import com.kno10.svm.libmodernsvm.svm.SVR_Epsilon;
import com.kno10.svm.libmodernsvm.svm.SVR_Nu;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.LinearKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.PolynomialKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.RadialBasisKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.SigmoidKernelFunction;

public class svm_parameter implements Cloneable, java.io.Serializable {
	/* svm_type */
	public static final int C_SVC = 0;
	public static final int NU_SVC = 1;
	public static final int ONE_CLASS = 2;
	public static final int EPSILON_SVR = 3;
	public static final int NU_SVR = 4;

	/* kernel_type */
	public static final int LINEAR = 0;
	public static final int POLY = 1;
	public static final int RBF = 2;
	public static final int SIGMOID = 3;
	public static final int PRECOMPUTED = 4;

	public int svm_type;
	public int kernel_type;
	public int degree; // for poly
	public double gamma; // for poly/rbf/sigmoid
	public double coef0; // for poly/sigmoid

	// these are for training only
	public double cache_size; // in MB
	public double eps; // stopping criteria
	public double C; // for C_SVC, EPSILON_SVR and NU_SVR
	public int nr_weight; // for C_SVC
	public int[] weight_label; // for C_SVC
	public double[] weight; // for C_SVC
	public double nu; // for NU_SVC, ONE_CLASS, and NU_SVR
	public double p; // for EPSILON_SVR
	public int shrinking; // use the shrinking heuristics
	public int probability; // do probability estimates

	@Override
	public Object clone() {
		try {
			return super.clone();
		} catch (CloneNotSupportedException e) {
			return null;
		}
	}

	public AbstractSVM<svm_node[]> makeSVM(double Cp, double Cn) {
		switch (svm_type) {
		case svm_parameter.C_SVC:
			return new SVM_C<svm_node[]>(eps, shrinking, cache_size,
					makeKernelFunction(), Cp, Cn);
		case svm_parameter.NU_SVC:
			return new SVM_Nu<svm_node[]>(eps, shrinking, cache_size,
					makeKernelFunction(), nu);
		case svm_parameter.ONE_CLASS:
			return new SVM_OneClass<svm_node[]>(eps, shrinking, cache_size,
					makeKernelFunction(), nu);
		case svm_parameter.EPSILON_SVR:
			return new SVR_Epsilon<svm_node[]>(eps, shrinking, cache_size,
					makeKernelFunction(), C, p);
		case svm_parameter.NU_SVR:
			return new SVR_Nu<svm_node[]>(eps, shrinking, cache_size,
					makeKernelFunction(), C, nu);
		}
		throw new RuntimeException("Unknown SVM type");
	}

	public KernelFunction<svm_node[]> makeKernelFunction() {
		switch (kernel_type) {
		case LINEAR:
			return new LinearKernelFunction();
		case POLY:
			return new PolynomialKernelFunction(degree, gamma, coef0);
		case RBF:
			return new RadialBasisKernelFunction(gamma);
		case SIGMOID:
			return new SigmoidKernelFunction(gamma, coef0);
		case PRECOMPUTED:
			throw new RuntimeException("Incomplete support");
		default:
			throw new RuntimeException("Unknown kernel type: " + kernel_type);
		}
	}
}
