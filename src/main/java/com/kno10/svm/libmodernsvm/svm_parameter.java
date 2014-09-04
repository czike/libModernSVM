package com.kno10.svm.libmodernsvm;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.LinearKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.PolynomialKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.RadialBasisKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.SigmoidKernelFunction;
import com.kno10.svm.libmodernsvm.variants.AbstractSingleSVM;
import com.kno10.svm.libmodernsvm.variants.SVC_C;
import com.kno10.svm.libmodernsvm.variants.SVC_Nu;
import com.kno10.svm.libmodernsvm.variants.SVR_Epsilon;
import com.kno10.svm.libmodernsvm.variants.SVR_Nu;
import com.kno10.svm.libmodernsvm.variants.SVR_OneClass;

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

	public AbstractSingleSVM<svm_node[]> makeSVM() {
		switch (svm_type) {
		case svm_parameter.C_SVC:
			return new SVC_C<svm_node[]>(eps, shrinking, cache_size);
		case svm_parameter.NU_SVC:
			return new SVC_Nu<svm_node[]>(eps, shrinking, cache_size, nu);
		case svm_parameter.ONE_CLASS:
			return new SVR_OneClass<svm_node[]>(eps, shrinking, cache_size, nu);
		case svm_parameter.EPSILON_SVR:
			return new SVR_Epsilon<svm_node[]>(eps, shrinking, cache_size, C, p);
		case svm_parameter.NU_SVR:
			return new SVR_Nu<svm_node[]>(eps, shrinking, cache_size, C, nu);
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

	public String svm_check_parameter(svm_problem<svm_node[]> prob) {
		// svm_type

		if (svm_type != svm_parameter.C_SVC && svm_type != svm_parameter.NU_SVC
				&& svm_type != svm_parameter.ONE_CLASS
				&& svm_type != svm_parameter.EPSILON_SVR
				&& svm_type != svm_parameter.NU_SVR)
			return "unknown svm type";

		// kernel_type, degree

		if (kernel_type != svm_parameter.LINEAR
				&& kernel_type != svm_parameter.POLY
				&& kernel_type != svm_parameter.RBF
				&& kernel_type != svm_parameter.SIGMOID
				&& kernel_type != svm_parameter.PRECOMPUTED)
			return "unknown kernel type";

		if (gamma < 0)
			return "gamma < 0";

		if (degree < 0)
			return "degree of polynomial kernel < 0";

		// cache_size,eps,C,nu,p,shrinking

		if (cache_size <= 0)
			return "cache_size <= 0";

		if (eps <= 0)
			return "eps <= 0";

		if (svm_type == svm_parameter.C_SVC
				|| svm_type == svm_parameter.EPSILON_SVR
				|| svm_type == svm_parameter.NU_SVR)
			if (C <= 0)
				return "C <= 0";

		if (svm_type == svm_parameter.NU_SVC
				|| svm_type == svm_parameter.ONE_CLASS
				|| svm_type == svm_parameter.NU_SVR)
			if (nu <= 0 || nu > 1)
				return "nu <= 0 or nu > 1";

		if (svm_type == svm_parameter.EPSILON_SVR)
			if (p < 0)
				return "p < 0";

		if (shrinking != 0 && shrinking != 1)
			return "shrinking != 0 and shrinking != 1";

		if (probability != 0 && probability != 1)
			return "probability != 0 and probability != 1";

		if (probability == 1 && svm_type == svm_parameter.ONE_CLASS)
			return "one-class SVM probability output not supported yet";

		// check whether nu-svc is feasible

		if (svm_type == svm_parameter.NU_SVC) {
			int l = prob.l;
			int max_nr_class = 16;
			int nr_class = 0;
			int[] label = new int[max_nr_class];
			int[] count = new int[max_nr_class];

			int i;
			for (i = 0; i < l; i++) {
				int this_label = (int) prob.y[i];
				int j;
				for (j = 0; j < nr_class; j++)
					if (this_label == label[j]) {
						++count[j];
						break;
					}

				if (j == nr_class) {
					if (nr_class == max_nr_class) {
						max_nr_class *= 2;
						int[] new_data = new int[max_nr_class];
						System.arraycopy(label, 0, new_data, 0, label.length);
						label = new_data;

						new_data = new int[max_nr_class];
						System.arraycopy(count, 0, new_data, 0, count.length);
						count = new_data;
					}
					label[nr_class] = this_label;
					count[nr_class] = 1;
					++nr_class;
				}
			}

			for (i = 0; i < nr_class; i++) {
				int n1 = count[i];
				for (int j = i + 1; j < nr_class; j++) {
					int n2 = count[j];
					if (nu * (n1 + n2) / 2 > Math.min(n1, n2))
						return "specified nu is infeasible";
				}
			}
		}

		return null;
	}
}
