package com.kno10.svm.libmodernsvm;

import java.util.Random;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.variants.AbstractSingleSVM;

public class svm<T> {
	static final Logger LOG = Logger.getLogger(svm.class.getName());

	//
	// construct and solve various formulations
	//
	public static final int LIBSVM_VERSION = 318;
	public static final Random rand = new Random();

	// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
	private static void sigmoid_train(int l, double[] dec_values,
			double[] labels, double[] probAB) {
		double A, B;
		double prior1 = 0, prior0 = 0;

		for (int i = 0; i < l; i++)
			if (labels[i] > 0)
				++prior1;
			else
				++prior0;

		int max_iter = 100; // Maximal number of iterations
		double min_step = 1e-10; // Minimal step taken in line search
		double sigma = 1e-12; // For numerically strict PD of Hessian
		double eps = 1e-5;
		double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
		double loTarget = 1 / (prior0 + 2.0);
		double[] t = new double[l];
		double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
		double newA, newB, newf, d1, d2;

		// Initial Point and Initial Fun Value
		A = 0.0;
		B = Math.log((prior0 + 1.0) / (prior1 + 1.0));
		double fval = 0.0;

		for (int i = 0; i < l; i++) {
			t[i] = (labels[i] > 0) ? hiTarget : loTarget;
			fApB = dec_values[i] * A + B;
			if (fApB >= 0)
				fval += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
			else
				fval += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
		}
		for (int iter = 0; iter < max_iter; iter++) {
			// Update Gradient and Hessian (use H' = H + sigma I)
			h11 = sigma; // numerically ensures strict PD
			h22 = sigma;
			h21 = 0.0;
			g1 = 0.0;
			g2 = 0.0;
			for (int i = 0; i < l; i++) {
				fApB = dec_values[i] * A + B;
				if (fApB >= 0) {
					p = Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
					q = 1.0 / (1.0 + Math.exp(-fApB));
				} else {
					p = 1.0 / (1.0 + Math.exp(fApB));
					q = Math.exp(fApB) / (1.0 + Math.exp(fApB));
				}
				d2 = p * q;
				h11 += dec_values[i] * dec_values[i] * d2;
				h22 += d2;
				h21 += dec_values[i] * d2;
				d1 = t[i] - p;
				g1 += dec_values[i] * d1;
				g2 += d1;
			}

			// Stopping Criteria
			if (Math.abs(g1) < eps && Math.abs(g2) < eps)
				break;

			// Finding Newton direction: -inv(H') * g
			det = h11 * h22 - h21 * h21;
			dA = -(h22 * g1 - h21 * g2) / det;
			dB = -(-h21 * g1 + h11 * g2) / det;
			gd = g1 * dA + g2 * dB;

			stepsize = 1; // Line Search
			while (stepsize >= min_step) {
				newA = A + stepsize * dA;
				newB = B + stepsize * dB;

				// New function value
				newf = 0.0;
				for (int i = 0; i < l; i++) {
					fApB = dec_values[i] * newA + newB;
					if (fApB >= 0)
						newf += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
					else
						newf += (t[i] - 1) * fApB
								+ Math.log(1 + Math.exp(fApB));
				}
				// Check sufficient decrease
				if (newf < fval + 0.0001 * stepsize * gd) {
					A = newA;
					B = newB;
					fval = newf;
					break;
				} else
					stepsize = stepsize / 2.0;
			}

			if (stepsize < min_step) {
				LOG.info("Line search fails in two-class probability estimates\n");
				break;
			}
			if (iter >= max_iter)
				LOG.info("Reaching maximal iterations in two-class probability estimates\n");
		}

		probAB[0] = A;
		probAB[1] = B;
	}

	private static double sigmoid_predict(double decision_value, double A,
			double B) {
		double fApB = decision_value * A + B;
		if (fApB >= 0)
			return Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
		else
			return 1.0 / (1 + Math.exp(fApB));
	}

	// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
	private static void multiclass_probability(int k, double[][] r, double[] p) {
		int max_iter = Math.max(100, k);
		double[][] Q = new double[k][k];
		double[] Qp = new double[k];
		double pQp, eps = 0.005 / k;

		for (int t = 0; t < k; t++) {
			p[t] = 1.0 / k; // Valid if k = 1
			Q[t][t] = 0;
			for (int j = 0; j < t; j++) {
				Q[t][t] += r[j][t] * r[j][t];
				Q[t][j] = Q[j][t];
			}
			for (int j = t + 1; j < k; j++) {
				Q[t][t] += r[j][t] * r[j][t];
				Q[t][j] = -r[j][t] * r[t][j];
			}
		}
		for (int iter = 0; iter < max_iter; iter++) {
			// stopping condition, recalculate QP,pQP for numerical accuracy
			pQp = 0;
			for (int t = 0; t < k; t++) {
				Qp[t] = 0;
				for (int j = 0; j < k; j++)
					Qp[t] += Q[t][j] * p[j];
				pQp += p[t] * Qp[t];
			}
			double max_error = 0;
			for (int t = 0; t < k; t++) {
				double error = Math.abs(Qp[t] - pQp);
				if (error > max_error)
					max_error = error;
			}
			if (max_error < eps)
				break;

			for (int t = 0; t < k; t++) {
				double diff = (-Qp[t] + pQp) / Q[t][t];
				p[t] += diff;
				pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff)
						/ (1 + diff);
				for (int j = 0; j < k; j++) {
					Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
					p[j] /= (1 + diff);
				}
			}
			if (iter >= max_iter)
				LOG.info("Exceeds max_iter in multiclass_prob\n");
		}
	}

	// Cross-validation decision values for probability estimates
	private static void svm_binary_svc_probability(
			svm_problem<svm_node[]> prob, svm_parameter param, double Cp,
			double Cn, double[] probAB) {
		int nr_fold = 5;
		int[] perm = new int[prob.l];
		double[] dec_values = new double[prob.l];

		// random shuffle
		for (int i = 0; i < prob.l; i++)
			perm[i] = i;
		for (int i = 0; i < prob.l; i++) {
			int j = i + rand.nextInt(prob.l - i);
			ArrayUtil.swap(perm, i, j);
		}
		for (int i = 0; i < nr_fold; i++) {
			int begin = i * prob.l / nr_fold;
			int end = (i + 1) * prob.l / nr_fold;
			svm_problem<svm_node[]> subprob = new svm_problem<svm_node[]>();

			subprob.l = prob.l - (end - begin);
			subprob.x = new svm_node[subprob.l][];
			subprob.y = new double[subprob.l];

			int k = 0;
			for (int j = 0; j < begin; j++) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
				++k;
			}
			for (int j = end; j < prob.l; j++) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
				++k;
			}
			int p_count = 0, n_count = 0;
			for (int j = 0; j < k; j++)
				if (subprob.y[j] > 0)
					p_count++;
				else
					n_count++;

			if (p_count == 0 && n_count == 0)
				for (int j = begin; j < end; j++)
					dec_values[perm[j]] = 0;
			else if (p_count > 0 && n_count == 0)
				for (int j = begin; j < end; j++)
					dec_values[perm[j]] = 1;
			else if (p_count == 0 && n_count > 0)
				for (int j = begin; j < end; j++)
					dec_values[perm[j]] = -1;
			else {
				svm_parameter subparam = (svm_parameter) param.clone();
				subparam.probability = 0;
				subparam.C = 1.0;
				subparam.nr_weight = 2;
				subparam.weight_label = new int[2];
				subparam.weight = new double[2];
				subparam.weight_label[0] = +1;
				subparam.weight_label[1] = -1;
				subparam.weight[0] = Cp;
				subparam.weight[1] = Cn;
				svm_model<svm_node[]> submodel = svm_train(subprob, subparam);
				for (int j = begin; j < end; j++) {
					double[] dec_value = new double[1];
					svm_predict_values(submodel, prob.x[perm[j]], dec_value);
					dec_values[perm[j]] = dec_value[0];
					// ensure +1 -1 order; reason not using CV subroutine
					dec_values[perm[j]] *= submodel.label[0];
				}
			}
		}
		sigmoid_train(prob.l, dec_values, prob.y, probAB);
	}

	// Return parameter of a Laplace distribution
	private static double svm_svr_probability(svm_problem<svm_node[]> prob,
			svm_parameter param) {
		int nr_fold = 5;
		double[] ymv = new double[prob.l];
		double mae = 0;

		svm_parameter newparam = (svm_parameter) param.clone();
		newparam.probability = 0;
		svm_cross_validation(prob, newparam, nr_fold, ymv);
		for (int i = 0; i < prob.l; i++) {
			ymv[i] = prob.y[i] - ymv[i];
			mae += Math.abs(ymv[i]);
		}
		mae /= prob.l;
		double std = Math.sqrt(2 * mae * mae);
		int count = 0;
		mae = 0;
		for (int i = 0; i < prob.l; i++)
			if (Math.abs(ymv[i]) > 5 * std)
				++count;
			else
				mae += Math.abs(ymv[i]);
		mae /= (prob.l - count);
		LOG.info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma="
				+ mae + "\n");
		return mae;
	}

	// label: label name, start: begin of each class, count: #data of classes,
	// perm: indices to the original data
	// perm, length l, must be allocated before calling this subroutine
	private static void svm_group_classes(svm_problem<svm_node[]> prob,
			int[] nr_class_ret, int[][] label_ret, int[][] start_ret,
			int[][] count_ret, int[] perm) {
		int l = prob.l;
		int max_nr_class = 16;
		int nr_class = 0;
		int[] label = new int[max_nr_class];
		int[] count = new int[max_nr_class];
		int[] data_label = new int[l];

		for (int i = 0; i < l; i++) {
			int this_label = (int) (prob.y[i]);
			int j;
			for (j = 0; j < nr_class; j++) {
				if (this_label == label[j]) {
					++count[j];
					break;
				}
			}
			data_label[i] = j;
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

		//
		// Labels are ordered by their first occurrence in the training set.
		// However, for two-class sets with -1/+1 labels and -1 appears first,
		// we swap labels to ensure that internally the binary SVM has positive
		// data corresponding to the +1 instances.
		//
		if (nr_class == 2 && label[0] == -1 && label[1] == +1) {
			ArrayUtil.swap(label, 0, 1);
			ArrayUtil.swap(count,0,1);
			for (int i = 0; i < l; i++) {
				data_label[i] = (data_label[i] == 0) ? 1 : 0;
			}
		}

		int[] start = new int[nr_class];
		start[0] = 0;
		for (int i = 1; i < nr_class; i++)
			start[i] = start[i - 1] + count[i - 1];
		for (int i = 0; i < l; i++) {
			perm[start[data_label[i]]] = i;
			++start[data_label[i]];
		}
		start[0] = 0;
		for (int i = 1; i < nr_class; i++)
			start[i] = start[i - 1] + count[i - 1];

		nr_class_ret[0] = nr_class;
		label_ret[0] = label;
		start_ret[0] = start;
		count_ret[0] = count;
	}

	//
	// Interface functions
	//
	public static svm_model<svm_node[]> svm_train(svm_problem<svm_node[]> prob,
			svm_parameter param) {
		svm_model<svm_node[]> model = new svm_model<svm_node[]>();
		model.param = param;
		KernelFunction<svm_node[]> kf = param.makeKernelFunction();

		if (param.svm_type == svm_parameter.ONE_CLASS
				|| param.svm_type == svm_parameter.EPSILON_SVR
				|| param.svm_type == svm_parameter.NU_SVR) {
			// regression or one-class-svm
			model.nr_class = 2;
			model.label = null;
			model.nSV = null;
			model.probA = null;
			model.probB = null;
			model.sv_coef = new double[1][];

			if (param.probability == 1
					&& (param.svm_type == svm_parameter.EPSILON_SVR || param.svm_type == svm_parameter.NU_SVR)) {
				model.probA = new double[1];
				model.probA[0] = svm_svr_probability(prob, param);
			}

			AbstractSingleSVM<svm_node[]> svm = param.makeSVM(0, 0);

			svm.svm_train_one(prob.l, prob.x, prob.y, kf);
			model.rho = new double[1];
			model.rho[0] = svm.rho;

			int nSV = 0;
			for (int i = 0; i < prob.l; i++)
				if (Math.abs(svm.alpha[i]) > 0)
					++nSV;
			model.l = nSV;
			model.SV = new svm_node[nSV][];
			model.sv_coef[0] = new double[nSV];
			model.sv_indices = new int[nSV];
			for (int i = 0, j = 0; i < prob.l; i++)
				if (Math.abs(svm.alpha[i]) > 0) {
					model.SV[j] = prob.x[i];
					model.sv_coef[0][j] = svm.alpha[i];
					model.sv_indices[j] = i + 1;
					++j;
				}
		} else {
			// classification
			int l = prob.l;
			int[] tmp_nr_class = new int[1];
			int[][] tmp_label = new int[1][];
			int[][] tmp_start = new int[1][];
			int[][] tmp_count = new int[1][];
			int[] perm = new int[l];

			// group training data of the same class
			svm_group_classes(prob, tmp_nr_class, tmp_label, tmp_start,
					tmp_count, perm);
			int nr_class = tmp_nr_class[0];
			int[] label = tmp_label[0];
			int[] start = tmp_start[0];
			int[] count = tmp_count[0];

			if (nr_class == 1)
				LOG.info("WARNING: training data in only one class. See README for details.\n");

			svm_node[][] x = new svm_node[l][];
			for (int i = 0; i < l; i++)
				x[i] = prob.x[perm[i]];

			// calculate weighted C

			double[] weighted_C = new double[nr_class];
			for (int i = 0; i < nr_class; i++)
				weighted_C[i] = param.C;
			for (int i = 0; i < param.nr_weight; i++) {
				int j;
				for (j = 0; j < nr_class; j++)
					if (param.weight_label[i] == label[j])
						break;
				if (j == nr_class)
					System.err.print("WARNING: class label "
							+ param.weight_label[i]
							+ " specified in weight is not found\n");
				else
					weighted_C[j] *= param.weight[i];
			}

			// train k*(k-1)/2 models

			boolean[] nonzero = new boolean[l];
			for (int i = 0; i < l; i++)
				nonzero[i] = false;
			final int pairs = nr_class * (nr_class - 1) / 2;
			double[][] f_alpha = new double[pairs][];
			double[] f_rho = new double[pairs];

			double[] probA = null, probB = null;
			if (param.probability == 1) {
				probA = new double[pairs];
				probB = new double[pairs];
			}

			int p = 0;
			for (int i = 0; i < nr_class; i++)
				for (int j = i + 1; j < nr_class; j++) {
					svm_problem<svm_node[]> sub_prob = new svm_problem<svm_node[]>();
					final int si = start[i], sj = start[j];
					final int ci = count[i], cj = count[j];
					sub_prob.l = ci + cj;
					sub_prob.x = new svm_node[sub_prob.l][];
					sub_prob.y = new double[sub_prob.l];
					for (int k = 0; k < ci; k++) {
						sub_prob.x[k] = x[si + k];
						sub_prob.y[k] = +1;
					}
					for (int k = 0; k < cj; k++) {
						sub_prob.x[ci + k] = x[sj + k];
						sub_prob.y[ci + k] = -1;
					}

					if (param.probability == 1) {
						double[] probAB = new double[2];
						svm_binary_svc_probability(sub_prob, param,
								weighted_C[i], weighted_C[j], probAB);
						probA[p] = probAB[0];
						probB[p] = probAB[1];
					}

					AbstractSingleSVM<svm_node[]> svm = param.makeSVM(
							weighted_C[i], weighted_C[j]);
					svm.svm_train_one(sub_prob.l, sub_prob.x, sub_prob.y, kf);
					f_alpha[p] = svm.alpha;
					f_rho[p] = svm.rho;
					for (int k = 0; k < ci; k++)
						if (!nonzero[si + k] && Math.abs(f_alpha[p][k]) > 0)
							nonzero[si + k] = true;
					for (int k = 0; k < cj; k++)
						if (!nonzero[sj + k]
								&& Math.abs(f_alpha[p][ci + k]) > 0)
							nonzero[sj + k] = true;
					++p;
				}

			// build output

			model.nr_class = nr_class;

			model.label = new int[nr_class];
			for (int i = 0; i < nr_class; i++)
				model.label[i] = label[i];

			model.rho = new double[pairs];
			for (int i = 0; i < pairs; i++)
				model.rho[i] = f_rho[i];

			if (param.probability == 1) {
				model.probA = new double[pairs];
				model.probB = new double[pairs];
				for (int i = 0; i < pairs; i++) {
					model.probA[i] = probA[i];
					model.probB[i] = probB[i];
				}
			} else {
				model.probA = null;
				model.probB = null;
			}

			int nnz = 0;
			int[] nz_count = new int[nr_class];
			model.nSV = new int[nr_class];
			for (int i = 0; i < nr_class; i++) {
				int nSV = 0;
				for (int j = 0; j < count[i]; j++)
					if (nonzero[start[i] + j]) {
						++nSV;
						++nnz;
					}
				model.nSV[i] = nSV;
				nz_count[i] = nSV;
			}

			LOG.info("Total nSV = " + nnz + "\n");

			model.l = nnz;
			model.SV = new svm_node[nnz][];
			model.sv_indices = new int[nnz];
			p = 0;
			for (int i = 0; i < l; i++)
				if (nonzero[i]) {
					model.SV[p] = x[i];
					model.sv_indices[p++] = perm[i] + 1;
				}

			int[] nz_start = new int[nr_class];
			nz_start[0] = 0;
			for (int i = 1; i < nr_class; i++)
				nz_start[i] = nz_start[i - 1] + nz_count[i - 1];

			model.sv_coef = new double[nr_class - 1][];
			for (int i = 0; i < nr_class - 1; i++)
				model.sv_coef[i] = new double[nnz];

			p = 0;
			for (int i = 0; i < nr_class; i++)
				for (int j = i + 1; j < nr_class; j++) {
					// classifier (i,j): coefficients with
					// i are in sv_coef[j-1][nz_start[i]...],
					// j are in sv_coef[i][nz_start[j]...]

					int si = start[i], sj = start[j];
					int ci = count[i], cj = count[j];

					int q = nz_start[i];
					for (int k = 0; k < ci; k++)
						if (nonzero[si + k])
							model.sv_coef[j - 1][q++] = f_alpha[p][k];
					q = nz_start[j];
					for (int k = 0; k < cj; k++)
						if (nonzero[sj + k])
							model.sv_coef[i][q++] = f_alpha[p][ci + k];
					++p;
				}
		}
		return model;
	}

	// Stratified cross validation
	public static void svm_cross_validation(svm_problem<svm_node[]> prob,
			svm_parameter param, int nr_fold, double[] target) {
		int[] fold_start = new int[nr_fold + 1];
		int l = prob.l;
		int[] perm = new int[l];

		// stratified cv may not give leave-one-out rate
		// Each class to l folds -> some folds may have zero elements
		if ((param.svm_type == svm_parameter.C_SVC || param.svm_type == svm_parameter.NU_SVC)
				&& nr_fold < l) {
			int[] tmp_nr_class = new int[1];
			int[][] tmp_label = new int[1][];
			int[][] tmp_start = new int[1][];
			int[][] tmp_count = new int[1][];

			svm_group_classes(prob, tmp_nr_class, tmp_label, tmp_start,
					tmp_count, perm);

			int nr_class = tmp_nr_class[0];
			int[] start = tmp_start[0];
			int[] count = tmp_count[0];

			// random shuffle and then data grouped by fold using the array perm
			int[] fold_count = new int[nr_fold];
			int[] index = new int[l];
			for (int i = 0; i < l; i++)
				index[i] = perm[i];
			for (int c = 0; c < nr_class; c++)
				for (int i = 0; i < count[c]; i++) {
					int j = i + rand.nextInt(count[c] - i);
					ArrayUtil.swap(index, start[c] + i, start[c] + j);
				}
			for (int i = 0; i < nr_fold; i++) {
				fold_count[i] = 0;
				for (int c = 0; c < nr_class; c++)
					fold_count[i] += (i + 1) * count[c] / nr_fold - i
							* count[c] / nr_fold;
			}
			fold_start[0] = 0;
			for (int i = 1; i <= nr_fold; i++)
				fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
			for (int c = 0; c < nr_class; c++)
				for (int i = 0; i < nr_fold; i++) {
					int begin = start[c] + i * count[c] / nr_fold;
					int end = start[c] + (i + 1) * count[c] / nr_fold;
					for (int j = begin; j < end; j++) {
						perm[fold_start[i]] = index[j];
						fold_start[i]++;
					}
				}
			fold_start[0] = 0;
			for (int i = 1; i <= nr_fold; i++)
				fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
		} else {
			for (int i = 0; i < l; i++)
				perm[i] = i;
			for (int i = 0; i < l; i++) {
				int j = i + rand.nextInt(l - i);
				ArrayUtil.swap(perm, i,j);
			}
			for (int i = 0; i <= nr_fold; i++)
				fold_start[i] = i * l / nr_fold;
		}

		for (int i = 0; i < nr_fold; i++) {
			int begin = fold_start[i];
			int end = fold_start[i + 1];
			svm_problem<svm_node[]> subprob = new svm_problem<svm_node[]>();

			subprob.l = l - (end - begin);
			subprob.x = new svm_node[subprob.l][];
			subprob.y = new double[subprob.l];

			int k = 0;
			for (int j = 0; j < begin; ++j, ++k) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
			}
			for (int j = end; j < l; ++j, ++k) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
			}
			svm_model<svm_node[]> submodel = svm_train(subprob, param);
			if (param.probability == 1
					&& (param.svm_type == svm_parameter.C_SVC || param.svm_type == svm_parameter.NU_SVC)) {
				double[] prob_estimates = new double[submodel
						.svm_get_nr_class()];
				for (int j = begin; j < end; j++)
					target[perm[j]] = svm_predict_probability(submodel,
							prob.x[perm[j]], prob_estimates);
			} else
				for (int j = begin; j < end; j++)
					target[perm[j]] = svm_predict(submodel, prob.x[perm[j]]);
		}
	}

	public static double svm_get_svr_probability(svm_model<svm_node[]> model) {
		if ((model.param.svm_type == svm_parameter.EPSILON_SVR || model.param.svm_type == svm_parameter.NU_SVR)
				&& model.probA != null)
			return model.probA[0];
		else {
			LOG.warning("Model doesn't contain information for SVR probability inference\n");
			return 0;
		}
	}

	public static double svm_predict_values(svm_model<svm_node[]> model,
			svm_node[] x, double[] dec_values) {
		KernelFunction<svm_node[]> kf = model.param.makeKernelFunction();
		if (model.param.svm_type == svm_parameter.ONE_CLASS
				|| model.param.svm_type == svm_parameter.EPSILON_SVR
				|| model.param.svm_type == svm_parameter.NU_SVR) {
			double[] sv_coef = model.sv_coef[0];
			double sum = 0;
			for (int i = 0; i < model.l; i++)
				sum += sv_coef[i] * kf.similarity(x, model.SV[i]);
			sum -= model.rho[0];
			dec_values[0] = sum;

			if (model.param.svm_type == svm_parameter.ONE_CLASS)
				return (sum > 0) ? 1 : -1;
			else
				return sum;
		} else {
			int nr_class = model.nr_class;
			int l = model.l;

			double[] kvalue = new double[l];
			for (int i = 0; i < l; i++)
				kvalue[i] = kf.similarity(x, model.SV[i]);

			int[] start = new int[nr_class];
			start[0] = 0;
			for (int i = 1; i < nr_class; i++)
				start[i] = start[i - 1] + model.nSV[i - 1];

			int[] vote = new int[nr_class];
			for (int i = 0; i < nr_class; i++)
				vote[i] = 0;

			int p = 0;
			for (int i = 0; i < nr_class; i++)
				for (int j = i + 1; j < nr_class; j++, p++) {
					double sum = 0;
					int si = start[i], sj = start[j];
					int ci = model.nSV[i], cj = model.nSV[j];

					double[] coef1 = model.sv_coef[j - 1];
					double[] coef2 = model.sv_coef[i];
					for (int k = 0; k < ci; k++)
						sum += coef1[si + k] * kvalue[si + k];
					for (int k = 0; k < cj; k++)
						sum += coef2[sj + k] * kvalue[sj + k];
					sum -= model.rho[p];
					dec_values[p] = sum;

					++vote[(dec_values[p] > 0) ? i : j];
				}

			int vote_max_idx = 0;
			for (int i = 1; i < nr_class; i++)
				if (vote[i] > vote[vote_max_idx])
					vote_max_idx = i;

			return model.label[vote_max_idx];
		}
	}

	public static double svm_predict(svm_model<svm_node[]> model, svm_node[] x) {
		int nr_class = model.nr_class;
		double[] dec_values;
		if (model.param.svm_type == svm_parameter.ONE_CLASS
				|| model.param.svm_type == svm_parameter.EPSILON_SVR
				|| model.param.svm_type == svm_parameter.NU_SVR)
			dec_values = new double[1];
		else
			dec_values = new double[nr_class * (nr_class - 1) / 2];
		return svm_predict_values(model, x, dec_values);
	}

	public static double svm_predict_probability(svm_model<svm_node[]> model,
			svm_node[] x, double[] prob_estimates) {
		if ((model.param.svm_type == svm_parameter.C_SVC || model.param.svm_type == svm_parameter.NU_SVC)
				&& model.probA != null && model.probB != null) {
			int nr_class = model.nr_class;
			double[] dec_values = new double[nr_class * (nr_class - 1) / 2];
			svm_predict_values(model, x, dec_values);

			double min_prob = 1e-7;
			double[][] pairwise_prob = new double[nr_class][nr_class];

			for (int i = 0, k = 0; i < nr_class; i++)
				for (int j = i + 1; j < nr_class; j++, k++) {
					pairwise_prob[i][j] = Math.min(Math.max(
							sigmoid_predict(dec_values[k], model.probA[k],
									model.probB[k]), min_prob), 1 - min_prob);
					pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
				}
			multiclass_probability(nr_class, pairwise_prob, prob_estimates);

			int prob_max_idx = 0;
			for (int i = 1; i < nr_class; i++)
				if (prob_estimates[i] > prob_estimates[prob_max_idx])
					prob_max_idx = i;
			return model.label[prob_max_idx];
		} else
			return svm_predict(model, x);
	}
}
