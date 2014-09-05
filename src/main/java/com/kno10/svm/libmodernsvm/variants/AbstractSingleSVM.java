package com.kno10.svm.libmodernsvm.variants;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.kno10.svm.libmodernsvm.ArrayUtil;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;

public abstract class AbstractSingleSVM<T> {
	protected double eps;
	protected int shrinking;
	protected double cache_size;

	// Output variables
	public double[] alpha;
	public double rho;

	public AbstractSingleSVM(double eps, int shrinking, double cache_size) {
		this.eps = eps;
		this.shrinking = shrinking;
		this.cache_size = cache_size;
	}

	abstract protected Solver.SolutionInfo solve(DataSet<T> x,
			KernelFunction<? super T> kernel_function);

	public void svm_train_one(DataSet<T> x,
			KernelFunction<? super T> kernel_function) {
		final int l = x.size();
		alpha = new double[l];
		Solver.SolutionInfo si = solve(x, kernel_function);
		rho = si.rho;

		if (getLogger().isLoggable(Level.INFO)) {
			getLogger().info("obj = " + si.obj + ", rho = " + si.rho + "\n");
		}

		// output SVs

		int nSV = 0;
		int nBSV = 0;
		for (int i = 0; i < l; i++) {
			if (Math.abs(alpha[i]) > 0) {
				++nSV;
				if (Math.abs(alpha[i]) >= ((x.value(i) > 0) ? si.upper_bound_p
						: si.upper_bound_n))
					++nBSV;
			}
		}

		if (getLogger().isLoggable(Level.INFO)) {
			getLogger().info("nSV = " + nSV + ", nBSV = " + nBSV + "\n");
		}
	}

	abstract protected Logger getLogger();

	public void set_weights(double Cp, double Cn) {
		// Ignore by default.
	}

	/**
	 * Build a shuffled index array.
	 * 
	 * @param perm
	 *            Array storing the permutation
	 * @param l
	 *            Size
	 */
	public static int[] shuffledIndex(int[] perm, int l) {
		Random rand = new Random();
		// Shuffle data set.
		for (int i = 0; i < l; i++)
			perm[i] = i;
		for (int i = 0; i < l; i++) {
			int j = i + rand.nextInt(l - i);
			ArrayUtil.swap(perm, i, j);
		}
		return perm;
	}

	public static void stratifiedFolds(DataSet<?> x, int nr_fold, int[] perm,
			int[] fold_start) {
		final int l = x.size();
		int[] tmp_nr_class = new int[1];
		int[][] tmp_label = new int[1][];
		int[][] tmp_start = new int[1][];
		int[][] tmp_count = new int[1][];

		svm_group_classes(x, tmp_nr_class, tmp_label, tmp_start, tmp_count,
				perm);

		int nr_class = tmp_nr_class[0];
		int[] start = tmp_start[0];
		int[] count = tmp_count[0];

		Random rand = new Random();
		// random shuffle and then data grouped by fold using the array perm
		int[] fold_count = new int[nr_fold];
		int[] index = new int[l];
		for (int i = 0; i < l; i++) {
			index[i] = perm[i];
		}
		for (int c = 0; c < nr_class; c++) {
			for (int i = 0; i < count[c]; i++) {
				int j = i + rand.nextInt(count[c] - i);
				ArrayUtil.swap(index, start[c] + i, start[c] + j);
			}
		}
		for (int i = 0; i < nr_fold; i++) {
			fold_count[i] = 0;
			for (int c = 0; c < nr_class; c++) {
				fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c]
						/ nr_fold;
			}
		}
		fold_start[0] = 0;
		for (int i = 1; i <= nr_fold; i++)
			fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
		for (int c = 0; c < nr_class; c++) {
			for (int i = 0; i < nr_fold; i++) {
				int begin = start[c] + i * count[c] / nr_fold;
				int end = start[c] + (i + 1) * count[c] / nr_fold;
				for (int j = begin; j < end; j++) {
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		}
		fold_start[0] = 0;
		for (int i = 1; i <= nr_fold; i++) {
			fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
		}
	}

	// label: label name, start: begin of each class, count: #data of classes,
	// perm: indices to the original data
	// perm, length l, must be allocated before calling this subroutine
	protected static void svm_group_classes(DataSet<?> x, int[] nr_class_ret,
			int[][] label_ret, int[][] start_ret, int[][] count_ret, int[] perm) {
		final int l = x.size();
		int max_nr_class = 16;
		int nr_class = 0;
		int[] label = new int[max_nr_class];
		int[] count = new int[max_nr_class];
		int[] data_label = new int[l];

		for (int i = 0; i < l; i++) {
			int this_label = x.classnum(i);
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
			ArrayUtil.swap(count, 0, 1);
			for (int i = 0; i < l; i++) {
				data_label[i] = (data_label[i] == 0) ? 1 : 0;
			}
		}

		int[] start = new int[nr_class];
		start[0] = 0;
		for (int i = 1; i < nr_class; i++) {
			start[i] = start[i - 1] + count[i - 1];
		}
		for (int i = 0; i < l; i++) {
			perm[start[data_label[i]]] = i;
			++start[data_label[i]];
		}
		start[0] = 0;
		for (int i = 1; i < nr_class; i++) {
			start[i] = start[i - 1] + count[i - 1];
		}

		nr_class_ret[0] = nr_class;
		label_ret[0] = label;
		start_ret[0] = start;
		count_ret[0] = count;
	}
}