package com.kno10.svm.libmodernsvm;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.regex.Pattern;

import com.kno10.svm.libmodernsvm.data.ByteWeightedArrayDataSet;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.RadialBasisKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.SparseVectorEntry;
import com.kno10.svm.libmodernsvm.model.ClassificationModel;
import com.kno10.svm.libmodernsvm.variants.SVC_C;

/**
 * Simple application for debugging training.
 * 
 * TODO: parameters - develop into a fully compatible replacement for svm-train?
 * 
 * @author Erich Schubert
 */
public class SimpleTrainApp {

	public static void main(String[] args) {
		try {
			DataSet<SparseVectorEntry[]> data = loadData(new FileInputStream(
					args[0]));
			double gamma = 1. / 13; // Default: 1/numfeatures
			System.err.println("Data set size: " + data.size());
			KernelFunction<SparseVectorEntry[]> kf = new RadialBasisKernelFunction(
					gamma);
			ClassificationModel<SparseVectorEntry[]> m = new SVC_C<SparseVectorEntry[]>(
					0.1, 1, 100).train(data, kf, null);
			System.err.println(m.l + " " + m.nr_class + " " + m.SV.size());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static DataSet<SparseVectorEntry[]> loadData(FileInputStream in)
			throws IOException {
		ByteWeightedArrayDataSet<SparseVectorEntry[]> data = new ByteWeightedArrayDataSet<SparseVectorEntry[]>(
				1000);
		Pattern p = Pattern.compile("[ :]");
		BufferedReader r = null;
		try {
			r = new BufferedReader(new InputStreamReader(in));
			String line;
			while ((line = r.readLine()) != null) {
				String[] b = p.split(line);
				byte c = (byte) Integer.parseInt(b[0]);
				SparseVectorEntry[] d = new SparseVectorEntry[(b.length - 1) >> 1];
				for (int i = 0, j = 1; j < b.length; i++) {
					int idx = Integer.parseInt(b[j++]);
					double val = Double.parseDouble(b[j++]);
					d[i] = new SparseVectorEntry(idx, val);
				}
				data.add(d, c);
			}
		} finally {
			if (r != null) {
				r.close();
			}
		}
		return data;
	}
}
