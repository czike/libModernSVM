package com.kno10.svm.libmodernsvm;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.regex.Pattern;

import com.kno10.svm.libmodernsvm.data.ByteWeightedArrayDataSet;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.sparsevec.LinearKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.sparsevec.RadialBasisKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.sparsevec.SparseVector;
import com.kno10.svm.libmodernsvm.model.ClassificationModel;
import com.kno10.svm.libmodernsvm.variants.CSVC;
import com.kno10.svm.libmodernsvm.variants.NuSVC;

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
      DataSet<SparseVector> data = loadData(new FileInputStream(args[0]));
      double gamma = 1. / dimensionality(data); // Default: 1/numfeatures
      KernelFunction<SparseVector> kf = new RadialBasisKernelFunction(gamma);
      // KernelFunction<SparseVector> kf = new LinearKernelFunction();
      System.err.println("Data set size: " + data.size());
      ClassificationModel<SparseVector> m;
      m = new CSVC<SparseVector>(1e-3, true, 100).train(data, kf, null);
      // m = new NuSVC<SparseVector>(0.1, true, 100, .5).train(data, kf, null);
      System.err.println(m.l + " " + m.nr_class + " " + m.SV.size());

      writeModel(new FileOutputStream(args[1]), m);
    }
    catch(IOException e) {
      e.printStackTrace();
    }
  }

  private static int dimensionality(DataSet<SparseVector> data) {
    int max = -1;
    for(int i = 0; i < data.size(); ++i) {
      SparseVector v = data.get(i);
      for(int j : v.index) {
        max = (j > max) ? j : max;
      }
    }
    return max + 1;
  }

  private static void writeModel(FileOutputStream fileOutputStream, ClassificationModel<SparseVector> m) throws IOException {
    PrintWriter p = new PrintWriter(fileOutputStream);
    p.println("svm_type c_svc"); // FIXME: SVM type
    p.println("kernel_type linear"); // FIXME: Kernel type
    p.println("nr_class " + m.nr_class);
    p.println("total_sv " + m.SV.size());
    p.print("rho");
    for(int i = 0; i < m.rho.length; i++) {
      p.print(' ');
      p.print(m.rho[i]);
    }
    p.println();
    p.print("label");
    for(int i = 0; i < m.label.length; i++) {
      p.print(' ');
      p.print(m.label[i]);
    }
    p.println();
    p.print("nr_sv");
    for(int i = 0; i < m.nSV.length; i++) {
      p.print(' ');
      p.print(m.nSV[i]);
    }
    p.println();
    p.println("SV");
    for(int j = 0, n = m.SV.size(); j < n; ++j) {
      // Print all but last one:
      for(int i = 0; i < m.nr_class - 1; i++) {
        if(i > 0) {
          p.print(' ');
        }
        p.print(m.sv_coef[i][j]);
      }
      // Print vector
      SparseVector sv = m.SV.get(j);
      for(int i = 0, l = sv.index.length; i < l; ++i) {
        p.print(' ');
        p.print(sv.index[i] + 1);
        p.print(':');
        p.print(sv.value[i]);
      }
      p.println(' '); // Compatibility with libsvm output
    }
    p.close();
  }

  private static DataSet<SparseVector> loadData(FileInputStream in) throws IOException {
    ByteWeightedArrayDataSet<SparseVector> data = new ByteWeightedArrayDataSet<SparseVector>(1000);
    Pattern p = Pattern.compile("[ :]");
    BufferedReader r = null;
    try {
      r = new BufferedReader(new InputStreamReader(in));
      String line;
      while((line = r.readLine()) != null) {
        String[] b = p.split(line);
        byte c = (byte) Integer.parseInt(b[0]);
        int[] idx = new int[(b.length - 1) >> 1];
        double[] val = new double[(b.length - 1) >> 1];
        for(int i = 0, j = 1; j < b.length; i++) {
          idx[i] = Integer.parseInt(b[j++]) - 1;
          val[i] = Double.parseDouble(b[j++]);
        }
        SparseVector d = new SparseVector(idx, val);
        data.add(d, c);
      }
    }
    finally {
      if(r != null) {
        r.close();
      }
    }
    return data;
  }
}
