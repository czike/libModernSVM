package com.kno10.svm.libmodernsvm;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.format.libsvm.LibSVMDataReader;
import com.kno10.svm.libmodernsvm.format.libsvm.LibSVMModelWriter;
import com.kno10.svm.libmodernsvm.kernelfunction.KernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.unsafe.LinearKernelFunction;
import com.kno10.svm.libmodernsvm.kernelfunction.unsafe.UnsafeSparseVector;
import com.kno10.svm.libmodernsvm.model.ClassificationModel;
import com.kno10.svm.libmodernsvm.variants.AbstractSVC;
import com.kno10.svm.libmodernsvm.variants.CSVC;

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
      DataSet<UnsafeSparseVector> data = LibSVMDataReader.loadData(new FileInputStream(args[0]));
      double gamma = 1. / dimensionality(data); // Default: 1/numfeatures
      // KernelFunction<UnsafeSparseVector> kf = new
      // RadialBasisKernelFunction(gamma);
      KernelFunction<UnsafeSparseVector> kf = new LinearKernelFunction();
      System.err.println("Data set size: " + data.size());
      ClassificationModel<UnsafeSparseVector> m;
      AbstractSVC<UnsafeSparseVector> svm = new CSVC<UnsafeSparseVector>(1e-3, true, 500);
      // AbstractSVC<UnsafeSparseVector> svm = new NuSVC<UnsafeSparseVector>(1,
      // true, 500, .5);
      m = svm.train(data, kf, null);
      System.err.println(m.l + " " + m.nr_class + " " + m.SV.size());

      LibSVMModelWriter.writeModel(new PrintWriter(new FileOutputStream(args[1])), m, svm, kf);
    }
    catch(IOException e) {
      e.printStackTrace();
    }
  }

  private static int dimensionality(DataSet<UnsafeSparseVector> data) {
    int max = -1;
    for(int i = 0; i < data.size(); ++i) {
      UnsafeSparseVector v = data.get(i);
      for(int j = 0, l = v.size(); j < l; ++j) {
        int idx = v.index(j);
        max = (idx > max) ? idx : max;
      }
    }
    return max + 1;
  }
}
