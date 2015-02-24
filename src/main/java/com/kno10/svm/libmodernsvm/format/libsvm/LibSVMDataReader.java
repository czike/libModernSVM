package com.kno10.svm.libmodernsvm.format.libsvm;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.regex.Pattern;

import javolution.text.TypeFormat;

import com.kno10.svm.libmodernsvm.data.ByteWeightedArrayDataSet;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.unsafe.UnsafeSparseVector;

public class LibSVMDataReader {
  public static DataSet<UnsafeSparseVector> loadData(FileInputStream in) throws IOException {
    ByteWeightedArrayDataSet<UnsafeSparseVector> data = new ByteWeightedArrayDataSet<UnsafeSparseVector>(1000);
    Pattern p = Pattern.compile("[ :]");
    BufferedReader r = null;
    try {
      r = new BufferedReader(new InputStreamReader(in));
      String line;
      int[] idx = new int[100];
      double[] val = new double[100];
      while((line = r.readLine()) != null) {
        String[] b = p.split(line);
        byte c = TypeFormat.parseByte(b[0]);
        int dim = 0;
        for(int j = 1; j < b.length; dim++) {
          if(dim == idx.length) { // Resize buffer.
            idx = Arrays.copyOf(idx, dim << 1);
            val = Arrays.copyOf(val, dim << 1);
          }
          idx[dim] = TypeFormat.parseInt(b[j++]) - 1;
          val[dim] = TypeFormat.parseDouble(b[j++]);
        }
        UnsafeSparseVector d = new UnsafeSparseVector(idx, val, dim);
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
