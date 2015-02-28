package com.kno10.svm.libmodernsvm.format.libsvm;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import javolution.text.Cursor;
import javolution.text.TypeFormat;

import com.kno10.svm.libmodernsvm.data.ByteWeightedArrayDataSet;
import com.kno10.svm.libmodernsvm.data.DataSet;
import com.kno10.svm.libmodernsvm.kernelfunction.Vector;
import com.kno10.svm.libmodernsvm.kernelfunction.VectorBuilder;

public class LibSVMDataReader {
  public static <V extends Vector<?>> DataSet<V> loadData(FileInputStream in, VectorBuilder<? extends V> builder) throws IOException {
    ByteWeightedArrayDataSet<V> data = new ByteWeightedArrayDataSet<V>(1000);
    BufferedReader r = null;
    try {
      r = new BufferedReader(new InputStreamReader(in));
      String line;
      Cursor cursor = new Cursor();
      while((line = r.readLine()) != null) {
        cursor.setIndex(0);
        byte c = TypeFormat.parseByte(line, cursor);
        builder.clear();
        while(cursor.skip(' ', line) && cursor.getIndex() < line.length()) {
          int idx = TypeFormat.parseInt(line, cursor) - 1;
          cursor.skip(':', line);
          double val = TypeFormat.parseDouble(line, cursor);
          builder.add(idx, val);
        }
        data.add(builder.build(), c);
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
