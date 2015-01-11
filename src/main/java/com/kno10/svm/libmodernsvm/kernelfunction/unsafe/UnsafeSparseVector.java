package com.kno10.svm.libmodernsvm.kernelfunction.unsafe;

import java.lang.reflect.Field;

import sun.misc.Unsafe;

/**
 * More compact sparse vector type.
 */
@SuppressWarnings("restriction")
public class UnsafeSparseVector {
  /**
   * Base address of the vectors memory
   */
  protected final long address;

  /**
   * Length of the vector.
   */
  protected final int size;

  /**
   * Memory requirement per integer.
   */
  public final static int BYTES_PER_INDEX = Integer.SIZE >>> 3;

  /**
   * Memory requirement per double.
   */
  public final static int BYTES_PER_VALUE = Double.SIZE >>> 3;

  /**
   * Memory requirement per entry.
   */
  public final static int BYTES_PER_ENTRY = BYTES_PER_INDEX + BYTES_PER_VALUE;

  /**
   * Unsafe memory access.
   */
  static final Unsafe unsafe;

  static {
    try {
      // fetch theUnsafe object
      Field field = Unsafe.class.getDeclaredField("theUnsafe");
      field.setAccessible(true);
      unsafe = (Unsafe) field.get(null);
      if(unsafe == null) {
        throw new RuntimeException("Unsafe access not available");
      }
    }
    catch(Exception e) {
      throw new RuntimeException(e);
    }
  }

  public UnsafeSparseVector(int[] index, double[] value) {
    super();
    assert (index.length == value.length);
    long addr = this.address = unsafe.allocateMemory(index.length * BYTES_PER_ENTRY);
    final int size = this.size = index.length;
    for(int i = 0; i < size; i++) {
      unsafe.putInt(addr, index[i]);
      addr += BYTES_PER_INDEX;
      unsafe.putDouble(addr, value[i]);
      addr += BYTES_PER_VALUE;
    }
  }

  public int index(int i) {
    assert (i >= 0 && i < size);
    return unsafe.getInt(address + BYTES_PER_ENTRY * i);
  }

  public double value(int i) {
    assert (i >= 0 && i < size);
    return unsafe.getDouble(address + BYTES_PER_ENTRY * i + BYTES_PER_INDEX);
  }

  public static double dot(UnsafeSparseVector x, UnsafeSparseVector y) {
    double sum = 0.;
    long addr1 = x.address, addr2 = y.address;
    final long aend1 = x.address + BYTES_PER_ENTRY * x.size;
    final long aend2 = y.address + BYTES_PER_ENTRY * y.size;
    while(addr1 < aend1 && addr2 < aend2) {
      final int xi = unsafe.getInt(addr1), yi = unsafe.getInt(addr2);
      if(xi == yi) {
        addr1 += BYTES_PER_INDEX;
        addr2 += BYTES_PER_INDEX;
        sum += unsafe.getDouble(addr1) * unsafe.getDouble(addr2);
        addr1 += BYTES_PER_VALUE;
        addr2 += BYTES_PER_VALUE;
      }
      else if(xi < yi) {
        addr1 += BYTES_PER_ENTRY;
      }
      else {
        addr2 += BYTES_PER_ENTRY;
      }
    }
    return sum;
  }

  public static double squareEuclidean(UnsafeSparseVector x, UnsafeSparseVector y) {
    double sum = 0.;
    long addr1 = x.address, addr2 = y.address;
    final long aend1 = x.address + BYTES_PER_ENTRY * x.size;
    final long aend2 = y.address + BYTES_PER_ENTRY * y.size;
    while(addr1 < aend1 && addr2 < aend2) {
      final int xi = unsafe.getInt(addr1), yi = unsafe.getInt(addr2);
      if(xi == yi) {
        addr1 += BYTES_PER_INDEX;
        addr2 += BYTES_PER_INDEX;
        double d = unsafe.getDouble(addr1) - unsafe.getDouble(addr2);
        sum += d * d;
        addr1 += BYTES_PER_VALUE;
        addr2 += BYTES_PER_VALUE;
      }
      else if(xi < yi) {
        addr1 += BYTES_PER_INDEX;
        double d = unsafe.getDouble(addr1);
        sum += d * d;
        addr1 += BYTES_PER_VALUE;
      }
      else {
        addr2 += BYTES_PER_INDEX;
        double d = unsafe.getDouble(addr2);
        sum += d * d;
        addr2 += BYTES_PER_VALUE;
      }
    }
    while(addr1 < aend1) {
      addr1 += BYTES_PER_INDEX;
      double d = unsafe.getDouble(addr1);
      sum += d * d;
      addr1 += BYTES_PER_VALUE;
    }
    while(addr2 < aend2) {
      addr2 += BYTES_PER_INDEX;
      double d = unsafe.getDouble(addr2);
      sum += d * d;
      addr2 += BYTES_PER_VALUE;
    }
    return sum;
  }

  public int size() {
    return size;
  }
}
