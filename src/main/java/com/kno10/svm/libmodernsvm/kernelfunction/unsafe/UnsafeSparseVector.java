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
  private final long address;

  /**
   * Length of the vector.
   */
  private final int size;

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

  // Initialize the unsafe object
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

  /**
   * Constructor
   * 
   * @param index Dimension indexes
   * @param value Values
   * @param size Number of valid components.
   */
  public UnsafeSparseVector(int[] index, double[] value, int size) {
    super();
    long addr = this.address = unsafe.allocateMemory(index.length * BYTES_PER_ENTRY);
    this.size = size;
    for(int i = 0; i < size; i++) {
      unsafe.putInt(addr, index[i]);
      addr += BYTES_PER_INDEX;
      unsafe.putDouble(addr, value[i]);
      addr += BYTES_PER_VALUE;
    }
  }

  /**
   * Java finalizer, frees the vector memory.
   */
  @Override
  protected void finalize() throws Throwable {
    unsafe.freeMemory(address);
    super.finalize();
  }

  /**
   * Number of valid entries in this sparse vector.
   * 
   * @return Size
   */
  public int size() {
    return size;
  }

  /**
   * Index at position i.
   * 
   * <i>Warning:</i> does not check bounds. This can crash your VM.
   * 
   * @param i Position
   * @return Index
   */
  public int index(int i) {
    assert (i >= 0 && i < size);
    return unsafe.getInt(address + BYTES_PER_ENTRY * i);
  }

  /**
   * Value at position i
   *
   * <i>Warning:</i> does not check bounds. This can crash your VM.
   * 
   * @param i Position
   * @return Value
   */
  public double value(int i) {
    assert (i >= 0 && i < size);
    return unsafe.getDouble(address + BYTES_PER_ENTRY * i + BYTES_PER_INDEX);
  }

  /**
   * Dot product of two vectors. Low level.
   * 
   * @param x First vector
   * @param y Second vector
   * @return Dot product
   */
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

  /**
   * Squared Euclidean distance of two vectors.
   * 
   * @param x First vector
   * @param y Second vector
   * @return Squared Euclidean distance
   */
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
}
