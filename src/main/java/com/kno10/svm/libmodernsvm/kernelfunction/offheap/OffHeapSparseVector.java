package com.kno10.svm.libmodernsvm.kernelfunction.offheap;

import java.lang.reflect.Field;

import sun.misc.Unsafe;

import com.kno10.svm.libmodernsvm.kernelfunction.Vector;
import com.kno10.svm.libmodernsvm.kernelfunction.unsafeonheap.UnsafeSparseVector;

/**
 * Compact sparse vector type using <em>off-heap memory</em>.
 * 
 * Note: this may appear very "sexy" initially to allocate the vectors off-heap,
 * and access them low level. But unfortunately we need to integrate this
 * carefully with Java memory management. And then it suddenly comes with quite
 * some overhead.
 * 
 * <ul>
 * <li>The {@link #finalize()} method is implemented in Java via a special
 * {@link java.lang.ref.Finalizer} instance, which needs 40 bytes of memory.</li>
 * <li>The address is always a {@code long} address, so the vector uses 24 bytes
 * of memory.</li>
 * <li>Usually, the C library will need another 4-16 bytes of overhead (size).</li>
 * <li>Since we do not check bounds and {@link Unsafe}, using this class
 * potentially allows reading all Java memory.</li>
 * </ul>
 * 
 * Largely due to the finalizer, the expected overhead of this implementation is
 * 68 bytes. See {@link UnsafeSparseVector} for an unsafe on-heap implementation
 * with less overhead.
 */
@SuppressWarnings("restriction")
public class OffHeapSparseVector implements Vector<OffHeapSparseVector> {
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
  private final static int BYTES_PER_INDEX = Integer.SIZE >>> 3;

  /**
   * Memory requirement per double.
   */
  private final static int BYTES_PER_VALUE = Double.SIZE >>> 3;

  /**
   * Memory requirement per entry.
   */
  private final static int BYTES_PER_ENTRY = BYTES_PER_INDEX + BYTES_PER_VALUE;

  /**
   * Unsafe memory access.
   */
  private static final Unsafe unsafe;

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
  public OffHeapSparseVector(int[] index, double[] value, int size) {
    super();
    long addr = this.address = unsafe.allocateMemory(size * BYTES_PER_ENTRY);
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

  @Override
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
  @Override
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
  @Override
  public double value(int i) {
    assert (i >= 0 && i < size);
    return unsafe.getDouble(address + BYTES_PER_ENTRY * i + BYTES_PER_INDEX);
  }

  /**
   * Dot product of two vectors. Low level.
   * 
   * @param y Second vector
   * @return Dot product
   */
  @Override
  public double dot(OffHeapSparseVector y) {
    double sum = 0.;
    long addr1 = this.address, addr2 = y.address;
    final long aend1 = this.address + BYTES_PER_ENTRY * this.size;
    final long aend2 = y.address + BYTES_PER_ENTRY * y.size;
    while(true) {
      final int xi = unsafe.getInt(addr1), yi = unsafe.getInt(addr2);
      if(xi == yi) {
        sum += unsafe.getDouble(addr1 + BYTES_PER_INDEX) * unsafe.getDouble(addr2 + BYTES_PER_INDEX);
      }
      if(xi <= yi) {
        addr1 += BYTES_PER_ENTRY;
        if(addr1 == aend1) {
          return sum;
        }
      }
      if(yi <= xi) {
        addr2 += BYTES_PER_ENTRY;
        if(addr2 == aend2) {
          return sum;
        }
      }
    }
  }

  /**
   * Squared Euclidean distance of two vectors.
   * 
   * @param y Second vector
   * @return Squared Euclidean distance
   */
  public double squareEuclidean(OffHeapSparseVector y) {
    double sum = 0.;
    long addr1 = this.address, addr2 = y.address;
    final long aend1 = addr1 + BYTES_PER_ENTRY * this.size;
    final long aend2 = addr2 + BYTES_PER_ENTRY * y.size;
    while(true) {
      final int xi = unsafe.getInt(addr1), yi = unsafe.getInt(addr2);
      double d = 0.;
      if(xi <= yi) {
        addr1 += BYTES_PER_INDEX;
        d += unsafe.getDouble(addr1);
        addr1 += BYTES_PER_VALUE;
      }
      if(yi <= xi) {
        addr2 += BYTES_PER_INDEX;
        d -= unsafe.getDouble(addr2);
        addr2 += BYTES_PER_VALUE;
      }
      sum += d * d;
      if(addr2 == aend2) {
        while(addr1 < aend1) {
          addr1 += BYTES_PER_INDEX;
          double d2 = unsafe.getDouble(addr1);
          sum += d2 * d2;
          addr1 += BYTES_PER_VALUE;
        }
        return sum;
      }
      if(addr1 == aend1) {
        while(addr2 < aend2) {
          addr2 += BYTES_PER_INDEX;
          double d2 = unsafe.getDouble(addr2);
          sum += d2 * d2;
          addr2 += BYTES_PER_VALUE;
        }
        return sum;
      }
    }
  }
}
