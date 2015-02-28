package com.kno10.svm.libmodernsvm.kernelfunction.unsafeonheap;

import java.lang.reflect.Field;

import sun.misc.Unsafe;

import com.kno10.svm.libmodernsvm.kernelfunction.Vector;

/**
 * Compact sparse vector type using {@code byte[]} for storage.
 * 
 * This is a hybrid implementation that uses on-heap allocation, and thus has a
 * slightly lower impact than the off-heap implementation.
 * 
 * Since the data is stored in a {@code byte[]}, it can be garbage-collected as
 * usual by the Java VM. At the same time, this object is low in overhead, so we
 * only need 16 bytes for this wrapper object, and 16 bytes for the array extra.
 * Thus, this implementation should save ~36 bytes per vector compared to the
 * off-heap implementation.
 * 
 * <i>Security concerns due to the use of {@link sun.misc.Unsafe} however
 * remain!</i>
 */
@SuppressWarnings("restriction")
public class UnsafeSparseVector implements Vector<UnsafeSparseVector> {
  /**
   * The data storage.
   */
  private final byte[] data;

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
   * Offset from the data object, usually 16.
   */
  private final static long DATA_BASE_OFFSET;

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
      DATA_BASE_OFFSET = unsafe.arrayBaseOffset(byte[].class);
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
    this.data = new byte[size * BYTES_PER_ENTRY];
    long off = DATA_BASE_OFFSET;
    for(int i = 0; i < size; i++) {
      unsafe.putInt(data, off, index[i]);
      off += BYTES_PER_INDEX;
      unsafe.putDouble(data, off, value[i]);
      off += BYTES_PER_VALUE;
    }
  }

  @Override
  public int size() {
    return data.length / BYTES_PER_ENTRY;
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
    assert (i >= 0 && i * BYTES_PER_ENTRY < data.length);
    return unsafe.getInt(data, DATA_BASE_OFFSET + BYTES_PER_ENTRY * i);
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
    assert (i >= 0 && i * BYTES_PER_ENTRY < data.length);
    return unsafe.getDouble(data, DATA_BASE_OFFSET + BYTES_PER_ENTRY * i + BYTES_PER_INDEX);
  }

  /**
   * Dot product of two vectors. Low level.
   * 
   * @param y Second vector
   * @return Dot product
   */
  @Override
  public double dot(UnsafeSparseVector y) {
    double sum = 0.;
    final byte[] dx = this.data, dy = y.data;
    long addrx = DATA_BASE_OFFSET, addry = DATA_BASE_OFFSET;
    final long aendx = addrx + dx.length, aendy = addry + dy.length;
    while(true) {
      final int xi = unsafe.getInt(dx, addrx), yi = unsafe.getInt(dy, addry);
      if(xi == yi) {
        sum += unsafe.getDouble(dx, addrx + BYTES_PER_INDEX) * unsafe.getDouble(dy, addry + BYTES_PER_INDEX);
      }
      if(xi <= yi) {
        addrx += BYTES_PER_ENTRY;
        if(addrx == aendx) {
          return sum;
        }
      }
      if(yi <= xi) {
        addry += BYTES_PER_ENTRY;
        if(addry == aendy) {
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
  public double squareEuclidean(UnsafeSparseVector y) {
    double sum = 0.;
    final byte[] dx = this.data, dy = y.data;
    long addr1 = DATA_BASE_OFFSET, addr2 = DATA_BASE_OFFSET;
    final long aend1 = addr1 + dx.length, aend2 = addr2 + dy.length;
    while(true) {
      final int xi = unsafe.getInt(dx, addr1), yi = unsafe.getInt(dy, addr2);
      double d = 0.;
      if(xi <= yi) {
        addr1 += BYTES_PER_INDEX;
        d += unsafe.getDouble(dx, addr1);
        addr1 += BYTES_PER_VALUE;
      }
      if(yi <= xi) {
        addr2 += BYTES_PER_INDEX;
        d -= unsafe.getDouble(dy, addr2);
        addr2 += BYTES_PER_VALUE;
      }
      sum += d * d;
      if(addr2 == aend2) {
        while(addr1 < aend1) {
          addr1 += BYTES_PER_INDEX;
          double d2 = unsafe.getDouble(dx, addr1);
          sum += d2 * d2;
          addr1 += BYTES_PER_VALUE;
        }
        return sum;
      }
      if(addr1 == aend1) {
        while(addr2 < aend2) {
          addr2 += BYTES_PER_INDEX;
          double d2 = unsafe.getDouble(dy, addr2);
          sum += d2 * d2;
          addr2 += BYTES_PER_VALUE;
        }
        return sum;
      }
    }
  }
}
