Refactored libSVM
=================

This is a refactored version of the popular libSVM library.

The original code is a literate port of C code to Java, and uses
more memory than necessary. This makes the Java version noticeably
slower, and harder to extend.

This version is heavily refactored (and still a work in progress), and
in particular makes it easier to use custom data types (e.g. non numeric data)
as well as custom kernel functions efficiently.


Benchmarking
============

These results are from a single run of 20news with linear kernel and otherwise
default options, as I'm still in progress of testing the codebase for bugs,
not for performance.

All accuracy evaluation was done with the C version.

C-SVM          | libSVM (C) | libSVM (Java) | libModernSVM
:--------------|-----------:| -------------:|-------------:
Training time  |     89.32s |       138.78s |      116.43s
Accuracy       |    84.022% |       84.022% |      84.022%
Accuracy (abs) |  3355/3993 |     3355/3993 |    3355/3993
Total nSV      |      12771 |         12771 |        12771
---------------|------------| --------------|--------------
nu-SVM         | libSVM (C) | libSVM (Java) | libModernSVM
---------------|------------| --------------|--------------
Training time  |    163.40s |       264.00s |      221.17s
Accuracy       |   81.5928% |      81.5928% |     81.5928%
Accuracy (abs) |  3258/3993 |     3258/3993 |    3258/3993
Total nSV      |      14015 |         14015 |        14015

As desired, our runtime improved substantially (15-20% faster) over the
original Java version, but we do not yet achieve the performance of the C version.
As is, the results appear to be identical, when the parameters are chosen
the same way (note that the current version does not have a command
line version).


Explanation of Performance Difference
-------------------------------------

The original implementation used a very C style data type: each vector was
represented as an `svm_node[]` array. For Java, this is a rather costly choice.
C uses structs, and `svm_node[]` then will be a continuous block of memory,
interleaving `int` and `double` (with the worst case that the values get padded
to 8 byte boundaries).

Java on the other hand does not have efficient structs. Many small objects can
hurt your performance, in particular when the garbage collection kicks in. This
representation of sparse vectors yields a kind of ragged array: each array
entries points to an object (which may reside at a different memory location)
and these tiny objects, carrying 12 bytes of payload but using 24 bytes of memory.
This memory layout therefore has an overhead of about 100%.

My replacement vector representation - consisting of an `int[]` for the indexes
and a `double[]` array - only adds 36-40 bytes of total overhead per vector.
Plus, I assume, this memory layout is easier to optimize for Java and easier
to perform garbage collection. I found these vector representations to work
well when working on [ELKI Data Mining](http://elki.dbs.ifi.lmu.de/). Many of
the changes done to this version of libSVM are meant to make it easier to
integrate in ELKI...

First benchmarking results showed that most of the runtime is indeed used to
compute the kernel functions. Benefits in computing dot products can therefore
be expected to directly translate into overall performance gains.
