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

These results are from a single run of 20news with linear kernel and C-SVM, as
I'm still in progress of testing the codebase for bugs, not for performance.

All accuracy evaluation was done with the C version.

Version        | libSVM (C) | libSVM (Java) | libModernSVM
:--------------|-----------:| -------------:|-------------:
Training time  |     89.32s |       138.78s |      103.16s
Accuracy       |    84.022% |       84.022% |     84.0721%
Accuracy (abs) |  3355/3993 |     3355/3993 |    3357/3993
Total nSV      |      12771 |         12771 |        12371

Interestingly enough, our runtime improved to about halfway between the C
and the original Java version (this is substantial!) - but for some reason
not yet clear, we also improved on result quality marginally by 2 samples!
This needs further investigation.

The original Java version apparently returned exactly the same result as
the C version, but our version also kept 400 fewer support vectors. So the
good news so far are that this version is faster, more flexible, and yields
a smaller (and thus faster at predicting) SVM. The bad news is, it's not clear
whether this trend is consistent, and what caused it.
