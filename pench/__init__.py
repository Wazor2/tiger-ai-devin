"""Pench tiger camera-trap intelligence system."""
import os

# torch ships libiomp5md.dll and faiss-cpu ships libomp140.x86_64.dll; on
# Windows the second OpenMP runtime to initialise aborts the process ("OMP:
# Error #15"), which kills the pipeline non-deterministically depending on
# import order. Allowing the duplicate runtime is the only workaround short of
# rebuilding one of the wheels.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
