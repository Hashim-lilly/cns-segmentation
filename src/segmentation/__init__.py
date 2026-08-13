"""Automated multi-label segmentation pipeline for CFD-ready SAS extraction.

Orchestrates pre-trained models (TotalSpineSeg, model-canal-seg, RootletSeg)
to produce multi-label segmentation of the spinal subarachnoid space:
  - Spinal cord (inner boundary)
  - Dural sac / spinal canal (outer boundary)
  - Nerve rootlets (obstructions that drive steady-streaming)

The fluid domain is then: CSF = canal − cord − rootlets
"""
