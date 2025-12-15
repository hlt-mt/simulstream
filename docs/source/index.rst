.. simulstream documentation master file, created by
   sphinx-quickstart on Mon Sep  8 12:01:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to *simulstream* documentation
=====================================

``simulstream`` is a Python library for simultaneous/streaming speech recognition and translation.
It enables both the simulation with existing files to score systems, like in the SimulEval_
project, and the possibility to run demos on a browser.

.. _SimulEval: https://github.com/facebookresearch/SimulEval

``simulstream`` provides a WebSocket server and utilities for running streaming speech processing
experiments and demos. It supports real-time transcription and translation through streaming audio
input. By streaming, we mean that the library by default assumes that the input is an unbounded
speech signal, rather than many short speech segments as in simultaneous speech processing.
The simultaneous setting can be easily addressed by pre-segmenting the audio into many small
segments and feed each segment to ``simulstream``.

The repository is tested using Python 3.11. Although it may work also with other Python versions,
wedo not ensure compatibility with them. Check out the :doc:`usage` section for instructions on how
to use the repository and the :doc:`installation` section for further information about how to
install the project.

- **Github:** `https://github.com/hlt-mt/simulstream.git <https://github.com/hlt-mt/simulstream.git>`__
- **PyPi:** `https://pypi.org/project/simulstream <https://pypi.org/project/simulstream>`__

.. toctree::
   :hidden:

   installation

.. toctree::
   :hidden:

   usage


Python API Documentation
------------------

Here is the list of the modules currently part of the repository with
the corresponding documentation:

.. toctree::
   :maxdepth: 2

   modules

Credits
________


If you use this library, please cite:

.. code-block::

  @article{gaido-et-al-2025-simulstream,
    title={{simulstream: Open-Source Toolkit for Evaluation and Demonstration of Streaming Speech-to-Text Translation Systems}},
    author={Gaido, Marco and Papi, Sara and Cettolo, Mauro and Negri, Matteo and Bentivogli, Luisa},
    journal = "arXiv",
    year={2025}
  }