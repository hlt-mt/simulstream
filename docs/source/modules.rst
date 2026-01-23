Python Modules
==============

Server
--------

.. automodule:: simulstream.server
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: generated
   :template: custom-module-template.rst

   simulstream.server.websocket_server
   simulstream.server.message_processor
   simulstream.server.speech_processors
   simulstream.server.speech_processors.base
   simulstream.server.speech_processors.incremental_output
   simulstream.server.speech_processors.sliding_window_retranslation
   simulstream.server.speech_processors.vad_wrapper
   simulstream.server.speech_processors.base_streamatt
   simulstream.server.speech_processors.remote
   simulstream.server.speech_processors.remote.http_proxy_speech_processor


Client
---------

.. automodule:: simulstream.client
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: generated
   :template: custom-module-template.rst

   simulstream.client.wav_reader_client

Evaluation
-----------

.. automodule:: simulstream.metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: generated
   :template: custom-module-template.rst

   simulstream.metrics.score_quality
   simulstream.metrics.score_latency
   simulstream.metrics.stats
   simulstream.metrics.scorers.quality
   simulstream.metrics.scorers.quality.mwersegmenter
   simulstream.metrics.scorers.latency
   simulstream.metrics.scorers.latency.mwersegmenter
