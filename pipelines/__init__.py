"""Per-model DSL pipelines: algorithm DAG + named Schedules.

Each module exposes a `pipeline` (the algorithm) and one or more named
Schedule objects. The pipeline + schedule are consumed by dsl.codegen.build()
to produce a preprocessing callable.
"""
