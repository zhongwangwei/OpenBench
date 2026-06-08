"""Figure-namelist YAML resources for OpenBench visualizations.

This is an empty package marker. The directory is referenced as a
package via :func:`importlib.resources.files` from
``openbench.config.adapter`` and ``openbench.cli.check`` so that the
contained YAML files (``figlib.yaml``, ``ANOVA.yaml``, …) are
discoverable in any install layout — wheel, editable, sdist, or
zipimport.
"""
