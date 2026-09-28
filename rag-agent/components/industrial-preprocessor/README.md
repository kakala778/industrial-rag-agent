# Industrial Preprocessor Component

This directory contains the reusable document model and preprocessing code
from the Industrial Preprocessor project. It is a standalone Python component
inside the RAG repository; it is not imported by the RAG root package.

## Scope

The component includes document models, parser interfaces, MinerU output
reading, routing, quality checks, and native-text recovery helpers. MinerU
itself, model files, input documents, parser outputs, and data-dependent
evaluation scripts are not included. The source test suite remains in the
original development workspace.

## Setup

Use a separate Python 3.12 environment. Run commands from this directory so
its src package does not collide with the RAG project's root src package.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
~~~

The package metadata currently declares PyMuPDF. MinerU execution also needs a
separately installed and configured MinerU runtime; no runtime or model is
bundled here.

## Local data boundary

Put local inputs and generated artifacts under ignored paths such as
artifacts/, data/, or outputs/. Do not commit real industrial documents,
teacher-provided data, MinerU outputs, local models, or credentials.
