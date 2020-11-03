#!/usr/bin/env python3
__author__ = 'Rafael Zamora-Resendiz, rzamoraresendiz@protonmail.com'

from setuptools import setup, find_packages

setup(
    name="beacon",
    version="0.0.1",
    description="BEACON: BERT Extracted Attentions for Clinical Ontology aNnotation",
    license="MIT",
    packages=find_packages(exclude=["artifacts", "examples"]),
    package_data={
        'beacon': ['*.hy', '*.py'],
    },
    install_requires = ["hy", "torch", "dill", "pandas", "numpy", "tqdm"],
)
