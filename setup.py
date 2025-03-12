from setuptools import setup, find_packages

setup(
    name="qamodel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
    ],
    author="sachin kumar",
    author_email="sachin18449kumar@gmail.com",
    description="A QA dataset processing and RNN-based model library",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
