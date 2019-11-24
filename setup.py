import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data-modeling",
    version="0.0.1",
    author="Abhinav",
    author_email="abhi9520@gmail.com",
    description="Automated package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhi9520",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)