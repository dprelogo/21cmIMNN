from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="imnn21cm",
    version="0.1dev",
    author="David Prelogović",
    author_email="david.prelogovic@gmail.com",
    description="IMNN compression of the cosmic 21-cm signal",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dprelogo/imnn21cm",
    packages=["imnn21cm"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "jax",
        "flax",
        "optax",
        "numpy",
        "h5py",
    ],
)
