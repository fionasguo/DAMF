from setuptools import find_packages, setup

_deps = [
    'nltk',
    'emoji',
    'pandas',
    'typing',
    'transformers',
    'torch',
    'numpy',
    'sklearn',
    'seaborn',
    'psutil',
]

setup(
    name="DAMF",
    version=
    "1.0",
    author=
    "Siyi Guo, Department of Computer Science, University of Southern California",
    author_email="<fionasguo@gmail.com>",
    description=
    "Moral Foundations Inference with Domain Adapting Ability",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    python_requires=">=3.6.0",
    install_requires=_deps,
)
