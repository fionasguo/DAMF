from setuptools import find_packages, setup

_deps = [
    'nltk==3.7',
    'emoji==2.1.0',
    'pandas==1.4.4',
    'typing==4.3.0',
    'transformers==4.23.1',
    'torch==1.11.0',
    'numpy==1.21.5',
    'sklearn==0.24.2',
    'seaborn==0.12.0',
    'psutil==5.9.0',
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
