from setuptools import setup, find_packages

with open("README.md") as readme:
    long_description = readme.read()

requirements = [
    "matplotlib>=3.5.2",
    "numpy>=1.23.1",
    "pandas>=1.4.3",
    "ReliefF>=0.1.2",
    "scikit-learn>=1.1.1",
    "seaborn>=0.11.2",
    "sklearn>=0.0"
]

setup(
    name='feature-selection-package',
    version='1.0.0',
    license='MIT',
    author="Matteo Serafino",
    author_email='matteo.serafino1@gmail.com',
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url='https://github.com/matteo-serafino/feature-selection.git',
    keywords='feature-selection',
    install_requires=requirements,
    python_requires=">=3.6.2",
    include_package_data=True
)