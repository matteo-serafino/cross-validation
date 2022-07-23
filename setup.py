from setuptools import setup, find_packages

with open("README.md") as readme:
    long_description = readme.read()

requirements = [

]

setup(
    name='cross-validation-package',
    version='1.0.0',
    license='MIT',
    author="Matteo Serafino",
    author_email='matteo.serafino1@gmail.com',
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url='https://github.com/matteo-serafino/cross-validation.git',
    keywords='cross-validation',
    install_requires=requirements,
    python_requires=">=3.6.2",
    include_package_data=True
)