from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="lmserver",
    version="0.0.1",
    url="https://github.com/lambdaofgod/lmserver",
    author="Jakub Bartczuk",
    packages=find_packages(),
    install_requires=requirements,
)
