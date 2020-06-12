from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='mlutil',
    version='0.22',
    description='Python helpers for common ml tasks',
    url='https://github.com/lambdaofgod/mlutils',
    author='Jakub Bartczuk',
    packages=find_packages(),
    install_requires=requirements
)
