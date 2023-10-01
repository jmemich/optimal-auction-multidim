import os
from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as _in:
        return _in.read()


setup(
    name='auction',
    version='1.0',
    description='Approximating optimal auctions in multidimensional settings',
    author='James Michelson & Alexey Kushnir',
    author_email='jamesmic@andrew.cmu.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=read('requirements.txt').splitlines()
)
