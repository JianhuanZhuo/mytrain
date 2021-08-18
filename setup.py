from setuptools import setup
from setuptools import find_packages


VERSION = '0.1.0'

setup(
    name='mytrain',  # package name
    version=VERSION,  # package version
    description='my package',  # package description
    packages=find_packages(),
    zip_safe=False,
)

