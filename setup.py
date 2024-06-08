from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as r:
    requirements = r.read().splitlines()

setup(
    name='P10-federated-learning',
    version='0.1.0',
    description='A framework that extracts mutational signatures from mutational catalogues in a federated setting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dunkedolmer/P10-federated-learning.git',
    author='Frederik Rasmussen, Kevin Risgaard Sinding',
    author_email='frasm19@student.aau.dk, ksindi19@student.aau.dk',
    license='MIT',
    packages=['federated-learning-mutational-signatures'],
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False
)