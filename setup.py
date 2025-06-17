from setuptools import setup, find_packages

setup(
    name='mlp',            
    version='0.1',          
    packages=find_packages(),   
    install_requires=[
        'numpy',
        'pandas',
        'pickle'
    ],
    author='steffan',
    description='Простая реализация MLP на numpy',
)