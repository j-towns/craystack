from setuptools import setup

setup(
    name='craystack',
    version='0.0',
    description='Compression tools for machine learning researchers',
    author='Jamie Townsend and Thomas Bird and Julius Kunze',
    author_email='jamiehntownsend@gmail.com',
    packages=['craystack'],
    install_requires=['numpy', 'scipy', 'autograd'],
    url='https://github.com/j-towns/craystack',
    license='MIT',
)
