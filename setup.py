from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='pylore',
    url='https://github.com/gerzin/pylore',
    author='Gerardo Zinno',
    author_email='zinno.gerardo.95@gmail.com',
    version='0.0.1',
    description='LOcal Rule-based Explanations',
    packages=['pylore'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "numba"
    ],
    extras_require={
        "dev": [
            "pytest > 3.7",
            "twine",
        ]
    }
)
