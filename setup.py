import setuptools

with open('README.md','r') as f:
    long_description = f.read()

setuptools.setup(
    name='pyDEER',
    version='1.0.6',
    author='Timothy Keller',
    author_email='tkeller@bridge12.com',
    description='A Python Package for Processing Double Electron-Electron Resonance (DEER) Data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tkellerBridge12/pyDEER',
    project_urls={
        'Source Code':'https://github.com/tkellerBridge12/pyDEER',
        'Documentation':'https://pydeer.readthedocs.io/',
        },
    packages=setuptools.find_packages(),
    install_requires = ['scipy','numpy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=2.7, >=3.6',
)
