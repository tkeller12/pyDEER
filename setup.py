import setuptools

with open('README.md','r') as f:
    long_description = f.read()

setuptools.setup(
    name='pyDEER',
    version='1.0.1',
    author='Timothy Keller',
    author_email='tkeller@bridge12.com',
    description='A Python Package for Fitting Double Electron-Electron Resonance Data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://www.timothyjkeller.com/',
    packages=setuptools.find_packages(),
    install_requires = ['scipy','numpy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=2.7, >=3.6',
)
