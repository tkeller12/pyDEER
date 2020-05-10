import setuptools

with open('README.md','r') as f:
    long_description = f.read()

setuptools.setup(
    name='',
    version='1.0.1',
    author='Timothy Keller',
    author_email='tkeller@bridge12.com',
    description='A Python Package for Double Electron-Electron Resonance Data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://www.timothyjkeller.com/',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
