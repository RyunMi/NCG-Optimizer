from setuptools import find_packages, setup

import os
import re

install_requires = [
    'torch>=1.7.1',
]

def _read(f):
    with open(os.path.join(os.path.dirname(__file__), f)) as f_:
        return f_.read().strip()

def _read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), 'ncg_optimizer', '__init__.py'
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        raise RuntimeError(
            'Cannot find version in ncg_optimizer/__init__.py'
        )

keywords = [
    'ncg-optimizer',
    'pytorch',
    # optimizers
    'LCG',
    'FR',
    'PRP',
    'HS',
    'CD',
    'DY',
    'LS',
    'HZ',
    'HS_DY',
    # line search
    'Armijo',
    'Wolfe',
]

setup(
    name='ncg-optimizer',
    version=_read_version(),
    description='Pytorch optimizer based on nonlinear conjugate gradient method',
    long_description='\n\n'.join((_read('README.rst'), _read('CHANGES.rst'))),
    long_description_content_type='text/x-rst',
    url='https://github.com/RyunMi/NCG-optimizer',
    author='Kerun Mi',
    author_email='ryunxiaomi@gmail.com',
    license='Apache 2',
    keywords=keywords,
    packages=find_packages(exclude=('tests',)),
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',     
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)