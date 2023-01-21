from setuptools import find_packages, setup

import os
import re

install_requires = [
    'torch>=1.5.0',
]

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


setup(
    name='ncg-optimizer',
    version=_read_version(),
    description='Pytorch optimizer based on nonlinear conjugate gradient method',
    url='https://github.com/RyunMi/NCG-optimizer',
    author='Kerun Mi @ XTU',
    author_email='ryunxiaomi@gmail.com',
    license='Apache 2',
    packages=find_packages(exclude=('tests',)),
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)