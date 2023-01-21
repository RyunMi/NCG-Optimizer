from setuptools import find_packages, setup

import os
import re

install_requires = [
    'torch>=1.5.0',
    'pytorch_ranger>=0.1.1',
]

def _read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), 'torch_optimizer', '__init__.py'
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        raise RuntimeError(
            'Cannot find version in torch_optimizer/__init__.py'
        )


setup(
    name='ncg-optimizer',
    version=_read_version(),
    description='Pytorch optimizer based on nonlinear conjugate gradient method',
    url='https://github.com/RyunMi/NCG-optimizer',
    author='Kerun Mi @ XTU',
    author_email='ryunxiaomi@gmail.com',
    license='MIT License',
    packages=find_packages(exclude=('tests',)),
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)