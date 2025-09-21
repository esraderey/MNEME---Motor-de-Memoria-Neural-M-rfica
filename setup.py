"""
Setup script for MNEME - Motor de Memoria Neural Mórfica
"""

from setuptools import setup, find_packages
import os

# Leer README
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Leer requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mneme",
    version="1.0.0",
    author="Esraderey and Raul Cruz Acosta",
    author_email="mneme@yourdomain.com",
    description="Motor de Memoria Neural Mórfica - Sistema avanzado de memoria computacional con síntesis determinista",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mneme",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/mneme/issues",
        "Source": "https://github.com/yourusername/mneme",
        "Documentation": "https://github.com/yourusername/mneme/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU",
        "Environment :: CPU",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "gpu": [
            "cupy>=9.0",
            "nvidia-ml-py3>=7.0",
        ],
        "security": [
            "cryptography>=3.0",
            "pycryptodome>=3.10",
        ],
        "optimization": [
            "numba>=0.50",
            "cython>=0.29",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
            "cupy>=9.0",
            "nvidia-ml-py3>=7.0",
            "cryptography>=3.0",
            "pycryptodome>=3.10",
            "numba>=0.50",
            "cython>=0.29",
        ],
    },
    entry_points={
        "console_scripts": [
            "mneme-benchmark=mneme.benchmark:main",
            "mneme-optimize=mneme.optimize:main",
            "mneme-security=mneme.security:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mneme": [
            "*.py",
            "*.md",
            "LICENSE",
        ],
    },
    zip_safe=False,
    keywords=[
        "memory",
        "compression",
        "neural-networks",
        "tensor-decomposition",
        "machine-learning",
        "pytorch",
        "optimization",
        "security",
        "cryptography",
    ],
    license="BUSL-1.1",
    platforms=["any"],
    # Metadatos adicionales
    provides=["mneme"],
    requires_python=">=3.8",
    # Configuración de build
    setup_requires=[
        "setuptools>=45",
        "wheel>=0.36",
    ],
    # Configuración de tests
    test_suite="tests",
    tests_require=[
        "pytest>=6.0",
        "pytest-cov>=2.0",
    ],
    # Configuración de desarrollo
    cmdclass={},
    # Configuración de distribución
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
    # Metadatos de PyPI
    download_url="https://github.com/yourusername/mneme/archive/v1.0.0.tar.gz",
    # Configuración de compatibilidad
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tensorly>=0.7.0",
        "lz4>=3.0.0",
        "xxhash>=2.0.0",
        "psutil>=5.8.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    # Configuración de seguridad
    dependency_links=[],
    # Configuración de versionado
    use_scm_version=False,
    # Configuración de build
    build_requires=[
        "setuptools>=45",
        "wheel>=0.36",
    ],
)