from skbuild import setup

setup(
    name="lumin-matrix",
    version="0.1.3",
    author="Philip Wisniewski",
    description="High-performance matrix operations library with multiple backends",
    packages=["lumin"],
    cmake_install_dir="lumin", 
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ],
)