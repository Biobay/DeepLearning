from setuptools import setup, find_packages

setup(
    name="pikapikaGen",
    version="0.1.0",
    description="Text-to-Image Generator for Pokémon Sprites",
    author="Mario",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "pillow>=9.5.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "gradio>=3.35.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "notebook": ["jupyter", "ipywidgets"],
    },
)
