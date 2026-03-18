from setuptools import find_namespace_packages, setup
setup(
    name='graph_eqa',
    version='0.1.0',
    packages=find_namespace_packages(include=["src", "src.*", "mapg_eqa", "mapg_eqa.*"]),
    install_requires=[
        "imageio",
        "omegaconf", 
        "rerun-sdk",
        "opencv-python",
        "openai", 
        "transformers",
        "scikit-image",
        "yacs",
        "networkx",
        "SentencePiece",
        "anthropic",
        "google-generativeai"
    ],
    include_package_data=True,
)
