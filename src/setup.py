import setuptools

setuptools.setup(
    version="0.1.1",
    description="Ablation and Extension study of Neural Symbolic Regression that Scales",
    name="nesymres",
    packages=setuptools.find_packages("."),
    package_dir={"": "."},
    install_requires=[
        "numpy",
        "sympy",
        "pandas",
        "click",
        "tqdm",
        "numexpr",
        "jsons",
        "h5py",
        "scipy",
        "dataclass_dict_convert",
        "hydra-core==1.0.0",
        "ordered_set",
        "wandb",
        "torch",
        "pytorch_lightning",
    ],
)
