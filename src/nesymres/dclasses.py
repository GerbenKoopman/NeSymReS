from dataclasses import dataclass, field
from types import CodeType
from typing import List, Tuple, Optional
from torch.distributions import Uniform, Normal, Distribution
from dataclass_dict_convert import dataclass_dict_convert
import torch

@dataclass
class Equation:
    code: CodeType
    expr: str
    coeff_dict: dict
    variables: list
    support: Optional[tuple] = None
    tokenized: Optional[list] = None
    valid: bool = True
    number_of_points: Optional[int] = None
    
@dataclass 
class NNEquation:
    numerical_values: torch.tensor
    tokenized: torch.tensor
    expr: List[str]


@dataclass
class DataModuleParams:
    max_number_of_points: int
    type_of_sampling_points: str
    support_extremes: Tuple
    constant_degree_of_freedom: int
    predict_c: bool
    distribution_support: str
    input_normalization: bool
    


# @dataclass
# class Dataset:
#     eqs: List[Equation]
#     config: dict
#     total_coefficients: list
#     total_variables: list
#     word2id: dict
#     id2word: dict
#     una_ops: list
#     bin_ops: list
#     rewrite_functions: list 
#     unique_index: set = None
#     total_number_of_eqs: int = 0
@dataclass
class GeneratorDetails:
    max_len: int
    operators: str
    max_ops: int
    #int_base: int
    #precision: int
    rewrite_functions: str
    variables: list
    eos_index: int
    pad_index: int

@dataclass
class DatasetDetails:
    #eqs: List[Equation]
    config: dict
    total_coefficients: list
    total_variables: list
    word2id: dict
    id2word: dict
    una_ops: list
    bin_ops: list
    rewrite_functions: list 
    total_number_of_eqs: int
    eqs_per_hdf: int
    generator_details: GeneratorDetails
    unique_index: Optional[set] = None
    


@dataclass
class BFGSParams:
    activated: bool = True
    n_restarts: int = 10
    add_coefficients_if_not_existing: bool = False
    normalization_o: bool = False
    idx_remove: bool = True
    normalization_type: str = "MSE"
    stop_time: int = int(1e9)

@dataclass
class FitParams:
    word2id: dict
    id2word: dict
    total_coefficients: list
    total_variables: list
    rewrite_functions: list
    una_ops: Optional[list] = None
    bin_ops: Optional[list] = None
    bfgs: BFGSParams = field(default_factory=BFGSParams)
    beam_size: int = 2
    length_penalty: float = 1.0
    max_len: int = 100
    
# @dataclass
# class ConstantsOptions:
#     max_constants: int
#     min_additive_constant_support: float
#     max_additive_constant_support: float 
#     min_multiplicative_constant_support: float
#     max_multiplicative_constant_support: float