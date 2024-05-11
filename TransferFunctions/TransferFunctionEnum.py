from enum import Enum

class TransferFunctionEnum(Enum):
    StepPositiveFunction = "StepFunctionPositive"
    StepNegativeFunction = "StepFunctionNegative"
    LinealFunction = "LinealFunction"
    IdentityPositiveFunction = "IdentityFunctionPositive"
    NoneFunction = "NoneFunction"
    StepPositiveDoubleFunction = "StepDoubleFunctionPositive"