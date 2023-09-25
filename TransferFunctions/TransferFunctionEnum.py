from enum import Enum

class TransferFunctionEnum(Enum):
    StepFunctionPositive = "StepFunctionPositive"
    StepFunctionNegative = "StepFunctionNegative"
    LinealFunction = "LinealFunction"
    IdentityFunction = "IdentityFunction"