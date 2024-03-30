from TransferFunctions.LinealTransferFunction import LinealTransferFunction
from TransferFunctions.PercentageTransferFunction import PercentageTransferFunction
from TransferFunctions.IdentityTransferFunction import IdentityTransferFunction
from TransferFunctions.TransferFunction import TransferFunction
from TransferFunctions.TransferFunctionEnum import TransferFunctionEnum

class TransferFunctionFactory():
    def create_transfer_function(selected_transfer_function: TransferFunctionEnum) -> TransferFunction:
        if selected_transfer_function == TransferFunctionEnum.StepFunctionPositive:
            return PercentageTransferFunction(0.65, False)
        elif selected_transfer_function == TransferFunctionEnum.StepFunctionNegative:
            return PercentageTransferFunction(0.65, True)
        elif selected_transfer_function == TransferFunctionEnum.LinealFunction:
            return LinealTransferFunction()
        elif selected_transfer_function == TransferFunctionEnum.IdentityFunctionPositive:
            return IdentityTransferFunction(True)
        elif selected_transfer_function == TransferFunctionEnum.NoneFunction:
            return IdentityTransferFunction(False)