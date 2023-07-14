from external_package.custom_model import CustomModel
from modulus.models import Module

model_from_library = CustomModel(16)
model_from_modulus = Module.factory("CustomModel")(16)
print(model_from_library)
print(model_from_modulus)
