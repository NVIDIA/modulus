# Base class Animal
class Animal:
    def foo(self):
        R.static_method()

# Default dependency R
class R:
    @staticmethod
    def static_method():
        print("Called R.static_method()")

# Derived class Dog
class Dog(Animal):
    pass

# New dependency class to replace R
class NewR:
    @staticmethod
    def static_method():
        print("Called NewR.static_method()")

print(dir(Dog))

# Override R in Dog's scope
Dog.R = NewR
print(dir(Dog))

# Example usage
dog_instance = Dog()
dog_instance.foo = lambda: Dog.R.static_method()  # Override `foo` for Dog instance

# Test the behavior
dog_instance.foo()  # Output: Called NewR.static_method()