Simple Training and Inference recipe
=====================================

In this tutorial, we will see how to use utilites from Modulus to setup a simple model
training pipeline. Once the initial setup is complete, we will look into optimizing
the training loop, and also run it in a distributed fashion. 
We will finish the tutorial with an inference workflow that will demonstrate how to use
Modulus models in inference.


Basic Training workflow
------------------------

Let's get started. For the purposes of this tutorial, we will focus more on the Modulus
utilities and not the correctness of the problem definition or the results. A typical
training workflow requires data, a trainable model and an optimizer to update the model
parameters. 


Using built-in models
^^^^^^^^^^^^^^^^^^^^^^

In this example, we will look at different ways one can interact with Models in Modulus.
Modulus presents a library of models suitable for Physics-ML applications for you to
use directly in your training workflows. In this tutorial we will see how to use a
simple model in Modulus to setup a data-driven training. Using the models from Modulus
will enable us to use various other Modulus features like optimization and 
quality-of-life functionalites like checkpointing and model entrypoints.

Later we will also see how to customize these models in Modulus.

In this example we will use the 
FNO model from Modulus. To demonstrate the training using this model, we would need some
dataset to train the model. To allow for fast prototyping of models, Modulus provides
a set of benchmark datasets that can be used out of the box without the need to setup
data-loading pipelines. In this example, we will use one such datapipe called `Darcy2D`
to get the training data. 

Let's start with importing a few utils and packages. 

.. literalinclude:: ../test_scripts/test_basic.py
   :language: python
   :start-after: [imports] 
   :end-before: [imports]

In this example we want to develop a mapping between the permeability and its subsequent
pressure field for a given forcing function. Refer :ref:`Modulus Datapipes` for
additional details.

Then a simple training loop for this example can be written as follows:

.. literalinclude:: ../test_scripts/test_basic.py
   :language: python
   :start-after: [code] 
   :end-before: [code]

That's it! This shows how to use a model from Modulus. Most of the models in Modulus are
highly configurable allowing you to use them out-of-the-box for different applications.
Refer :ref:`Modulus Models` for a more complete list of available models. 

Using custom models in Modulus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modulus provides a lot of pre-built optimized models. However,
there might be times where the shipped models might not serve your application. In such
cases, you can easily write your own models and have them interact with the other Modulus
utilites and features. Modulus uses PyTorch in the backend and most Modulus models are,
at the core, PyTorch models. In this section we will see how to go from a typical PyTorch
model to a Modulus model. 

Let's get started with the same application of Darcy problem. Let's write a simple UNet
to solve the problem. A simple PyTorch model for a UNet can be written as shown below:

.. literalinclude:: ../test_scripts/test_custom_model_demo_1.py
   :language: python
   :start-after: [pytorch model] 
   :end-before: [pytorch model]

Let's now convert this to a Modulus Model. Modulus provides ``Module`` class that is
designed to be a drop-in replacement for the ``torch.nn.module``. Along with that, you
need to also pass a ``MetaData`` that captures the optimizations and other features
supported by the model. Using the ``Module`` subclass allows using these optimizations, 
and other features like checkpointing etc. from Modulus. 

Thus, converting a PyTorch model to a Modulus model is very simple. For the above model,
the diff would look something like below:

.. code-block:: diff

   -    import torch.nn as nn
   +    from dataclasses import dataclass
   +    from modulus.models.meta import ModelMetaData
   +    from modulus.models.module import Module 
   
   -    class UNet(nn.Module):
   +    @dataclass
   +    class MetaData(ModelMetaData):
   +        name: str = "UNet"
   +        # Optimization
   +        jit: bool = False
   +        cuda_graphs: bool = True
   +        amp_cpu: bool = True
   +        amp_gpu: bool = True
   +    
   +    class UNet(Module):
            def __init__(self, in_channels=1, out_channels=1):
   -            super(UNet, self).__init__()
   +            super(UNet, self).__init__(meta=MetaData())
   
                self.enc1 = self.conv_block(in_channels, 64)
                self.enc2 = self.conv_block(64, 128)


With simple changes like this you can convert a PyTorch model to a Modulus Model!

.. note::

   The optimizations are not automatically applied. The user is responsible for writing
   the model with the optimizations supported. However, if the models supports the
   optimization and the same is captured in the MetaData, then the downstream features
   will work out-of-the-box. 

.. note::

   For utilizing the checkpointing functionality of Modulus, the Model instantiation
   arguments must be json serializable.  


You can also use a Modulus model as a standard PyTorch model as they are interoperable.


Let's say you don't want to make changes to the code, but you have a PyTorch model
already. You can convert it to a Modulus model by using the ``modulus.Module.from_torch``
method. This is described in detail in :ref:`modulus-models-from-torch`. 

.. literalinclude:: ../test_scripts/test_custom_model_demo_1.py
   :language: python
   :start-after: [modulus model] 
   :end-before: [modulus model]


And just like that you can use your existing PyTorch model as a Modulus Model.
A very similar process can be followed to convert a Modulus model to a Modulus Sym model
so that you can use the Constraints and other defitions from the Modulus Sym repository.
Here you will use the ``Arch`` class from Modulus Sym that provides utilites and methods
to go from a tensor data to a dict format which Modulus Sym uses. 

.. literalinclude:: ../test_scripts/test_custom_model_demo_1.py
   :language: python
   :start-after: [modulus sym model] 
   :end-before: [modulus sym model]


Optimized Training workflow
----------------------------

Once we have a model defined in the Modulus style, we can use the optimizations 
like AMP, CUDA Graphs, and JIT using the ``modulus.utils.StaticCaptureTraining`` decorator.
This decorator will capture the training step function and optimize it for the specified
optimizations.

.. note::
    The ``StaticCaptureTraining`` decorator is still under development and may be
    refactored in the future.


.. literalinclude:: ../test_scripts/test_custom_model_demo_1.py
   :language: python
   :start-after: [code] 
   :end-before: [code]


Distributed Training workflow
------------------------------

Modulus has several Distributed utilites to simplify the implementation of parallel training
and make inference scripts easier by providing a unified way to configure and query parameters
associated with distributed environment.

In this example, we will see how to convert our existing workflow to use data-parallelism.
For an deep-dive on Modulus Distributed utilities, refer :ref:`Modulus Distributed`.


.. literalinclude:: ../test_scripts/test_simple_distributed.py
   :language: python
   :start-after: [code] 
   :end-before: [code]



.. _running-inference-on-trained-models:

Running infernece on trained models
------------------------------------

Running inference on trained model is simple! This is shown by the code below. 

.. literalinclude:: ../test_scripts/test_basic_inference.py
   :language: python
   :start-after: [code] 
   :end-before: [code]

The static capture and distributed utilities can also be used during inference for
speeding up the inference workflow, but that is out of the scope for this tutorial.