Profiling Applications in Modulus
==================================

In this tutorial, we'll learn about profiling your scientific AI applications in Modulus.
First, we'll discuss the high-level philosophy of profiling AI applications, then get 
in to the profiling tools available to understand performance.  We'll look at the 
features available in modulus to deploy these tools easily on your model, and finally
we'll walk through a full example of profiling a Modulus model (FigConvNet) with these 
tools.

Profiling Techniques
---------------------

Before we get in to the details of profiling AI applications in Modulus, it's worth
discussing a few general techniques and terminologies of profiling - and especially 
how they relate to profiling AI applications and Modulus.  There are some links
to other resources here as well, and deeper resources on profiling are abudant.  For 
example, 
`Nvidia's Nsight User Guide <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>`
is a useful reference for the many options available.  In many profiling guides, you'll
find the following terms discussed:

#. **Profiling Overhead** When you run an application with profiling enabled, 
be aware that you are nearly always *changing* the application and it's performance.
The amount an application's performance is affected by the profiling tools is often
called the **profiling overhead** and can, potentially, skew profiling results
and conclusions.  In many cases, this is just fine and even necessary.  
In general, profiling tools try to limit performance overhead. For AI applications,
overhead can vary by profiling tool (python vs. lower-level profilers, for example)
as well the type of profiler (sampling, tracing, or another)

#. **Hotspots** in an application are performance critical regions that the application
spends significant time in.  Application optimization is a data-driven endeavor, 
and the **hotspots** of an application are typically the highest priority code regions
for optimization.  The goal of profiling is first to discover hotspots, and subsequently
to find optimizations and measure their impact.  For AI Applications, hotspots
can take many forms and not all can be found in one pass.  As an example, a poorly
configured dataloader can easily stall an AI application by starving the GPU
of data.  In this case, no amount of CUDA optimization will make a significant
impact on application performance.  On the other hand, if your application
uses custom kernels or experimental features, they may be a hotspot and target 
for tuning.  These two examples typically require separate profiling tools 
to discover effectively in AI applications.

#. **Sampling** is the process of periodically asking an application, "What are 
you doing right now?".  Sampling-based profiling techniques can be very powerful:
they are low-overhead (small impact on application's performance) and when 
analyzed with statistical methods, they can quickly identify hotspots.  Sampling
can sometimes be considered a "bottom-up" view of an application.  Profilers with
sampling capabilities include 
`Nvidia's Nsight <https://developer.nvidia.com/nsight-systems>`,
`HPCToolkit <https://hpctoolkit.org/index.html>`,
`Tau <https://www.cs.uoregon.edu/research/tau/home.php>`,
`Scalene <https://github.com/plasma-umass/scalene>`, and many others.  For AI
applications, sampling both Python and CUDA code can be beneficial.

#. **Tracing** an application is the "top-down" profiling of an application and 
the logical opposite of sampling.  When **tracing** an application, the profilers
follow execution call paths and measure execution time.  Depending on the 
application, tracing can have a relatively high overhead but also provides
more detailed information for function execution times.  A common and useful
tracing tool in AI Applications if the profiler built in to 
`pytorch <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`
but other tools also offer tracing capabilities: 
`Nvidia's Nsight <https://developer.nvidia.com/nsight-systems>`,
Python's built-in `CProfile <https://docs.python.org/3/library/profile.html>`,
and others.

#. **Instrumentation** is the act of modifying your code to produce more useful 
profiling results.  A familiar and accesible (but not recommended) method of
**instrumentation** is to simply wrap your code blocks of interest in timing-
measuring logic (like using `std::chrono::steady_clock::time_point` in C++
or `time.perf_countner()` in Python).
Instrumentation can sometimes be done semi-automatically,
though even in those cases it often requires programmer intervention to insert
(and later, possibly, remove) instrumentation.  A useful instrumentation tool 
for Python level code is, for example, 
`line_profiler <https://github.com/pyutils/line_profiler>`
which requires manually decorating functions to be profiled, and executing them
in a specific context.

#. **Annotations** are a 
common and useful type of instrumentation: programmers can **annotate** their
application with sensible, user-friendly names and regions to more easily
detangle the results of profiling and map back to application code.  One of the 
most useful annotation tools in AI Applications is `Nvidia's Tools Extension
Library <https://github.com/NVIDIA/NVTX>`, available in both C and Python.
Annotations are also useful for marking regions or events of specific interest.


#. **Application Timelines** are a visual representation of an application's 
operations, as well as the order they are executed and - potentially - visualization
of idle time and unusued compute resources.  Often the **timeline** is the end 
product of a profiler (though not always) and robust visualization tools
such as `Nsight <>` as well as the `Perfetto Viewer <https://perfetto.dev/>` 
built into chrome can be critical to producing actionable output when profiling.

#. **Asynchronous Execution and Concurrency** is an incredibly powerful tool 
for obtaining high application performance, but also a potential challenge 
in capturing application profiles and reliably detecting hotspots and 
performance bottlenecks.  Specifically, **concurrency** can mislead you in analyzing
total time spent in functions (does a function with 100x calls more than 
another function constitute a bottleneck?  Are they all running and 
finishing concurrently?), while **asynchronous execution** can lead you 
to the wrong conclusions in sampling-based profiling methods.  A very important
example of this, in AI Application performance measurements, is the fact
that Pytorch submits kernels to GPUs asynchronously and returns control flow
to python immediately, and imperatively.  This can lead to situations where one
pytorch operation, `B`` is awaiting the output of another operation `A` as input.
A python-level, sampling-based profile will indicate that the application is 
spending too much time in operation `B`, while the reality may be that kernel `A`
is the true hotspot.  Mixed-language tracing tools, such as pytorch's own 
`profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`,
are powerful ways to visualize

Profiling AI Applications and Modulus
--------------------------------------

As described in brief above, many of the tools and tricks of profiling applications
have corner cases and sharp edges when applied to AI Applications.  First and foremost,
the incredible benefits of executing GPU kernels asynchronously leads to misleading
conclusions about hotspots if you use the wrong tool.  Second, 
mixed-language applications like Python + C++ + CUDA seen in most AI workloads can make
it challenging to see entire application performance at once.  

Because of these challenges, and because the end goal of profiling is to study an
application to determine hotspots that the user can take action on, profiling AI
workloads almost always employs multiple tools in multiple passes.  In Modulus, we have
designed profiling tool integration to make profiling your workloads straightforward,
without introducing additional overhead, and we've designed these tools to be "leave-in".

In the rest of this tutororial, we'll look at how to use Modulus's profiling tools to 
profile a simple application, as well as how to extend the tools to new profilers.
Modulus's tools are designed to be inserted into your workload once, and generally
provide no overhead or even do anything else until enabled.

The Workload
^^^^^^^^^^^^

This is a toy workload, using an Attention layer on randomly generated data. But, to
make it look like the tools you're familiar with in Modulus, it is configured with
Hydra and uses the usual tools (Pytorch Dataloader, for example).  Here's the workload:

.. literalinclude:: ../test_scripts/profiling/workload_raw.py
    :language: python

The Attention layer is fairly simple:

.. literalinclude:: ../test_scripts/profiling/attn_baseline.py
    :language: python

And the dataset is even more crude, using randomly generated data from numpy:

.. literalinclude:: ../test_scripts/profiling/dataset.py
    :language: python

Baseline measurements
^^^^^^^^^^^^^^^^^^^^^

One small piece of instrumentation this code already does - as many training models do -
is to instrument the main loop to capture the time it takes per batch.  The table below captures
some information about the performance as a function of batch size and train/inference.
This data was captured on an H100 platform, using the 24.12 pytorch container.

.. list-table:: Title
   :widths: 25 25 50
   :header-rows: 1
    
    * - Heading row 1, column 1
    - Heading row 1, column 2
    - Heading row 1, column 3
    * - Row 1, column 1
    -
    - Row 1, column 3
    * - Row 2, column 1
    - Row 2, column 2
    - Row 2, column 3

train:
- BS 1 - 0.11s (9.1)
- BS 2 - 0.21s (9.3)
- BS 4 - 0.52s (7.66)
- BS 8 - 0.923s (8.66)

inference:
- BS 1 - 0.07s (14.3)
- BS 2 - 0.133s (15.1)
- BS 4 - 0.286s (14.0)
- BS 8 - 0.568s (14.1)


- BS 8 - 0.61s


Let's apply some hooks into both the model and the workload to do some profiling.  
The profiling tools are imported from modulus:

.. code-block:: python
    from modulus.utils.profiling import Profiler, profile, annotate

which enables you to use them in your code later.  `profile` is a decorator 
and `annotate` is context-decorator.  You can mark functions for profiling
(instrumenting them) with `@profile`.  Similarly, you can annotate functions
for use in other profilers (like NSight) with `@annotate`, but `annotate` 
can also be used freely as a context:

.. code-block:: python

    with annotate(color="blue"):
        output = model(input)

One thing to note in this workload is that the profiler configuration is done
dynamically:

.. code-block:: python
    # configure the profiling tools:
    p = Profiler()
    
    for key, val in config.profile.items():
        # This is not the mandatory way to enable tools
        # I've set up the config to have the keys match
        # the registered profilers.  You can do it manually
        # too such as `p.enable("torch")`
        if val: p.enable(key)
    
    # The profiler has to be initilized before use.  Using it in a context
    # will do it automatically, but to use it as a decorator we should do
    # it manually here:
    p.initialize()