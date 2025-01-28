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
provide no significant overhead (or even do anything at all!) until enabled.

Modulus Profiling Tools
^^^^^^^^^^^^^^^^^^^^^^^

Modulus provides three high level tools for profiling utilities:

.. code-block:: python
    from modulus.utils.profiling import Profiler, profile, annotate

We'll get more into the details of some of these tools later, but at a high level:

#. The `Profiler` is the core utility of `modulus`'s profiling features.
Similar to the `modulus` DistributedManager, this class is a singleton instance
designed to enable easy access to profiling handles and propogate configurations.
The Profiler is meant to be your steering wheel for driving different profiling
techniques, without having to go into your code and change annotations or decorations.
We'll see how to enable the profiler with different techniques below. An instance of the 
Modulus Profiler can be used as a context for profiling.

#. `profile` is a function decorator that you can use to mark specific functions
in your code for profiling.  It is targeting tools like python's `line_profiler`
utility, and generally most useful for python operations that aren't backed by
an asynchronous backend like CUDA.  You can freely decorate functions with 
`@profile` but it is not a context manager.  `@profile` takes no arguments.

#. `annotate` is context-decorator and shortcut to `nvtx` annotations.
`annotate` can also be used as a context just like `nvtx.annotate`.

which will (in the modulus
profiling tools) turn on automatic annotation for the duration of the context.
This can be expensive, and has to be explicitly

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
- BS 1 - 0.030 (33.541 examples / s)
- BS 2 - 0.056 (35.544 examples / s)
- BS 4 - 0.117 (34.236 examples / s)
- BS 8 - 0.235 (34.074 examples / s)

inference:
- BS 1 - 0.029 (35.013 examples / s)
- BS 2 - 0.055 (36.545 examples / s)
- BS 4 - 0.114 (35.132 examples / s)
- BS 8 - 0.229 (34.881 examples / s)

..note::
    These numbers are approximate measurements and the workload, as written, 
    has some variations in performance.  Dont' worry if your results 
    differ - if you're following along with the example, just capture
    your baseline measurements to compare with later!

.. warning::
    These numbers should give you pause.  Most models don't have the
    same execution time in training vs. inference mode... 
    We will debug this below!


Python Level Profiling
^^^^^^^^^^^^^^^^^^^^^^

A great place to start when running profiles of AI code is the python level
profilers.  They are quick, easy, and generally low overhead.  Modulus
has support to enable `line_profiler` built in - all you might need is to
run `pip install line_profiler` to ensure you've installed the base package too.

Next, take a look at the first instrumented version of the model code, 
compared to the original:

.. code-block:: diff
    *** attn_baseline.py	2025-01-27 07:41:37.749753000 -0800
    --- attn_instrumented.py	2025-01-27 11:27:09.162202000 -0800
    ***************
    *** 1,6 ****
    --- 1,8 ----
      import torch
      from torch import nn

    + from modulus.utils.profiling import profile, annotate
    +
      class Attention(nn.Module):
          """Dummy example Attention mechanism.  Meant not for efficienct computation
          but to show how to use the profiling tools!
    ***************
    *** 26,31 ****
    --- 28,34 ----
              self.proj = nn.Linear(dim, dim)
              self.proj_drop = nn.Dropout(proj_drop)

    +     @profile
          def forward(self, x: torch.Tensor) -> torch.Tensor:

              B, N, C = x.shape
    ***************
    *** 59,64 ****
    --- 62,68 ----
              self.fc2 = nn.Linear(hidden_features, out_features)
              self.drop2 = nn.Dropout(drop)

    +     @profile
          def forward(self, x):
              x = self.fc1(x)
              x = self.gelu(x)
    ***************
    *** 97,102 ****
    --- 101,107 ----
                  drop=proj_drop,
              )

    +     @profile
          def forward(self, x: torch.Tensor) -> torch.Tensor:
              x = x + self.attn(self.norm1(x))
              x = x + self.mlp(self.norm2(x))

As you can see, we added a profile decorator around each forward pass.  Note that
this code will run just fine if you have these annotations but aren't profiling!

If you replace the model code with the instrumented code, nothing
significant will change.  To actually see changes, we have to enable the 
profiling tools dynamically at runtime:

.. note:: 
    We're running this via a configuration file and hydra - `pip install hydra-core`
    if it's not installed on your machine.  The config file is located in the same
    directory as the profiler and you can edit it to switch tools.

.. code-block:: python

    # configure the profiling tools:
    p = Profiler()
    
    print(p)
    
    for key, val in config.profile.items():
        # This is not the mandatory way to enable tools
        # I've set up the config to have the keys match
        # the registered profilers.  You can do it manually
        # too such as `p.enable("line_profiler")`
        if val: p.enable(key)
    
    # The profiler has to be initilized before use.  Using it in a context
    # will do it automatically, but to use it as a decorator we should do
    # it manually here:
    p.initialize()
    print(p)

    workload(config)

In the instrumented version of the workload, you can see we've even decorated 
the `workload` function itself.

.. note::
    The profiler interface - and all tools it pulls in - is unintialized until
    told otherwise.  If you don't call `p.initialize()`, the profiler will 
    generally do nothing.
 
.. warning::
    The one time the profiler will initialize itself is if you use it as a context!
    Upon entering the context, if the profiler interface isn't initialized
    it will trigger automatically.

Once the profiled run has completed, modulus will automatically deposit the outputs
into a folder `modulus_profiling_ouputs`.

.. note:: 
    You can change the location of the output.  Call `Profiler.output_dir(your_path)`
    before initialization.

Looking into the results, which for line profiler are a text file, we see the `workload`
function breakdown looks like this:

.. code-block::
    Total time: 2.41238 s
    File: /root/modulus/docs/test_scripts/profiling/workload_annotated.py
    Function: workload at line 30

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
        30                                           @profile
        31                                           def workload(cfg):
        32
        33         1     269136.0 269136.0      0.0      ds = RandomNoiseDataset(cfg["shape"])
        34
        35         2     172874.0  86437.0      0.0      loader = DataLoader(
        36         1        147.0    147.0      0.0          ds,
        37         1      30516.0  30516.0      0.0          batch_size=cfg["batch_size"],
        38         1        150.0    150.0      0.0          shuffle = True,
        39                                               )
        40
        41
        42                                               # Initialize the model:
        43         3   73464664.0    2e+07      3.0      model = Block(
        44         1      62983.0  62983.0      0.0          dim = cfg["shape"][-1],
        45         1      47585.0  47585.0      0.0          num_heads = cfg.model["num_heads"],
        46         1      38048.0  38048.0      0.0          qkv_bias  = cfg.model["qkv_bias"] ,
        47         1      37779.0  37779.0      0.0          attn_drop = cfg.model["attn_drop"],
        48         1      36155.0  36155.0      0.0          proj_drop = cfg.model["proj_drop"],
        49         1  349518830.0    3e+08     14.5      ).to("cuda")
        50
        51         1      57263.0  57263.0      0.0      if cfg["train"]:
        52                                                   opt = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        53
        54         1        228.0    228.0      0.0      times = []
        55         2      62432.0  31216.0      0.0      with Profiler() as p:
        56         1       1149.0   1149.0      0.0          start = time.perf_counter()
        57         9 1765021523.0    2e+08     73.2          for i, batch in enumerate(loader):
        58         8      14014.0   1751.8      0.0              image = batch["image"]
        59         8   52030436.0    7e+06      2.2              image = image.to("cuda")
        60        16     309466.0  19341.6      0.0              with annotate(domain="forward", color="blue"):
        61         8  169954425.0    2e+07      7.0                  output = model(image)
        62         8     581723.0  72715.4      0.0              if cfg["train"]:
        63                                                           opt.zero_grad()
        64                                                           # Compute the loss:
        65                                                           loss = loss_fn(output)
        66                                                           # Do the gradient calculation:
        67                                                           with annotate(domain="backward", color="green"):
        68                                                               loss.backward()
        69                                                               # Apply the gradients
        70                                                               opt.step()
        71         8      36510.0   4563.8      0.0              p.step()
        72         8      23381.0   2922.6      0.0              end = time.perf_counter()
        73         8     265785.0  33223.1      0.0              print(f"Finished step {i} in {end - start:.4f} seconds")
        74         8       5595.0    699.4      0.0              times.append(end - start)
        75         8       3939.0    492.4      0.0              start = time.perf_counter()
        76
        77         1      67726.0  67726.0      0.0      times = torch.tensor(times)
        78                                               # Drop first and last:
        79         1     139217.0 139217.0      0.0      avg_time = times[1:-1].mean()
        80                                               # compute throughput too:
        81         1      85639.0  85639.0      0.0      throughput = cfg["batch_size"] / avg_time
        82         1      36396.0  36396.0      0.0      print(f"Average time per iteration: {avg_time:.3f} ({throughput:.3f} examples / s)")

And of course, when presented like this, the issue is clear.  The dataloader is too slow!
Here' what's in the dataloader:

.. code-block:: python
    class RandomNoiseDataset(Dataset):
    """
    Random normal distribution dataset.
    
    Mean AND STD of the distribution is set to the index
    of the sample requested.
    
    Length is hardcoded to 64.
    
    (Don't use this anywhere that isn't an example of how to write non-performant python code!)
    
    """

    def __init__(self, image_shape, ):
        """
        Arguments:
            image_shape (string): Shape of a single example to generate
        """
        self.shape = image_shape

        self.rng = np.random.default_rng()

    def __len__(self):
        return 64

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Generate the raw data:
        raw = self.gen_single_image(idx)

        sample = {
            'image' : raw
        }
        
        return sample

    def gen_single_image(self, idx):
        
        return self.rng.normal(loc=idx, scale=idx, size=self.shape).astype(np.float32)

We can instrument this (and we should, soon) but the obvious problem is that the 
CPU is generating the data!  Let's convert it to GPU-driven data generation, and add
annotations while we're at it.  Instead of 
`return self.rng.normal(loc=idx, scale=idx, size=self.shape).astype(np.float32)`,
we can use
`return torch.normal(idx, idx, self.shape, device="cuda" )`


Running again with the fixed data loader:

train:
- BS 1 - 0.003 (322.914 examples / s)
- BS 2 - 0.004 (498.593 examples / s)
- BS 4 - 0.002 (1710.015 examples / s)
- BS 8 - 0.006 (1298.319 examples / s)

inference:
- BS 1 - 0.029 (35.013 examples / s)
- BS 2 - 0.055 (36.545 examples / s)
- BS 4 - 0.114 (35.132 examples / s)
- BS 8 - 0.229 (34.881 examples / s)


From the above, we can see at a python level that _most_ of our time is spent in 4 places:

#. First, significant time (11%) is spent in torch.optim.SGD.  This is a one-time call, and only 
looks expensive because our whole run only lasts 11.5s total.  Don't worry about this one.

#. Second, almost 15% of the execution time is spent in `for i, batch in enumerate(loader)`.  
We'll certainly revisit that, looks like our data load could be better.

#. Third, 24.5% of the time is in `output=model(image)` - for a total of about 11.5*0.245 = 2.8s.
Luckily, we instrumented the forward pass of the model already, we can inspect that next.

#. Finally, 47.2% of the time is spent in `loss.backward()`.  From our simple python profiling, 
we can gain no insights into the backward pass.  We'll dig deeper soon with other tools.

What's happening in the Forward Pass?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because we instrumented the forward pass already, the results are already available:

.. code-block :: 
    Total time: 2.79481 s
    File: /root/modulus/docs/test_scripts/profiling/attn_instrumented.py
    Function: forward at line 104

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       104                                               @profile
       105                                               def forward(self, x: torch.Tensor) -> torch.Tensor:
       106         8 2147933917.0    3e+08     76.9          x = x + self.attn(self.norm1(x))
       107         8  646863364.0    8e+07     23.1          x = x + self.mlp(self.norm2(x))
       108         8      11102.0   1387.8      0.0          return x

This result, from the `Block.forward` function, suggests the attn layer is 75% of the cost.

Looking closer, if `Attention.forward` shows this:

.. code-block :: 
    Total time: 1.92844 s
    File: /root/modulus/docs/test_scripts/profiling/attn_instrumented.py
    Function: forward at line 31

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
        31                                               @profile
        32                                               def forward(self, x: torch.Tensor) -> torch.Tensor:
        33
        34         8      47852.0   5981.5      0.0          B, N, C = x.shape
        35         8  167395353.0    2e+07      8.7          qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        36         8     391969.0  48996.1      0.0          q, k, v = qkv.unbind(0)
        37
        38
        39                                                   # This is not optimal code right here ...
        40         8   33681705.0    4e+06      1.7          q = q * self.scale
        41         8  636895850.0    8e+07     33.0          attn = q @ k.transpose(-2, -1)
        42         8  856165404.0    1e+08     44.4          attn = attn.softmax(dim=-1)
        43         8    1215481.0 151935.1      0.1          attn = self.attn_drop(attn)
        44         8  108034249.0    1e+07      5.6          x = attn @ v
        45
        46         8   61744601.0    8e+06      3.2          x = x.transpose(1, 2).reshape(B, N, C)
        47         8   62203870.0    8e+06      3.2          x = self.proj(x)
        48         8     645708.0  80713.5      0.0          x = self.proj_drop(x)
        49         8      18165.0   2270.6      0.0          return x

And here, the bulk of the time is spent in forming the attention matrix and taking the softmax.
Even though we've not instrumented the backwards pass, you can suspect that these two operations
will be important there too!

.. note ::
    If you are at all paying attention (pun intended) in the computational developments of the Flash Attention
    mechanism, you'll know already what's going on here.  Regardless, let's try to figure it out with
    profiling tools anyways!

To look a little deeper and see what is happening in the attention steps as well as the backward pass,
we'll need different tools that have access to under-the-python-hood profiling techniques.


Digression: Scalene and Custom Profilers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By the way - the python `line_profiler`, while powerful, is not the only profiler available
at a python level.  An even more powerful tool is 
`Scalene <https://github.com/plasma-umass/scalene/tree/master>`, which we can quickly 
enable in the modulus profiling tools.

In your tutorial space, add this file:


And, in the annotate workload, add this in the main loop to register the profiler:

.. code-block:: python

    from custom_profiler import CustomProfiler
    
    from modulus.utils.profiling import ProfileRegistry
    ProfileRegistry.register_profiler("custom", CustomProfiler)
    
    p.enable("custom")

Pytorch Profiler
^^^^^^^^^^^^^^^^^

A common and useful tool in AI profiling is the pytorch profiler.  Modulus wraps
this profiler too, and we can enable it easily with a flip of a configuration switch:


