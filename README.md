# pyASP - Ames Stereo Pipeline Python Wrappers

Python wrappers around NASA's Ames Stereo Pipeline (ASP) for automated DEM generation from satellite imagery.

## Features

- Easy-to-use Python API for ASP commands via the [`Command`](pyasp/utils/shell.py) class
- Pipeline system using [`Pipeline`](pyasp/pipeline.py), [`DelayedTask`](pyasp/pipeline.py) and [`ParallelBlock`](pyasp/pipeline.py) for chaining ASP operations
- Automatic ASP binary management
- Support for SPOT5 satellite imagery through specialized classes like [`ParallelStereo`](pyasp/steps.py)
- Parallel processing capabilities using Dask

## Installation

### Install from source

Create a new Conda/Mamba environment with Python=3.11:

```bash
mamba create -n pyasp python=3.11
mamba activate pyasp
```

Clone the repository and install the package:

```bash
git clone https://github.com/franioli/pyASP.git
cd pyASP
pip install -e .
```

### Docker Installation
Build and run using Docker:

```bash
docker compose up -d 
```

To enter the running container:

```bash
docker compose exec pyasp /bin/bash
```

Alternatively (recommended), you can use VSCode with the Remote - Containers extension to develop inside the container.
Use "Attach to Running Container" and select the pyasp container.

### Download ASP Binaries

PyASP requires the ASP binaries to be downloaded on your system. 
PyASP will automatically download the binaries for you when you first import the package. 

```python
import pyasp
```

Alterantively, you can manually download the binaries from the [ASP website](https://stereopipeline.readthedocs.io/en/latest/index.html) and set set the path to the binaries directly within PyASP with:

```python
import pyasp

pyasp.add_asp_binary("/path/to/asp/binaries")
```


## Basic Usage

You can check a fully working example in the demo.py file. 
It will run a basic process with a DEM generation from small cutouts of SPOT5 images (included in the demo folder).

```python
import pyasp
from pathlib import Path

# Setup logging
logger = pyasp.setup_pyasp_logger(log_level="info", log_to_file=True)

# Create a path manager for input/output paths
pm = pyasp.PathManager(
    seed_dem=Path("dem.tif"),
    output_dir=Path("output"),
    compute_dir=Path("compute")
)

# Create a pipeline
pipeline = pyasp.Pipeline()

# Add ASP processing steps
pipeline.add_step(
    pyasp.steps.BundleAdjust(
        images=["image1.tif", "image2.tif"],
        cameras=["cam1.xml", "cam2.xml"],
        output_prefix="ba",
        t="rpc"
    )
)

# Add more steps as needed
pipeline.add_step(
    pyasp.steps.Point2dem(
        input_file="pointcloud.tif",
        tr=10,
        t_srs="EPSG:32632"
    )
)

# Run the pipeline
pipeline.run() 
```

## PyASP Design and Implementation

### Core Components

#### 1. Command Class
The `Command` class (in `utils/shell.py`) provides an interface for running shell commands:

```python
# Create and run a command
cmd = Command("parallel_stereo --version")
cmd.run()

# Add arguments dynamically
cmd.extend("input1.tif", "input2.tif", t="rpc", max_level=2)
cmd.run()
```

Key features:

- Converts Python arguments to command-line format
- Handles different argument types (bool flags, key-value pairs)
- Provides execution timing and output capture
- Supports both positional and keyword arguments

#### 2. Pipeline System

The `Pipeline` class enables chaining multiple processing steps:

```python
# Create a pipeline
pipeline = pyasp.Pipeline()

# Add steps
pipeline.add_step(pyasp.steps.AddSpotRPC("metadata.dim"))
pipeline.add_step(pyasp.steps.BundleAdjust(images, cameras))

# Execute all steps
pipeline.run()
```

Features:

- Lazy initialization of steps
- Sequential or parallel execution
- Support for any step with a `run()` method
- `ParallelBlock` for concurrent processing

#### 3. ASP Step Architecture
Each ASP command is wrapped in a Python class inheriting from `AspStepBase`:

```python
# Example ASP step usage
step = MapProject(
    dem="dem.tif",              # Positional arg
    camera_image="image.tif",   # Positional arg
    camera_model="camera.xml",  # Positional arg
    output_image="output.tif",  # Positional arg
    t="rpc",                    # Keyword arg (--t in ASP)
    tr=10,                      # Keyword arg (--tr in ASP)
    threads=4                   # Keyword arg (--threads in ASP)
)
```

Parameter handling:

- Positional parameters map to ASP command positional args
- Keyword parameters map to ASP command options
- Dashes in ASP options become underscores in Python

### Implementing New ASP Steps
To add support for a new ASP command:

1. Create a new class inheriting from `AspStepBase`:
```python
class NewAspStep(AspStepBase):
    _required_params = ["input_file"]  # Required parameters

    def __init__(
        self,
        input_file: Union[str, Path],  # Positional parameters first
        verbose: bool = False,
        **kwargs,                      # Additional ASP options
    ):
        super().__init__(verbose=verbose)
        
        # Store constructor args for serialization
        self._constructor_args = {
            "input_file": input_file,
            "kwargs": kwargs,
        }

        # Create command
        command = Command(
            cmd="asp_command_name",
            name="asp_command_name",
            verbose=self._verbose
        )

        # Add positional args
        command.extend(input_file)

        # Add keyword args
        command.extend(**kwargs_to_asp(kwargs))

        # Set command for execution
        self._command = command
```

2. Add the new class to `COMMAND_CLASS_MAP`:
```python
COMMAND_CLASS_MAP = {
    "asp_command_name": NewAspStep,
    # ... other mappings
}
```

The step can now be used in pipelines:
```python
pipeline.add_step(NewAspStep("input.tif", option1="value"))
```