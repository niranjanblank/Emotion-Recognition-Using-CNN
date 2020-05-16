#####################To train the model using GPU:##################################
TensorFlow GPU support requires an assortment of drivers and libraries. To simplify 
installation and avoid library conflicts, we recommend using a TensorFlow Docker image
 with GPU support (Linux only). This setup only requires the NVIDIA® GPU drivers.

These install instructions are for the latest release of TensorFlow. See the tested 
build configurations for CUDA and cuDNN versions to use with older TensorFlow releases.

###################### Hardware requirements ########################################
The following GPU-enabled devices are supported:

NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher.


###################### Software requirements ########################################
The following NVIDIA® software must be installed on your system:

NVIDIA® GPU drivers — CUDA 10.1 requires 418.x or higher.
CUDA® Toolkit — TensorFlow supports CUDA 10.1 (TensorFlow >= 2.1.0)
CUPTI ships with the CUDA Toolkit.
cuDNN SDK (>= 7.6)
(Optional) TensorRT 6.0 to improve latency and throughput for inference on some models.