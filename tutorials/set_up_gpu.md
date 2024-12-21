### **Tutorial: Setting up GPU**

#### **Objective**  
Fix the issue where PyTorch does not detect the GPU and CUDA, even though CUDA is installed on the system.

---

#### **Step 1: Verify PyTorch and CUDA Versions**

Check your PyTorch version, CUDA version, and GPU availability.

**Expected Output**  
- `torch.__version__`: PyTorch version installed.  
- `torch.version.cuda`: The CUDA version PyTorch was built with.  
- `torch.cuda.is_available()`: True if CUDA is properly set up, False otherwise.

```python
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected.")
```

---

#### **Step 2: Check Installed CUDA Version**

1. **Open a terminal or command prompt** on your system.
2. **Run the following command** to check your installed CUDA version:

   ```bash
   nvcc --version
   ```

The version output will help you confirm the system-installed CUDA version.  
**Note**: Ensure that the PyTorch version and the CUDA version installed on your system are compatible. For example, if PyTorch was installed with CUDA 12.1 (+cu121), the system CUDA version should match or be compatible (e.g., CUDA 12.1, 12.0, or later versions). You can check the version at [PyTorch Get Started](https://pytorch.org/get-started/locally/).

---

#### **Step 3: Install the Right Version**

![Install CUDA](imgs/install_cuda.png)  
