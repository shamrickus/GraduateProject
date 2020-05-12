# GRNN in TVM

## Building
### Requirements
* Python 3
* CUDA
  
Optionally:
* GRNN
    * CMake/Make
    * GCC
* Design Document
    * Make
    * Xelatex
* [TVM](https://tvm.apache.org/docs/install/from_source.html)
### Release
* Run
* ```
    git clone {repo}/GraduateProject --depth 1
    cd GraduateProject
    git submodule init
    git submodule update
    python -m pip install -r requirements.txt
    ```

### Dev Notes
* Build [TVM](https://tvm.apache.org/docs/install/from_source.html)
* Run
* ``` 
  chmod +x .sourcerc
   ./.sourcerc ```
* You may have add PYTHONPATH to your IDE interpreter manually (see `.sourcerc`)


### Design-document
* Run `make`
