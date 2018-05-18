## Sample code for MLInstrumention

[Titanic classifier in Keras](https://github.com/mlaicode/kerasTitanic)

## MLInstrumention: Machine Learning/Artificial intelligence code instrumentation

<b>MLInstrumentation</b> is a python package for instrumenting your ML/AI code. It has support for popular Keras/Tensorflow and FastAI libraries currently. And other deep learning libraries support is under development.

## Use Case
<b>MLInstrumentation</b>  package gathers metrics from a Linux/Windows/Solaris/AIX machine and reports to shared hosted Grafana dashboard. Users can select and filter the based on GroupName, FileName, etc to select data of their interest.

Examples of the metrics users can view and analyze:

* Program run duration/Training time
* Device types used for those - GPU/CPU
* Device details - TitanX, Intel Xeon, etc.
* Method call details - fit(), predict(), etc.
* Group name/Class name - custom tags including AIClass2018, etc.


## Installation
```bash
pip install mlinstrumentation
```

## Getting started

## Add start/stop methods

```python
   import mlinstrumentation
   mlinstrumentation.start()
   
   <your code>
   
   mlinstrumentation.stop()
```

## Query for the data
[Dashboard here](http://bit.ly/mlinstrumentation)

Sample image
![Dashboard ](assets/images/dashsnap.png)
