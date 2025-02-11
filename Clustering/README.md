### Example Usage : 


Running with Aeon: 

bash

python main_benchmark.py BasicMotions --library aeon


Running with Tslearn:

bash

python main_benchmark.py BasicMotions --library tslearn


###### Explanation of Arguments

    <dataset_name>:
        Replace BasicMotions with the name of the dataset you want to process.
        The dataset must be one of the predefined datasets (see Available Datasets).
    --library:
        Specify which library to use for clustering. Options are:
            aeon: Use clustering methods implemented with the Aeon library.
            tslearn: Use clustering methods implemented with the Tslearn library.
