<a name="localconda"></a>
## Setup Azure Automated ML using a Local Conda environment

To run these notebook on your own notebook server, use these installation instructions.
The instructions below will install everything you need and then start a Jupyter notebook.

### 1. Install mini-conda from [here](https://conda.io/miniconda.html), choose 64-bit Python 3.7 or higher.
- **Note**: if you already have conda installed, you can keep using it but it should be version 4.4.10 or later (as shown by: conda -V).  If you have a previous version installed, you can update it using the command: conda update conda.
There's no need to install mini-conda specifically.

### 2. Downloading the sample notebooks
- Download the sample notebooks from [GitHub](https://github.com/Azure/MachineLearningNotebooks) as zip and extract the contents to a local directory.  The automated ML sample notebooks are in the "automated-machine-learning" folder.

### 3. Setup a new conda environment
The **automl_setup** script creates a new conda environment, installs the necessary packages, configures the widget and starts a jupyter notebook. It takes the conda environment name as an optional parameter.  The default conda environment name is azure_automl.  The exact command depends on the operating system.  See the specific sections below for Windows, Mac and Linux.  It can take about 10 minutes to execute.

Packages installed by the **automl_setup** script:
    <ul><li>python</li><li>nb_conda</li><li>matplotlib</li><li>numpy</li><li>cython</li><li>urllib3</li><li>scipy</li><li>scikit-learn</li><li>pandas</li><li>tensorflow</li><li>py-xgboost</li><li>azureml-sdk</li><li>azureml-widgets</li><li>pandas-ml</li></ul>

For more details refer to the [automl_env.yml](./automl_env.yml)
## Windows
Start an **Anaconda Prompt** window, cd to the **how-to-use-azureml/automated-machine-learning** folder where the sample notebooks were extracted and then run:
```
automl_setup
```
## Mac
Install "Command line developer tools" if it is not already installed (you can use the command: `xcode-select --install`).

Start a Terminal windows, cd to the **how-to-use-azureml/automated-machine-learning** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_mac.sh
```

## Linux
cd to the **how-to-use-azureml/automated-machine-learning** folder where the sample notebooks were extracted and then run:

```
bash automl_setup_linux.sh
```

### 4. Running configuration.ipynb
- Before running any samples you next need to run the configuration notebook. Click on [configuration](../../configuration.ipynb) notebook
- Execute the cells in the notebook to Register Machine Learning Services Resource Provider and create a workspace. (*instructions in notebook*)

### 5. Running Samples
- Please make sure you use the Python [conda env:azure_automl] kernel when trying the sample Notebooks.
- Follow the instructions in the individual notebooks to explore various features in automated ML.

### 6. Starting jupyter notebook manually
To start your Jupyter notebook manually, use:

```
conda activate azure_automl
jupyter notebook
```

or on Mac or Linux:

```
source activate azure_automl
jupyter notebook
```