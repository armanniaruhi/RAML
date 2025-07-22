# Setting Up the Conda Environment

To set up the required Conda environment, follow these steps:

1. **Download the dataset**  
   Download the dataset from [this Google Drive link](https://drive.google.com/file/d/1oLIfYo21w744liIEN9EZi5i4Ej4shDQX/view?usp=share_link), and copy the entire `dataset` folder into the root directory of the project.

2. **Install Conda**  
   Make sure you have Conda installed. If not, you can download and install it from the [official Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

3. **Set up the environment**  
   Open a terminal, navigate to the directory containing the `environment.yml` file, and run the following command:

   ```bash
   conda env create -f environment.yml
