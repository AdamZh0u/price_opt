### Product Price Optimization

This Streamlit application allows users to optimize product prices to maximize total profit. The app consists of three main steps: uploading input files, modifying the data interactively, and solving the optimization problem to display the results.

#### Features

1. **Upload Files**: Upload CSV files for `df_params` and `df_variables`.
2. **Interactive Data Editing**: Modify the uploaded data directly within the app.
3. **Optimization**: Solve the price optimization problem using Pyomo and IPOPT solver.
4. **Result Display**: View the optimized prices and maximum profit.

#### Installation

1. Ensure you have Python installed on your system.
2. Install the required Python packages:
3. Install the IPOPT solver. Instructions can be found on the [IPOPT website](https://coin-or.github.io/Ipopt/INSTALL.html).

#### Usage

1. Save the provided Python code to a file, e.g., `app.py`.
2. Run the Streamlit app:

    ```sh
    streamlit run app.py
    ```

3. Follow the steps in the web interface to upload `df_params` and `df_variables` files, modify the data if necessary, and solve the optimization problem.

#### Example

1. **Upload CSV Files**: Upload your `df_params` and `df_variables` CSV files.
2. **Modify Data**: Interactively adjust any values in the tables if needed.
3. **Optimize**: Click the "Solve Optimization" button to calculate the optimal prices and display the maximum profit.

#### Files

- `df_params.csv`: Contains parameters such as return rate, shipping cost, exchange rate, etc.
- `df_variables.csv`: Contains variables including product bounds, slopes, and intercepts.

#### License

This project is licensed under the MIT License.