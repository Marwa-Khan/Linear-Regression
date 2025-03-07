import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, learning_curve, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred):
    """ TO-DO """
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    print(mse)
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(rmse)
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    print(mae)
    # Calculate R² Score
    r2 = r2_score(y_true, y_pred)
    print(r2)

    return mse, rmse, mae, r2

def plot_learning_curve(model, X, y, cv):

    """ TO-DO """

    # Preprocess X to handle categorical variables
    preprocessor = setup_preprocessor(X)


    # Fit the preprocessor on the entire dataset
    preprocessor.fit(X)  
    # Create a pipeline that includes both preprocessing and regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Perform learning curve analysis
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Calculate mean and standard deviation of scores
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_std = test_scores.std(axis=1)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training score', marker='o')
    plt.plot(train_sizes, test_scores_mean, label='Validation score', marker='o')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Negative Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.show()


def setup_preprocessor(X):

    """ TO-DO """
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features= X.select_dtypes(include=['object','category']).columns.tolist()

#     # Create transformers for numeric and categorical features
#     numeric_transformer= Pipeline(steps=[('scalar', StandardScaler())])
#     categorical_transformer= Pipeline(steps=[('encoder', OneHotEncoder())])
#     # Combine transformers into a preprocessor
#     preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ]
# )

    transformers = []

    # Add numeric transformer if there are numeric features
    if numeric_features:
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        transformers.append(('num', numeric_transformer, numeric_features))

    # Add categorical transformer if there are categorical features
    if categorical_features:
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
        transformers.append(('cat', categorical_transformer, categorical_features))

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor

def simple_linear_regression(X, y, split_type, k=None):
    """ TO-DO """
     
    # Preprocess X using setup_preprocessor()
    preprocessor= setup_preprocessor(X)
    # The following code creates a pipeline that first preprocesses the data and then applies the linear regression model
    model=Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    if split_type =="train-test":
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train) #fit the model to the training data
        y_pred  = model.predict(X_test) # evaluate the model

        # Calculate and display metrics
        mse, rmse, mae, r2= calculate_metrics( y_test, y_pred)
        print("\nMetrics for Train-Test Split:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R² Score: {r2}")

    elif split_type=="train-val-test":
        # Train-Validation-Test Split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 20% of 80% = 20% overall
        # Fit the model on training data
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_metrics= calculate_metrics(y_val, y_val_pred)
        print("\nMetrics for Validation Set:", val_metrics)

        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        test_metrics= calculate_metrics(y_test, y_test_pred)
        print("\nMetrics for Test Set:", test_metrics)

    
    elif split_type == "k-fold":
          # Preprocess X using setup_preprocessor()
        preprocessor = setup_preprocessor(X)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        # K-Fold Cross Validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

        print(f"K-Fold Cross Validation Results (R² Scores): {scores}")
        print(f"Mean R² Score: {np.mean(scores)}")

        # Plot learning curve
        print("\nPlotting Learning Curve...")
        plot_learning_curve(model, X, y, kf)

    else:
        raise ValueError(f"Invalid method: {split_type}. Supported methods are 'train-test', 'train-val-test', and 'k-fold'.")


def multiple_linear_regression(X, y, split_type, k=None):
    """ TO-DO """

    # Preprocess X using setup_preprocessor()
    preprocessor= setup_preprocessor(X)
    # The following code creates a pipeline that first preprocesses the data and then applies the linear regression model
    model=Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])

    if split_type=="train-val-test":
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        model.fit(X_train, y_train) 
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for validation and test sets
        val_metrics = calculate_metrics(y_val, y_val_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        print("Metrics for Validation Set:", val_metrics)
        print("Metrics for Test Set:", test_metrics)


    elif split_type == "k-fold":
          # Preprocess X using setup_preprocessor()
        preprocessor = setup_preprocessor(X)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        # K-Fold Cross Validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        print(f"K-Fold Cross Validation Results (R² Scores): {scores}")
        print(f"Mean R² Score: {np.mean(scores)}")
        
        # Plot learning curve
        plot_learning_curve(model, X, y, kf)
    
    else:
        print("Invalid method selected.")

def main():
    """ TO-DO """
    # Step 1: Accept file path for the dataset from the user
    try:
        file_path=input("Enter the file path for the dataset: ")
        # Step 2: Load the dataset into a Pandas DataFrame
        df=pd.read_csv(file_path)
        print("Dataset loaded successfully.")

        # Step 3: Handle missing values appropriately
        df.dropna(inplace=True)
        print("Missing values handled successfully.")
        print("The updated dataset is:",df)

        # Step 4: Display all the columns available in the dataset using enumerate
        print("\nColumns in the dataset:")
        for index, column in enumerate(df.columns):
            print(f"{index+1}. {column}")
        # Step 5: Ask the user to enter the target variable
        target_idx = int(input("\nEnter the index of the target variable (from the list above): ")) - 1
        target_col = df.columns[target_idx]

        # Step 6: Ask the user if they are interested in simple linear regression
        choice= input("\nWant to perform Simple Linear Regression? Enter 1 for Yes or 0 for No: ")
        if choice=='1':
            predictor_idx=int(input("\nEnter the column number for the predictor variable: "))-1
            predictor_col=df.columns[predictor_idx]
            if predictor_col != target_col:
                print(f"\nPredictor variable selected: {predictor_col}")
                print(f"Performing Simple Linear Regression with predictor: {predictor_col} and target: {target_col}.")
                # Defining X and y for Simple Linear Regression
                X = df[[predictor_col]] # Single predictor variable. The double square brackets makes sure X is a DataFrame (2D), because otherwise it would cause an error during the fit() step
                y = df[target_col]  # Target variable
            else:
                print("Error: Predictor variable cannot be the same as the target variable. Please select a different predictor.")

        elif choice == '0':
            print("\nYou have chosen not to perform Simple Linear Regression.")
            # Define X and y for Multiple Linear Regression
            X=df.drop(columns=[target_col]) # All columns except the target variable
            y = df[target_col]  # Target variable
            print(f"X contains all columns except the target variable: {', '.join(X.columns)}")
            print(f"y is the target variable: {target_col}")

        else:
            print("\nInvalid choice entered. Please enter 1 for Yes or 0 for No.")
            exit()

        # Step 8: Display the modeling options to the user
        print("1: Simple Linear Regression with Train-Test Split")
        print("2: Simple Linear Regression with Train-Validation-Test Split")
        print("3: Simple Linear Regression with K-Fold Cross Validation")
        print("4: Multiple Linear Regression with Train-Validation-Test Split")
        print("5: Multiple Linear Regression with K-Fold Cross Validation")

        option = input("\nEnter the number corresponding to your choice: ")
        if option == '1':
            print("\nSelected: Simple Linear Regression with Train-Test Split")
            
            simple_linear_regression(X, y, "train-test")

        elif option== '2':
            print("\nSelected: Simple Linear Regression with Train-Validation-Test Split")
            simple_linear_regression(X, y, "train-val-test")


        elif option=='3':
            print("\nSelected: Simple Linear Regression with K-Fold Cross Validation")
            # Ask for the number of folds
            k = int(input("Enter the number of folds for K-Fold Cross Validation: "))
            if k > 1:
            # Call simple_linear_regression with k-fold cross validation
                simple_linear_regression(X, y, "k-fold", k=k)
            else:
                print("Invalid number of folds. Please enter a value greater than 1.")


        elif option=='4':
            print("\nSelected: Multiple Linear Regression with Train-Validation-Test Split")
            multiple_linear_regression(X, y, "train-val-test")


        elif option=='5':
            print("\nSelected: Multiple Linear Regression with K-Fold Cross Validation")
            k = int(input("Enter the number of folds for K-Fold Cross Validation: "))
            if k > 1:
            # Call multiple_linear_regression with k-fold cross validation
                multiple_linear_regression(X, y, "k-fold", k=k)
            else:
                print("Invalid number of folds. Please enter a value greater than 1.")

        else:
            print("Invalid choice entered.")



    except FileNotFoundError:
        print(f"Error: File not found at the specified path: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
