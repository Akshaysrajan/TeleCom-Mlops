# import great_expectations as ge
# from typing import Tuple, List


# def validate_telco_data(df) -> Tuple[bool, List[str]]:
#     """
#     Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
#     This function implements critical data quality checks that must pass before model training.
#     It validates data integrity, business logic constraints, and statistical properties
#     that the ML model expects.
    
#     """
#     print("🔍 Starting data validation with Great Expectations...")
    
#     # Convert pandas DataFrame to Great Expectations Dataset
#     ge_df = ge.dataset.PandasDataset(df)
    
#     # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
#     print("   📋 Validating schema and required columns...")
    
#     # Customer identifier must exist (required for business operations)  
#     ge_df.expect_column_to_exist("customerID")
#     ge_df.expect_column_values_to_not_be_null("customerID")
    
#     # Core demographic features
#     ge_df.expect_column_to_exist("gender") 
#     ge_df.expect_column_to_exist("Partner")
#     ge_df.expect_column_to_exist("Dependents")
    
#     # Service features (critical for churn analysis)
#     ge_df.expect_column_to_exist("PhoneService")
#     ge_df.expect_column_to_exist("InternetService")
#     ge_df.expect_column_to_exist("Contract")
    
#     # Financial features (key churn predictors)
#     ge_df.expect_column_to_exist("tenure")
#     ge_df.expect_column_to_exist("MonthlyCharges")
#     ge_df.expect_column_to_exist("TotalCharges")
    
#     # === BUSINESS LOGIC VALIDATION ===
#     print("   💼 Validating business logic constraints...")
    
#     # Gender must be one of expected values (data integrity)
#     ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    
#     # Yes/No fields must have valid values
#     ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
#     ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
#     ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
#     # Contract types must be valid (business constraint)
#     ge_df.expect_column_values_to_be_in_set(
#         "Contract", 
#         ["Month-to-month", "One year", "Two year"]
#     )
    
#     # Internet service types (business constraint)
#     ge_df.expect_column_values_to_be_in_set(
#         "InternetService",
#         ["DSL", "Fiber optic", "No"]
#     )
    
#     # === NUMERIC RANGE VALIDATION ===
#     print("   📊 Validating numeric ranges and business constraints...")
    
#     # Tenure must be non-negative (business logic - can't have negative tenure)
#     ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    
#     # Monthly charges must be positive (business logic - no free service)
#     ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    
#     # Total charges should be non-negative (business logic)
#     ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    
#     # === STATISTICAL VALIDATION ===
#     print("   📈 Validating statistical properties...")
    
#     # Tenure should be reasonable (max ~10 years = 120 months for telecom)
#     ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
#     # Monthly charges should be within reasonable business range
#     ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    
#     # No missing values in critical numeric features  
#     ge_df.expect_column_values_to_not_be_null("tenure")
#     ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    
#     # === DATA CONSISTENCY CHECKS ===
#     print("   🔗 Validating data consistency...")
    
#     # Total charges should generally be >= Monthly charges (except for very new customers)
#     # This is a business logic check to catch data entry errors
#     ge_df.expect_column_pair_values_A_to_be_greater_than_B(
#         column_A="TotalCharges",
#         column_B="MonthlyCharges",
#         or_equal=True,
#         mostly=0.95  # Allow 5% exceptions for edge cases
#     )
    
#     # === RUN VALIDATION SUITE ===
#     print("   ⚙️  Running complete validation suite...")
#     results = ge_df.validate()
    
#     # === PROCESS RESULTS ===
#     # Extract failed expectations for detailed error reporting
#     failed_expectations = []
#     for r in results["results"]:
#         if not r["success"]:
#             expectation_type = r["expectation_config"]["expectation_type"]
#             failed_expectations.append(expectation_type)
    
#     # Print validation summary
#     total_checks = len(results["results"])
#     passed_checks = sum(1 for r in results["results"] if r["success"])
#     failed_checks = total_checks - passed_checks
    
#     if results["success"]:
#         print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
#     else:
#         print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
#         print(f"   Failed expectations: {failed_expectations}")
    
#     return results["success"], failed_expectations


import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset.
    Mirrors the original Great Expectations checks using plain pandas.
    """
    print("🔍 Starting data validation...")

    results = []  # List of (check_name, passed: bool)

    def check(name, condition):
        results.append((name, condition))

    # === SCHEMA VALIDATION ===
    print("   📋 Validating schema and required columns...")

    required_cols = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    for col in required_cols:
        check(f"expect_column_to_exist:{col}", col in df.columns)

    # customerID not null
    if "customerID" in df.columns:
        check("expect_column_values_to_not_be_null:customerID",
              df["customerID"].notnull().all())

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    set_checks = {
        "gender":           ["Male", "Female"],
        "Partner":          ["Yes", "No"],
        "Dependents":       ["Yes", "No"],
        "PhoneService":     ["Yes", "No"],
        "Contract":         ["Month-to-month", "One year", "Two year"],
        "InternetService":  ["DSL", "Fiber optic", "No"],
    }
    for col, valid_set in set_checks.items():
        if col in df.columns:
            check(f"expect_column_values_to_be_in_set:{col}",
                  df[col].isin(valid_set).all())

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    if "tenure" in df.columns:
        check("expect_column_values_to_be_between:tenure:0_120",
              df["tenure"].between(0, 120).all())

    if "MonthlyCharges" in df.columns:
        check("expect_column_values_to_be_between:MonthlyCharges:0_200",
              df["MonthlyCharges"].between(0, 200).all())

    if "TotalCharges" in df.columns:
        total_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
        check("expect_column_values_to_be_between:TotalCharges:>=0",
              (total_numeric.dropna() >= 0).all())
        # total_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # check("expect_column_values_to_be_between:TotalCharges:>=0",
        #       (total_numeric >= 0).all())

    # === STATISTICAL / NULL VALIDATION ===
    print("   📈 Validating statistical properties...")

    for col in ["tenure", "MonthlyCharges"]:
        if col in df.columns:
            check(f"expect_column_values_to_not_be_null:{col}",
                  df[col].notnull().all())

    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        total_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
        valid_rows = total_numeric.notnull()
        mostly_check = (
            (total_numeric[valid_rows] >= df["MonthlyCharges"][valid_rows])
            .mean() >= 0.95
        )
        check("expect_column_pair_A_gte_B:TotalCharges_MonthlyCharges:mostly_0.95",
              mostly_check)

    # === PROCESS RESULTS ===
    print("   ⚙️  Running complete validation suite...")

    failed_expectations = [name for name, passed in results if not passed]
    total_checks  = len(results)
    passed_checks = sum(1 for _, passed in results if passed)
    failed_checks = total_checks - passed_checks
    is_valid = len(failed_expectations) == 0

    if is_valid:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return is_valid, failed_expectations