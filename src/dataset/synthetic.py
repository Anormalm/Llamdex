from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import random
import torch

import json

def make_classification_categorical(n_samples=1000, n_features=20, n_informative=2, n_categorical=2, n_categories=2,
                                    dummy=False, shuffle=True, random_state=None, **kwargs):
    """
    Generate classification dataset with categorical features. The dataset is generated using make_classification from
    sklearn.datasets. The categorical features are generated from informative features. The informative features in
    float are converted to categorical features by binning.
    Args:
        n_samples: The number of samples to generate.
        n_features: The total number of features.
        n_informative: The number of informative features. (passed to make_classification)
        n_categorical: The number of categorical features. Must be less than or equal to n_informative.
        n_categories: The number of categories per categorical feature.
        dummy: If True, the categorical features are one-hot encoded. If False, the categorical features are ordinal.
        shuffle: If True, shuffle the instances of the dataset.
        random_state: Random state for shuffling. Also passed to make_classification.
        **kwargs: Other arguments passed to make_classification.

    Returns: X, y: the dataset and the labels.
    """
    if n_categorical > n_informative:
        raise ValueError("n_categorical must be less than or equal to n_informative.")

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               random_state=random_state, **kwargs)

    # Convert informative features to categorical
    for i in range(n_categorical):
        X[:, i] = pd.cut(X[:, i], bins=n_categories, labels=False).astype(int)

    if dummy:
        df = pd.DataFrame(X)
        X = pd.get_dummies(df, columns=range(n_categorical)).values

    if shuffle:
        np.random.seed(random_state)    # seed for shuffling

        # shuffle instances
        idx = np.random.permutation(n_samples)
        X, y = X[idx], y[idx]

    return X, y


def make_classification_from_json(n_samples: int, dataset_json: str | dict, n_noise_features: int = 0, n_classes: int = 2,
                                  class_sep: float = 7.0, random_state: int = 0, **kwargs) -> pd.DataFrame:
    """
    Generate a classification dataset from a JSON file containing column information.
    Args:
        n_samples: The number of samples to generate.
        dataset_json: The JSON file containing column information.
          Format:
            {X: {column_name: {type: int, range: [min, max]},
            column_name: {type: float, range: [min, max]},
            column_name: {type: bool},
            column_name: {type: category, categories: [cat1, cat2, ...]}},
            y: {name: str, type: category, categories: [cat1, cat2, ...]}
        label_column: The name of the label column.
        n_noise_features: The number of noise features to add to the dataset.
        n_classes: The number of classes in the dataset.
        class_sep: The separation between the classes in generation.
        random_state: Random state for generating the dataset.
        **kwargs: Other arguments passed to make_classification.

    Returns: A pandas DataFrame containing the generated dataset.
    """
    if isinstance(dataset_json, str):
        with open(dataset_json, 'r') as f:
            dataset_json = json.load(f)

    def scale_to_range(col: pd.DataFrame, min_val, max_val):
        # use MinMaxScaler to scale the array to the range [min_val, max_val]
        scaler = MinMaxScaler(feature_range=(min_val, max_val))
        return scaler.fit_transform(col.values.reshape(-1, 1)).flatten()

    n_features = len(dataset_json['X'])
    X, y = make_classification(n_samples=n_samples, n_features=n_features + n_noise_features,
                               n_informative=n_features, n_redundant=0,
                               n_classes=n_classes, class_sep=class_sep, random_state=random_state, **kwargs)
    # move the label column to the last column
    df = pd.DataFrame(X, columns=list(dataset_json['X'].keys()))

    # sort the columns by name
    df = df.reindex(sorted(df.columns), axis=1)

    def process_col(col, specs):
        print(col)
        if specs['type'] == 'int':
            processed_column = scale_to_range(col, specs['range'][0] - 0.49, specs['range'][1] + 0.49).round().astype(int)
        elif specs['type'] == 'float':
            processed_column = scale_to_range(col, specs['range'][0], specs['range'][1])
        elif specs['type'] == 'bool':
            mean_value = col.mean()
            processed_column = (col > mean_value).astype('bool')
        elif specs['type'] == 'category':
            num_categories = len(specs['categories'])
            processed_column = pd.cut(col, bins=num_categories, labels=specs['categories']).astype('category')
        else:
            raise ValueError(f"Unknown column type: {specs['type']}")
        return processed_column

    for column_name, specs in dataset_json['X'].items():
        col = df[column_name]
        df[column_name] = process_col(col, specs)

    df[dataset_json['y']['name']] = process_col(y, dataset_json['y'])

    return df

def adult_interpret_to_human_readable(df: pd.DataFrame):
    """
    Change the values to be more human-readable
    Args:
        df: The input DataFrame.

    Returns: The DataFrame with more human-readable values.

    """
    df = df.rename(columns={
        "fnlwgt": "number_of_people_in_this_group",
        "education-num": "years_of_education",
        "education": "highest_education",
        "martial-status": "martial-status",
        "hours-per-week": "working_hours_per_week",
    })

    df['highest_education'] = df['highest_education'].replace({
        "Assoc-acdm": "Associate degree - academic program",
        "Assoc-voc": "Associate degree - vocational program",
        "Bachelors": "Bachelor's degree",
        "Doctorate": "Doctorate",
        "HS-grad": "High school graduate",
        "Masters": "Master's degree",
        "Preschool": "Preschool",
        "Prof-school": "Professional school",
        "Some-college": "Attended college but did not graduate",
    })

    df['marital-status'] = df['marital-status'].replace({
        "Divorced": "Divorced",
        "Married-AF-spouse": "Married to a spouse in the Armed Forces",
        "Married-civ-spouse": "Married to a civilian spouse",
        "Married-spouse-absent": "Married but spouse is not around",
        "Never-married": "Never married",
        "Separated": "Separated",
        "Widowed": "Widowed",
    })

    df['native-country'] = df['native-country'].replace({
        "Cambodia": "Cambodia",
        "Canada": "Canada",
        "China": "China",
        "Columbia": "Columbia",
        "Cuba": "Cuba",
        "Dominican-Republic": "Dominican Republic",
        "Ecuador": "Ecuador",
        "El-Salvador": "El Salvador",
        "England": "England",
        "France": "France",
        "Germany": "Germany",
        "Greece": "Greece",
        "Guatemala": "Guatemala",
        "Haiti": "Haiti",
        "Holand-Netherlands": "Netherlands",
        "Honduras": "Honduras",
        "Hong": "Hong Kong",
        "Hungary": "Hungary",
        "India": "India",
        "Iran": "Iran",
        "Ireland": "Ireland",
        "Italy": "Italy",
        "Jamaica": "Jamaica",
        "Japan": "Japan",
        "Laos": "Laos",
        "Mexico": "Mexico",
        "Nicaragua": "Nicaragua",
        "Outlying-US(Guam-USVI-etc)": "Outlying US territories",
        "Peru": "Peru",
        "Philippines": "Philippines",
        "Poland": "Poland",
        "Portugal": "Portugal",
        "Puerto-Rico": "Puerto Rico",
        "Scotland": "Scotland",
        "South": "South Korea",
        "Taiwan": "Taiwan",
        "Thailand": "Thailand",
        "Trinadad&Tobago": "Trinidad and Tobago",
        "United-States": "United States",
        "Vietnam": "Vietnam",
        "Yugoslavia": "Yugoslavia ",
    })

    df['occupation'] = df['occupation'].replace({
        "Adm-clerical": "Administrative clerical",
        "Armed-Forces": "Armed Forces",
        "Craft-repair": "Craft repair",
        "Exec-managerial": "Executive managerial",
        "Farming-fishing": "Farming fishing",
        "Handlers-cleaners": "Handlers cleaners",
        "Machine-op-inspct": "Machine operation inspection",
        "Other-service": "Other service",
        "Priv-house-serv": "Private house service",
        "Prof-specialty": "Professional specialty",
        "Protective-serv": "Protective service",
        "Sales": "Sales",
        "Tech-support": "Tech support",
        "Transport-moving": "Transport moving",
    })

    df['race'] = df['race'].replace({
        "Amer-Indian-Eskimo": "American Indian or Alaska Native",
        "Asian-Pac-Islander": "Asian or Pacific Islander",
        "Black": "Black",
        "Other": "Other",
        "White": "White", 
    })

    df['workclass'] = df['workclass'].replace({
        "Federal-gov": "Federal government",
        "Local-gov": "Local government",
        "Never-worked": "Never worked",
        "Private": "Private sector",
        "Self-emp-inc": "Self-employed (incorporated)",
        "Self-emp-not-inc": "Self-employed (not incorporated)",
        "State-gov": "State government",
        "Without-pay": "Without pay",
    })

    return df

def titanic_interpret_to_human_readable(df: pd.DataFrame):
    """
    Change the values to be more human-readable for the Stanford Titanic dataset
    Args:
        df: The input DataFrame.

    Returns: The DataFrame with more human-readable values.
    """
    df = df.rename(columns={
        "Siblings/Spouses Aboard": "num_siblings_and_spouses_aboard",
        "Parents/Children Aboard": "num_parents_and_children_aboard",
        "Pclass": "ticket_class",
    })

    df['ticket_class'] = df['ticket_class'].replace({
        1: "First class",
        2: "Second class",
        3: "Third class",
    })

    return df

def wine_quality_interpret_to_human_readable(df: pd.DataFrame):
    df = df.rename(columns={
        "fixed acidity": "non volatile acids"
    })

    return df

def bank_marketing_interpret_to_human_readable(df: pd.DataFrame):
    """
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services")
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric)
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   # related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular")
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   # other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")"""
    df = df.rename(columns={
        "job": "type_of_job",
        "marital": "marital_status",
        "education": "education",
        "default": "has_credit_in_default",
        "balance": "average_yearly_balance_in_euros",
        "housing": "has_housing_loan",
        "loan": "has_personal_loan",
        "contact": "contact_communication_type",
        "day": "last_contact_day_of_month",
        "month": "last_contact_month",
        "duration": "last_contact_duration",
        "campaign": "num_contacts_performed",
        "pdays": "days_since_last_contact",
        "previous": "num_contacts_before",
        "poutcome": "outcome_of_previous_campaign",
        "y": "whether_subscribed_term_deposit"
    })

    return df


def abalone_interpret_to_human_readable(df: pd.DataFrame):
    """
    Change the values to be more human-readable for the Abalone dataset
    Args:
        df: The input DataFrame.

    Returns: The DataFrame with more human-readable values.
    """
    df = df.rename(columns={
        "Sex": "gender",
        "Length": "shell_length",
        "Diameter": "shell_diameter",
        "Height": "shell_height",
        "Whole weight": "whole_weight",
        "Shucked weight": "shucked_weight",
        "Viscera weight": "viscera_weight",
        "Shell weight": "shell_weight",
    })

    df['gender'] = df['gender'].replace({
        'M': "Male",
        'F': "Female",
        'I': "Infant"
    })

    # Convert measurements to more readable format
    for col in ['shell_length', 'shell_diameter', 'shell_height']:
        df[col] = df[col].apply(lambda x: f"{x:.4f} mm")

    for col in ['whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']:
        df[col] = df[col].apply(lambda x: f"{x:.4f} grams")

    if 'Age_Group' in df.columns:
        df['Age_Group'] = df['Age_Group'].replace({
            'A': "Young",
            'B': "Adult",
            'C': "Old",
        })

    return df

def nursery_interpret_to_human_readable(df: pd.DataFrame):
    """
    Change the values to be more human-readable for the Nursery dataset
    Args:
        df: The input DataFrame.

    The hierarchical model ranks nursery-school applications according
   to the following concept structure:

   NURSERY            Evaluation of applications for nursery schools
   . EMPLOY           Employment of parents and child's nursery
   . . parents        Parents' occupation
   . . has_nurs       Child's nursery
   . STRUCT_FINAN     Family structure and financial standings
   . . STRUCTURE      Family structure
   . . . form         Form of the family
   . . . children     Number of children
   . . housing        Housing conditions
   . . finance        Financial standing of the family
   . SOC_HEALTH       Social and health picture of the family
   . . social         Social conditions
   . . health         Health conditions

   Attribute Values:
   parents        usual, pretentious, great_pret
   has_nurs       proper, less_proper, improper, critical, very_crit
   form           complete, completed, incomplete, foster
   children       1, 2, 3, more
   housing        convenient, less_conv, critical
   finance        convenient, inconv
   social         non-prob, slightly_prob, problematic
   health         recommended, priority, not_recom

   Class Distribution (number of instances per class)

   class        N         N[%]
   ------------------------------
   not_recom    4320   (33.333 %)
   recommend       2   ( 0.015 %)
   very_recom    328   ( 2.531 %)
   priority     4266   (32.917 %)
   spec_prior   4044   (31.204 %)

    Returns: The DataFrame with more human-readable values.
    """
    df = df.rename(columns={
        "parents": "parents_occupation",
        "has_nurs": "child_nursery",
        "form": "family_form",
        "children": "num_children",
        "housing": "housing_conditions",
        "finance": "financial_standing",
        "social": "social_conditions",
        "health": "health_conditions",
        "evaluation": "evaluation"
    })

    df['parents_occupation'] = df['parents_occupation'].replace({
        'usual': "Usual",
        'pretentious': "Pretentious",
        'great_pret': "Great pretentious"
    })

    df['child_nursery'] = df['child_nursery'].replace({
        'proper': "Proper",
        'less_proper': "Less proper",
        'improper': "Improper",
        'critical': "Critical",
        'very_crit': "Very critical"
    })

    df['family_form'] = df['family_form'].replace({
        'complete': "Complete",
        'completed': "Completed",
        'incomplete': "Incomplete",
        'foster': "Foster"
    })

    df['num_children'] = df['num_children'].replace({
        1: "1 child",
        2: "2 children",
        3: "3 children",
        4: "4 or more children"
    })

    df['housing_conditions'] = df['housing_conditions'].replace({
        'convenient': "Convenient",
        'less_conv': "Less convenient",
        'critical': "Critical"
    })

    df['financial_standing'] = df['financial_standing'].replace({
        'convenient': "Convenient",
        'inconv': "Inconvenient"
    })

    df['social_conditions'] = df['social_conditions'].replace({
        'non-prob': "Non-problematic",
        'slightly_prob': "Slightly problematic",
        'problematic': "Problematic"
    })

    df['health_conditions'] = df['health_conditions'].replace({
        'recommended': "Recommended",
        'priority': "Priority",
        'not_recom': "Not recommended"
    })

    return df

def check_contradictions_with_llm(batch, tokenizer, model, answer_key, max_new_tokens=5):
    contradiction_query = """Analyze the following information and determine if there are any contradictions or inconsistencies. 
        If you find any contradictions, respond with 'REJECT' in one word. If there are no contradictions, respond with 'VALID' in one word. 
        Here's the information:\n"""
    
    queries = []
    batched_tokens = []
    batched_attention_mask = []

    for row in batch:
        items = list(row.items())
        query_str = contradiction_query + " ".join(f'#{key}: {value}' for key, value in items if (key != answer_key and not pd.isnull(value)))
        messages = [{"role": "user", "content": query_str}]
        tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        attention_mask = (tokens != tokenizer.pad_token_id).long()

        queries.append(query_str)
        batched_tokens.append(tokens)
        batched_attention_mask.append(attention_mask)

    max_length = max(t.size(-1) for t in batched_tokens)

    padded_tokens = torch.full((len(batched_tokens), max_length), tokenizer.pad_token_id, dtype=torch.long).to("cuda")
    padded_attention_mask = torch.zeros((len(batched_tokens), max_length), dtype=torch.long).to("cuda")

    for i, (tokens, attention_mask) in enumerate(zip(batched_tokens, batched_attention_mask)):
        seq_length = tokens.size(-1)
        padded_tokens[i, -seq_length:] = tokens
        padded_attention_mask[i, -seq_length:] = attention_mask
    
    with torch.no_grad():
        outputs = model.generate(input_ids=padded_tokens, attention_mask=padded_attention_mask,
                                max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    
    results = tokenizer.batch_decode(outputs[:, padded_tokens.shape[-1]:], skip_special_tokens=True)

    '''
    for result in results:
        if "REJECT" in result.upper():
            print("Contradiction found:")
            print(" ".join(f'#{key}: {value}' for key, value in items if (key != answer_key and not pd.isnull(value))))
            print(result)
        else:
            print("No contradiction found.")
            print(" ".join(f'#{key}: {value}' for key, value in items if (key != answer_key and not pd.isnull(value))))
            print(result)
    '''

    return ["REJECT" in result.upper() for result in results]

if __name__ == '__main__':
    import xgboost

    X_continuous, y_continuous = make_classification(n_samples=10000, n_features=10, n_informative=4, shuffle=True,
                                                    random_state=0, class_sep=4)
    print(X_continuous.shape, y_continuous.shape)

    X_train_continuous, y_train_continuous = X_continuous[:8000], y_continuous[:8000]
    X_test_continuous, y_test_continuous = X_continuous[8000:], y_continuous[8000:]

    model_continuous = xgboost.XGBClassifier(n_estimators=50, max_depth=4, objective='binary:logistic', device='cuda:1')
    model_continuous.fit(X_train_continuous, y_train_continuous)

    train_acc_continuous = model_continuous.score(X_train_continuous, y_train_continuous)
    test_acc_continuous = model_continuous.score(X_test_continuous, y_test_continuous)
    print(f"Train accuracy (continuous): {train_acc_continuous:.4f}, Test accuracy (continuous): {test_acc_continuous:.4f}")

    X, y = make_classification_categorical(n_samples=10000, n_features=10, n_informative=4, n_categorical=2,
                                           n_categories=10, dummy=False, shuffle=True, random_state=0, class_sep=4)
    print(X.shape, y.shape)

    X_train, y_train = X[:8000], y[:8000]
    X_test, y_test = X[8000:], y[8000:]

    model = xgboost.XGBClassifier(n_estimators=50, max_depth=4, objective='binary:logistic', device='cuda:1')
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy (categorical): {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    X_dummy, y_dummy = make_classification_categorical(n_samples=10000, n_features=10, n_informative=4, n_categorical=2,
                                                        n_categories=10, dummy=True, shuffle=True, random_state=0,
                                                        class_sep=4)
    print(X_dummy.shape, y_dummy.shape)

    X_train_dummy, y_train_dummy = X_dummy[:8000], y_dummy[:8000]
    X_test_dummy, y_test_dummy = X_dummy[8000:], y_dummy[8000:]

    model_dummy = xgboost.XGBClassifier(n_estimators=50, max_depth=4, objective='binary:logistic', device='cuda:1')
    model_dummy.fit(X_train_dummy, y_train_dummy)

    train_acc_dummy = model_dummy.score(X_train_dummy, y_train_dummy)
    test_acc_dummy = model_dummy.score(X_test_dummy, y_test_dummy)
    print(f"Train accuracy (dummy): {train_acc_dummy:.4f}, Test accuracy (dummy): {test_acc_dummy:.4f}")


def interpret_to_human_readable(df: pd.DataFrame, dataset: str):
    if dataset == 'adult':
        return adult_interpret_to_human_readable(df)
    elif dataset == 'titanic':
        return titanic_interpret_to_human_readable(df)
    elif dataset == 'wine_quality':
        return wine_quality_interpret_to_human_readable(df)
    elif dataset == 'bank_marketing':
        return bank_marketing_interpret_to_human_readable(df)
    elif dataset == "abalone":
        return abalone_interpret_to_human_readable(df)
    elif dataset == "nursery":
        return nursery_interpret_to_human_readable(df)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")





