import pandas as pd
import matplotlib.pyplot as plt

# Exercise 0: Read the dataset
def exercise_0(file):
    return pd.read_csv(file)

df = exercise_0('transactions.csv')


# Exercise 1: Return the column names as a list
def exercise_1(df):
    return df.columns.tolist()

print(exercise_1(df))


# Exercise 2: Return the first k rows
def exercise_2(df, k):
    return df.head(k)

print(exercise_2(df, 10))  # First 10 rows


# Exercise 3: Return a random sample of k rows
def exercise_3(df, k):
    return df.sample(k)

print(exercise_3(df, 10))  # Random 10 rows


# Exercise 4: Return unique transaction types
def exercise_4(df):
    return df['type'].unique().tolist()

print(exercise_4(df))


# Exercise 5: Top 10 transaction destinations with frequencies
def exercise_5(df):
    return df['nameDest'].value_counts().head(10)

print(exercise_5(df))


# Exercise 6: Return all rows where fraud was detected
def exercise_6(df):
    return df[df['isFraud'] == 1]

print(exercise_6(df).head())


# Exercise 7: Number of distinct destinations each source has interacted with
def exercise_7(df):
    return df.groupby('nameOrig')['nameDest'].nunique().sort_values(ascending=False).reset_index()

print(exercise_7(df).head())


def visual_1(df):
    def transaction_counts(df):
        return df['type'].value_counts()

    def transaction_counts_split_by_fraud(df):
        return df[df['isFraud'] == 1]['type'].value_counts()

    fig, axs = plt.subplots(2, figsize=(8, 12))

    transaction_counts(df).plot(ax=axs[0], kind='bar')
    axs[0].set_title('Transaction Types')
    axs[0].set_xlabel('Transaction Type')
    axs[0].set_ylabel('Count')

    transaction_counts_split_by_fraud(df).plot(ax=axs[1], kind='bar')
    axs[1].set_title('Fraudulent Transactions by Type')
    axs[1].set_xlabel('Transaction Type')
    axs[1].set_ylabel('Count')

    fig.suptitle('Transaction Analysis')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    return 'Bar chart showing the distribution of transaction types and the breakdown of fraudulent transactions.'


visual_1(df)


def visual_2(df):
    def query(df):
        return df[df['type'] == 'CASH_OUT'][['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]

    plot_data = query(df)
    plot = plot_data.plot.scatter(x='oldbalanceOrg', y='newbalanceOrig')
    plot.set_title('Account Balance Delta for Cash-Out Transactions')
    plot.set_xlabel('Origin Account Balance Delta')
    plot.set_ylabel('Destination Account Balance Delta')

    plt.show()

    return 'Scatter plot showing the relationship between origin and destination account balances for Cash-Out transactions.'


visual_2(df)


# To analyze which transaction types are most associated with flagged fraud.
def exercise_custom(df):
    return df[df['isFlaggedFraud'] == 1]['type'].value_counts()


def visual_custom(df):
    flagged_fraud_counts = exercise_custom(df)
    flagged_fraud_counts.plot(kind='bar')
    plt.title('Flagged Fraud by Transaction Type')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.show()

    return 'Bar chart showing the distribution of transaction types for flagged fraudulent transactions.'


visual_custom(df)
