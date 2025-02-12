import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define data loader class
class visualizer:
    def __init__(self, data_path):
        # Initialize the Folder path of the data
        self.df = data_path

    def labeled_barplot(self, feature, perc=False, n=None):
        """
        Barplot with percentage at the top

        data: dataframe
        feature: dataframe column
        perc: whether to display percentages instead of count (default is False)
        n: displays the top n category levels (default is None, i.e., display all levels)
        """
        
        total = len(self.df[feature])  # length of the column
        count = self.df[feature].nunique()
        if n is None:
            plt.figure(figsize=(count + 1, 5))
        else:
            plt.figure(figsize=(n + 1, 5))

        plt.xticks(rotation=90, fontsize=15)

        ordered_categories = self.df[feature].value_counts().index[:n]
        
        ax = sns.countplot(
            data=self.df,
            x=feature,
            palette="Paired",
            order=ordered_categories,
        )

        for p in ax.patches:
            if perc == True:
                label = "{:.1f}%".format(
                    100 * p.get_height() / total
                )  # percentage of each class of the category
            else:
                label = p.get_height()  # count of each level of the category

            x = p.get_x() + p.get_width() / 2  # width of the plot
            y = p.get_height()  # height of the plot

            ax.annotate(
                label,
                (x, y),
                ha="center",
                va="center",
                size=12,
                xytext=(0, 5),
                textcoords="offset points",
            )  # annotate the percentage

        plt.show()  # show the plot


    def histogram_boxplot(self, feature, figsize=(12, 7), kde=False, bins=None):
        """
        Boxplot and histogram combined

        data: dataframe
        feature: dataframe column
        figsize: size of figure (default (12,7))
        kde: whether to the show density curve (default False)
        bins: number of bins for histogram (default None)
        """
        f2, (ax_box2, ax_hist2) = plt.subplots(
            nrows=2,  # Number of rows of the subplot grid= 2
            sharex=True,  # x-axis will be shared among all subplots
            gridspec_kw={"height_ratios": (0.25, 0.75)},
            figsize=figsize,
        )  # creating the 2 subplots
        sns.boxplot(
            data=self.df, x=feature, ax=ax_box2, showmeans=True, color="violet"
        )  # boxplot will be created and a triangle will indicate the mean value of the column
        sns.histplot(
            data=self.df, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
        ) if bins else sns.histplot(
            data=self.df, x=feature, kde=kde, ax=ax_hist2
        )  # For histogram
        ax_hist2.axvline(
            self.df[feature].mean(), color="green", linestyle="--"
        )  # Add mean to the histogram
        ax_hist2.axvline(
            self.df[feature].median(), color="black", linestyle="-"
        )  # Add median to the histogram

    def plot_categorical_distribution(self, x_col, hue_col, title, x_label, y_label):
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.df, x=x_col, hue=hue_col, palette='tab10')
        plt.xticks(rotation=45)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(title=hue_col, bbox_to_anchor=(1, 1))
        plt.show()

    def plot_numerical_distribution(self, group_col, value_col, title, x_label, y_label):
        grouped_data = self.df.groupby(group_col)[value_col].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=grouped_data, x=group_col, y=value_col, palette='coolwarm')
        plt.xticks(rotation=45)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def plot_outlier_detection(self, numeric_cols):
        plt.figure(figsize=(12, 6))
        self.df[numeric_cols].boxplot(rot=45)
        plt.title("Outlier Detection using Box Plots")
        plt.xlabel("Numerical Features")
        plt.ylabel("Value Range")
        plt.show()

    def plot_correlation_heatmap(self, numeric_cols):
        plt.figure(figsize=(10, 6))
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_premium_vs_claims(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='TotalPremium', y='TotalClaims', hue='CoverType', palette='viridis', alpha=0.7)
        plt.title("Total Premium vs. Total Claims by Cover Type")
        plt.xlabel("Total Premium")
        plt.ylabel("Total Claims")
        plt.legend(title="Cover Type", bbox_to_anchor=(1, 1))
        plt.show()

    def plot_premium_trend(self):
        # Ensure TransactionMonth is a datetime type
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
         # Ensure TotalPremium is numeric and handle errors
        self.df['TotalPremium'] = pd.to_numeric(self.df['TotalPremium'], errors='coerce')

        # Drop rows where necessary values are missing
        self.df = self.df.dropna(subset=['TransactionMonth', 'TotalPremium'])

        # Aggregate by month
        monthly_trend = self.df.groupby(self.df['TransactionMonth'].dt.to_period("M")).agg({'TotalPremium': 'sum'}).reset_index()
         # Convert period to string for plotting
        monthly_trend['TransactionMonth'] = monthly_trend['TransactionMonth'].astype(str)

        # Plot the trend
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_trend, x='TransactionMonth', y='TotalPremium', marker="o", color='b')
        plt.title("Monthly Trend of Total Premium")
        plt.xlabel("Transaction Month")
        plt.ylabel("Total Premium")
        plt.xticks(rotation=45)
        plt.show()
