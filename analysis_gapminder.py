import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
from pyclustering.cluster import xmeans

current_palette = sns.color_palette("colorblind", 7)
sns.set(style='whitegrid')
if True:
    # "#0072b2", "#f0e442", "#009e73", "#d55e00", "#cc79a7"
    current_palette[6] = (0 / 255, 114 / 255, 178 / 255)
    current_palette[0] = (85 / 255, 85 / 255, 85 / 255)
    current_palette[1] = (240 / 255, 228 / 255, 66 / 255)
    current_palette[2] = (0 / 255, 158 / 255, 115 / 255)
    current_palette[3] = (213 / 255, 94 / 255, 0 / 255)
    current_palette[4] = (204 / 255, 121 / 255, 167 / 255)
    # current_palette[5] = (93 / 255, 58 / 255, 155 / 255)
    current_palette[5] = (75 / 255, 0 / 255, 146 / 255)

# current_palette

data_dir = "data/gapminder2023"
res_dir = "figs/gapminder2023"

# Transform each df to have columns: Country, Year, values
df_e0 = pd.read_csv(os.path.join(data_dir,'life_expectancy_years.csv'))
df_e0 = df_e0.melt(id_vars=["country"], var_name="Year", value_name="e0")
df_tfr = pd.read_csv(os.path.join(data_dir,'children_per_woman_total_fertility.csv'))
df_tfr = df_tfr.melt(id_vars=["country"], var_name="Year", value_name="tfr")
df_pop = pd.read_csv(os.path.join(data_dir,'population_total.csv'))
df_pop = df_pop.melt(id_vars=["country"], var_name="Year", value_name="pop")
df_pop_growth = pd.read_csv(os.path.join(data_dir,'population_growth_annual_percent.csv'))
df_pop_growth = df_pop_growth.melt(id_vars=["country"], var_name="Year", value_name="growth")
# df_pop_growth["growth"] = df_pop_growth["growth"].str.replace("−", "-").astype(float)
df_pop_child = pd.read_csv(os.path.join(data_dir,'population_aged_0_14_years_total_number.csv'))
df_pop_child = df_pop_child.melt(id_vars=["country"], var_name="Year", value_name="pop_child")
df_literacy = pd.read_csv(os.path.join(data_dir,'literacy_rate_adult.csv'))
df_literacy = df_literacy.melt(id_vars=["country"], var_name="Year", value_name="literacy")
df_birth = pd.read_csv(os.path.join(data_dir,'crude_birth_rate_births_per_1000_population.csv'))
df_birth = df_birth.melt(id_vars=["country"], var_name="Year", value_name="birth")

# Merge all dfs
df = df_e0.merge(df_tfr, on=["country", "Year"], how="left")
df = df.merge(df_pop, on=["country", "Year"], how="left")
df = df.merge(df_pop_growth, on=["country", "Year"], how="left")
df = df.merge(df_pop_child, on=["country", "Year"], how="left")
df = df.merge(df_literacy, on=["country", "Year"], how="left")
df = df.merge(df_birth, on=["country", "Year"], how="left")
df.head()

# Change datatype of values except for country name to float.
df[["Year", "e0", "tfr",  "literacy", "birth"]] = df[["Year", "e0", "tfr",  "literacy","birth"]].astype(float)
# df[["growth"]] = df[["growth"]].astype(float)

# Use the data for years before 2023.
df = df[df["Year"] < 2023]

# Change "B", "M" and "k" to 1e9, 1e6 and 1e3, respectively.
df["pop"] = df["pop"].str.replace("B", "*1e9").str.replace("M", "*1e6").str.replace("k", "*1e3").apply(pd.eval)
# By noting that, some values are nan for "pop_child", we changed "B", "M" and "k" to 1e9, 1e6 and 1e3, respectively.
df["pop_child"] = df["pop_child"].str.replace("B", "*1e9").str.replace("M", "*1e6").str.replace("k", "*1e3").replace(np.nan, 0).apply(pd.eval)
df["pop_child"] = df["pop_child"].replace(0.0, np.nan)

# # Change string "-" to float "-" in the column "pop_growth".
df["growth"] = df["growth"].str.replace("−", "-").astype(float)

# Calculate working population
df["pop_working"] = df["pop"] - df["pop_child"]

df["pop_growth"] = df.groupby("country")["pop"].pct_change()
df["pop_working_growth"] = df.groupby("country")["pop_working"].pct_change()

# Taking 5-year rolling average for growth rates.
df["pop_growth_rolling"] = df.groupby("country")["pop_growth"].rolling(5, center=True).mean().reset_index(drop=True)
df["pop_working_growth_rolling"] = df.groupby("country")["pop_working_growth"].rolling(5, center=True).mean().reset_index(drop=True)

if True:
    # Draw the scatterplot of "e0" and "birth" for all countries.
    plt.figure()
    # ax = sns.scatterplot(data=df, x="e0", y="birth", color=current_palette[0], s= 30, alpha = 0.1)
    ax = sns.scatterplot(data=df, x="e0", y="birth", color=current_palette[0], s=20, marker="$\circ$", ec="face", alpha = .7, lw = .05)
    ax.set_xlabel("life expectancy", fontsize=20)
    ax.set_ylabel("crude birth rate", fontsize=20)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir,f'life_birth_scatter_all.pdf'), bbox_inches='tight')
    plt.close('all')

if True:
    # Draw the pathways of "e0" and "birth" for all countries with 90% confidence interval.
    # X-axis is life expectancy and y-axis is birth. Colors indicate the year is before or after 1920.
    # Two master curves are drawn theoretically, "e0" * "birth" = 200 and "e0" * "birth" * exp("e0" / 15) = 16500.

    # Calculate r^2 scores of the master curves.
    year_threshold = 1930
    df_cur = df[(df["Year"] < year_threshold) & (df["e0"] > 30)]
    df_cur = df_cur[["e0", "birth"]].dropna()
    x = df_cur["e0"]
    y = df_cur["birth"]
    model = sm.OLS(y, 1 / x)
    results = model.fit()
    print(results.summary())
    print(results.rsquared)
    constant1 = results.params[0]
    # 1416

    df_cur = df[(df["Year"] >= year_threshold) & (df["e0"] > 60)]
    df_cur = df_cur[["e0", "birth"]].dropna()
    x = df_cur["e0"]
    y = df_cur["birth"]
    # for const in range(15, 30, 1):
    #     model = sm.OLS(y, 1 / (x * np.exp(x / 25)))
    #     results = model.fit()
    #     print(results.rsquared)
    #     const = 25 maximize r^2
    model = sm.OLS(y, 1 / (x * np.exp(x / 25)))
    results = model.fit()
    print(results.summary())
    print(results.rsquared)
    constant2 = results.params[0]
    # 26137

    df_cur = df[(df["Year"] >= year_threshold) & (df["e0"] < 60)]
    df_cur = df_cur[["e0", "birth"]].dropna()
    x = [1] * len(df_cur["e0"])
    y = df_cur["birth"]
    model = sm.OLS(y, x)
    results = model.fit()
    # print(results.summary())
    constant3 = results.params[0]
    # 43


    x1 = np.linspace(30, 85, 100)
    y1 = constant1 / x1
    master_curve1 = np.c_[x1[y1 < constant3], y1[y1 < constant3]]

    x2 = np.linspace(60, 85, 100)
    y2 = constant2 / (x2 * np.exp(x2 / 25))
    master_curve2 = np.c_[x2[y2 < constant3], y2[y2 < constant3]]

    x3 = np.linspace(20, 60, 100)
    y3 = np.ones(len(x2)) * constant3
    master_curve3 = np.c_[x3, y3]

    x1 = np.linspace(25, 70, 100)
    y1 = constant1 / x1
    x2 = np.linspace(55, 90, 100)
    y2 = constant2 / (x2 * np.exp(x2 / 25))

    df_cur = df[df["Year"] < year_threshold]
    plt.figure()
    # ax = sns.lineplot(data=df_cur, x="e0", y="birth", color=current_palette[0], errorbar=('ci', 90))
    ax = sns.lineplot(data=df_cur, x="e0", y="birth", color=current_palette[0], errorbar='sd')
    df_cur = df[df["Year"] >= year_threshold]
    # ax = sns.lineplot(data=df_cur, x="e0", y="birth", color=current_palette[1], errorbar=('ci', 90))
    ax = sns.lineplot(data=df_cur, x="e0", y="birth", color=current_palette[1], errorbar='sd')
    ax.plot(x1, y1, color=current_palette[0], linestyle='--')
    ax.plot(x2, y2, color=current_palette[1], linestyle='--')
    ax.set_xlabel("life expectancy", fontsize=20)
    ax.set_ylabel("crude birth rate", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_pathway_isoclines_{year_threshold}.pdf'), bbox_inches='tight')
    plt.close('all')


    # For each data point, the relative distances from ("e0", "birth") to the master curves are calculated.
    # The distance is calculated as the minimum distance from the master curves.

    for ind in df.index:
        df.loc[ind, "dist1"] = np.min(np.sum((np.array(df.loc[ind, ["e0", "birth"]]) - master_curve1) ** 2, axis=1))
        df.loc[ind, "dist2"] = np.min(np.sum((np.array(df.loc[ind, ["e0", "birth"]]) - master_curve2) ** 2, axis=1))
        df.loc[ind, "dist3"] = np.min(np.sum((np.array(df.loc[ind, ["e0", "birth"]]) - master_curve3) ** 2, axis=1))
    df["relative dist"] = np.log(df["dist1"] / df["dist2"])
    df["class"] = 1 * (df["dist1"] < df["dist2"]) * (df["dist1"] < df["dist3"]) + 2 * (df["dist2"] < df["dist1"]) * (df["dist2"] < df["dist3"]) + 3 * (df["dist3"] < df["dist1"]) * (df["dist3"] < df["dist2"])
    df[["country", "Year", "e0", "birth", "class"]].head(20)

    # For each country, I identified the first year and durations when the class is 1 or 2.
    # The first year is the first year when the class is 1 or 2.
    # The duration is the number of consecutive years when the class is 1 or 2.
    df_cur = df[(df["e0"] > 40)]
    df_country_class1 = df_cur[df_cur["class"] == 1].groupby("country").agg({"Year": ["min", "count"]})
    df_country_class2 = df_cur[df_cur["class"] == 2].groupby("country").agg({"Year": ["min", "count"]})
    df_country_class = df_country_class1.merge(df_country_class2, left_index=True, right_index=True, suffixes=("_1", "_2"), how = "outer")
    df_country_class.columns = ["first year 1", "duration 1", "first year 2", "duration 2"]
    df_country_class["class"] = 1 * (df_country_class["duration 1"] > df_country_class["duration 2"]) + 1 * (1 - df_country_class["duration 1"].isna()) * (df_country_class["duration 2"].isna()) + 2 * (df_country_class["duration 1"] < df_country_class["duration 2"]) + 2 * (df_country_class["duration 1"].isna()) * (1 - df_country_class["duration 2"].isna())
    df_country_class.head()
    df_country_class.value_counts("class")
    df_country_class.to_csv("country_class.csv")

    res_1, res_2 = [], []
    for country in df_country_class.index:
        if df_country_class.loc[country, "class"] == 1:
            res_1.append((country, df_country_class.loc[country, "first year 1"]))
        elif df_country_class.loc[country, "class"] == 2:
            res_2.append((country, df_country_class.loc[country, "first year 2"]))
    print(res_1)
    print(res_2)

    # Histogram of the first year when the class is 1 or 2.
    # X-axis is the first year and y-axis is the number of countries.
    # Color indicates the class of countries.
    plt.figure()
    ax = sns.histplot(data=df_country_class[df_country_class["class"] == 1], x="first year 1", color = current_palette[0], bins=30, binrange= (1800, 2020), alpha=0.5)
    sns.histplot(data=df_country_class[df_country_class["class"] == 2], x="first year 2", color = current_palette[1], bins=30, binrange= (1800, 2020), alpha=0.5)
    ax.set_xlabel("first year of transition", fontsize=20)
    ax.set_ylabel("number of countries", fontsize=20)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir,f'life_birth_transition_first_year.pdf'), bbox_inches='tight')
    plt.close('all')

    # Plot the relative distances of ("e0", "birth") from the master curves.
    # X-axis is year and y-axis is the relative distance. Colors indicate the country.
    plt.figure()
    ax = sns.lineplot(data=df_cur, x="Year", y="relative dist", hue="country", palette="tab20")
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel("relative distance", fontsize=20)
    ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir,f'life_birth_pathway_relative_dist.pdf'), bbox_inches='tight')
    plt.close('all')

if True:
    # Draw the pathway of "e0" * "birth" for each country.
    # X-axis is "year" and y-axis is the product of "e0" and "birth". Colors indicate the country.
    df_cur = df[["country", "Year", "e0", "birth", "class"]].dropna()
    df_cur["e0*birth"] = df_cur["e0"] * df_cur["birth"]
    df_cur["e0*birth_work"] = df_cur["e0"] * df_cur["birth"] * np.exp(df_cur["e0"] / 25)
    plt.figure()
    ax = sns.lineplot(data=df_cur, x="Year", y="e0*birth", hue="country", palette="tab20")
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel(r"$\lambda l$", fontsize=20)
    ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir,f'life_birth_year_lambdal.pdf'), bbox_inches='tight')
    plt.close('all')

    plt.figure()
    ax = sns.lineplot(data=df_cur, x="Year", y="birth", hue="country", palette="tab20")
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel(r"$\lambda$", fontsize=20)
    ax.get_legend().remove()
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_year_lambda.pdf'), bbox_inches='tight')
    plt.close('all')

    plt.figure()
    ax = sns.lineplot(data=df_cur, x="Year", y="e0", hue="country", palette="tab20")
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel(r"$l$", fontsize=20)
    ax.get_legend().remove()
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_year_l.pdf'), bbox_inches='tight')
    plt.close('all')

    plt.figure()
    ax = sns.lineplot(data=df_cur, x="Year", y="e0*birth_work", hue="country", palette="tab20")
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel(r"$\lambda l \exp(l / 25)$", fontsize=20)
    ax.get_legend().remove()
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_year_lambdalexpl.pdf'), bbox_inches='tight')
    plt.close('all')

    # Calculate the coefficients of variance of "e0" * "birth"  and "e0" * "birth" * "exp(e0)" for each country.
    df_cur[df_cur["Year"] < 1940][["e0*birth", "e0*birth_work"]].std() / df_cur[df_cur["Year"] < 1940][["e0*birth", "e0*birth_work"]].mean()
    df_cur[df_cur["Year"] > 1950][["e0*birth", "e0*birth_work"]].std() / df_cur[df_cur["Year"] > 1950][["e0*birth", "e0*birth_work"]].mean()
