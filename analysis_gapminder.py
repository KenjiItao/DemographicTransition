import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
import geopandas as gpd
import os
import json
import seaborn as sns
import pandas as pd
from shapely.geometry import Point

current_palette = sns.color_palette("colorblind", 7)
sns.set(style='whitegrid')
if True:
    # "#0072b2", "#f0e442", "#009e73", "#d55e00", "#cc79a7"
    current_palette[0] = (255 / 255, 0 / 255, 0 / 255)
    current_palette[1] = (0 / 255, 48 / 255, 163 / 255)
    current_palette[2] = (5 / 255, 163 / 255, 0 / 255)
    current_palette[3] = (213 / 255, 94 / 255, 0 / 255)
    current_palette[4] = (204 / 255, 121 / 255, 167 / 255)
    current_palette[5] = (75 / 255, 0 / 255, 146 / 255)
    current_palette[6] = (0 / 255, 114 / 255, 178 / 255)

current_palette2 = sns.color_palette("colorblind", 8)
sns.set(style='whitegrid')
if True:
    # "#0072b2", "#f0e442", "#009e73", "#d55e00", "#cc79a7"
    current_palette2[0] = (0 / 255, 0 / 255, 0 / 255)
    current_palette2[1] = (255 / 255, 0 / 255, 0 / 255)
    current_palette2[2] = (0 / 255, 48 / 255, 163 / 255)
    current_palette2[3] = (5 / 255, 163 / 255, 0 / 255)
    current_palette2[4] = (211 / 255, 0 / 255, 206 / 255)
    current_palette2[5] = (74 / 255, 190 / 255, 255 / 255)
    current_palette2[6] = (255 / 255, 209 / 255, 39 / 255)
    current_palette2[7] = (219 / 255, 109 / 255, 00 / 255)

# Plot the color palette
sns.palplot(current_palette)
plt.show()

data_dir = "data/gapminder2023"
res_dir = "figs/gapminder2023"
geo_dir = "data/dplace-data-master/geo"

map_df = gpd.read_file(os.path.join(geo_dir,'level2.json'))

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
df_hdi = pd.read_csv(os.path.join(data_dir,'hdi_human_development_index.csv'))
df_hdi = df_hdi.melt(id_vars=["country"], var_name="Year", value_name="hdi")
df_gdp = pd.read_csv(os.path.join(data_dir,'total_gdp_ppp_inflation_adjusted.csv'))
df_gdp = df_gdp.melt(id_vars=["country"], var_name="Year", value_name="gdp")
df_child = pd.read_csv(os.path.join(data_dir,'child_mortality_0_5_year_olds_dying_per_1000_born.csv'))
df_child = df_child.melt(id_vars=["country"], var_name="Year", value_name="child_mortality")
df_education = pd.read_csv(os.path.join(data_dir,'expenditure_per_student_primary_percent_of_gdp_per_person.csv'))
df_education = df_education.melt(id_vars=["country"], var_name="Year", value_name="education")
df_student = pd.read_csv(os.path.join(data_dir,'mean_years_in_school_men_25_to_34_years.csv'))
df_student = df_student.melt(id_vars=["country"], var_name="Year", value_name="student")
# df_child["child_mortality"] = df_child["child_mortality"].astype(float)
df_location = pd.read_csv('data/countries_codes_and_coordinates.csv')


# Merge all dfs
df = df_e0.merge(df_tfr, on=["country", "Year"], how="left")
df = df.merge(df_pop, on=["country", "Year"], how="left")
df = df.merge(df_pop_growth, on=["country", "Year"], how="left")
df = df.merge(df_pop_child, on=["country", "Year"], how="left")
df = df.merge(df_literacy, on=["country", "Year"], how="left")
df = df.merge(df_birth, on=["country", "Year"], how="left")
df = df.merge(df_hdi, on=["country", "Year"], how="left")
df = df.merge(df_gdp, on=["country", "Year"], how="left")
df = df.merge(df_child, on=["country", "Year"], how="left")
df = df.merge(df_education, on=["country", "Year"], how="left")
df = df.merge(df_student, on=["country", "Year"], how="left")
df.head()

# Change datatype of values except for country name to float.
df[["Year", "e0", "tfr",  "literacy", "birth", "hdi", "child_mortality"]] = df[["Year", "e0", "tfr",  "literacy","birth", "hdi", "child_mortality"]].astype(float)
# df[["growth"]] = df[["growth"]].astype(float)

# Use the data for years before 2023.
df = df[df["Year"] < 2023]

# Change "B", "M" and "k" to 1e9, 1e6 and 1e3, respectively.
df["pop"] = df["pop"].str.replace("B", "*1e9").str.replace("M", "*1e6").str.replace("k", "*1e3").apply(pd.eval)
# By noting that, some values are nan for "pop_child", we changed "B", "M" and "k" to 1e9, 1e6 and 1e3, respectively.
df["pop_child"] = df["pop_child"].str.replace("B", "*1e9").str.replace("M", "*1e6").str.replace("k", "*1e3").replace(np.nan, 0).apply(pd.eval)
df["pop_child"] = df["pop_child"].replace(0.0, np.nan)
df["gdp"] = df["gdp"].str.replace("TR", "*1e12").str.replace("B", "*1e9").str.replace("M", "*1e6").str.replace("k", "*1e3").replace(np.nan, 0).apply(pd.eval)
df["gdp"] = df["gdp"].replace(0.0, np.nan)

# # Change string "-" to float "-" in the column "pop_growth".
df["growth"] = df["growth"].str.replace("−", "-").astype(float)

# Calculate working population
df["pop_working"] = df["pop"] - df["pop_child"]

df["pop_growth"] = df.groupby("country")["pop"].pct_change()
df["pop_working_growth"] = df.groupby("country")["pop_working"].pct_change()

# Taking 5-year rolling average for growth rates.
df["pop_growth_rolling"] = df.groupby("country")["pop_growth"].rolling(5, center=True).mean().reset_index(drop=True)
df["pop_growth_rolling2"] = df.groupby("country")["pop_growth"].rolling(5, center=False).mean().reset_index(drop=True)
df["pop_working_growth_rolling"] = df.groupby("country")["pop_working_growth"].rolling(5, center=True).mean().reset_index(drop=True)

df["gdp_per_capita"] = df["gdp"] / df["pop"]
df["e0_growth"] = df.groupby("country")["e0"].pct_change()
df["e0_growth_rolling"] = df.groupby("country")["e0_growth"].rolling(5).mean().reset_index(drop=True)
df["e0_diff"] = df.groupby("country")["e0"].diff()
df["birth_diff"] = df.groupby("country")["birth"].diff()
df["gdp_growth"] = df.groupby("country")["gdp"].pct_change()
df["gdp_per_capita_growth"] = df.groupby("country")["gdp_per_capita"].pct_change()

if True:
    # Draw the scatterplot of "e0" and "birth" for all countries.
    plt.figure()
    # ax = sns.scatterplot(data=df, x="e0", y="birth", color=current_palette[0], s= 30, alpha = 0.1)
    ax = sns.scatterplot(data=df, x="e0", y="birth", color=current_palette2[0], s=20, marker="$\circ$", ec="face", alpha = .7, lw = .05)
    ax.set_xlabel("life expectancy", fontsize=20)
    ax.set_ylabel("crude birth rate", fontsize=20)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir,f'life_birth_scatter_all.pdf'), bbox_inches='tight')
    plt.close('all')

    plt.figure()
    # ax = sns.scatterplot(data=df, x="e0", y="birth", color=current_palette[0], s= 30, alpha = 0.1)
    ax = sns.scatterplot(data=df, x="e0", y="birth", hue="Year", s=20, marker="$\circ$", ec="face", alpha=.7, lw=.05)
    ax.set_xlabel(f"life expectancy $e_0$", fontsize=20)
    ax.set_ylabel(f"crude birth rate $\lambda$", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_scatter_all_year.pdf'), bbox_inches='tight')
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
    # print(results.summary())
    print(results.rsquared)
    constant1 = results.params[0]
    # 1416

    df_cur = df[(df["Year"] >= year_threshold) & (df["e0"] > 60)]
    df_cur = df_cur[["e0", "birth"]].dropna()
    x = df_cur["e0"]
    y = df_cur["birth"]
    # for const in range(15, 30, 1):
    #     model = sm.OLS(y, 1 / np.exp(x / const))
    #     results = model.fit()
    #     print(results.rsquared)
    #     const = 18 maximize r^2
    # for power in range(1, 6, 1):
    #     model = sm.OLS(y, 1 / (x ** power))
    #     results = model.fit()
    #     print(results.rsquared)
    #     power = 4 maximize r^2
    model = sm.OLS(y, 1 / np.exp(x / 18))
    results = model.fit()
    # print(results.summary())
    print(results.rsquared)
    constant2 = results.params[0]
    # 1109

    df_cur = df[(df["Year"] >= year_threshold) & (df["e0"] < 60)]
    df_cur = df_cur[["e0", "birth"]].dropna()
    x = [1] * len(df_cur["e0"])
    y = df_cur["birth"]
    model = sm.OLS(y, x)
    results = model.fit()
    # print(results.summary())
    constant3 = results.params[0]
    # constant3 = 45
    # 43


    x1 = np.linspace(30, 85, 100)
    y1 = constant1 / x1
    master_curve1 = np.c_[x1[y1 < constant3], y1[y1 < constant3]]

    x2 = np.linspace(60, 85, 100)
    y2 = constant2 / (np.exp(x2 / 18))
    master_curve2 = np.c_[x2[y2 < constant3], y2[y2 < constant3]]

    x3 = np.linspace(20, 60, 100)
    y3 = np.ones(len(x2)) * constant3
    master_curve3 = np.c_[x3, y3]

    x1 = np.linspace(25, 70, 100)
    y1 = constant1 / x1
    x2 = np.linspace(55, 90, 100)
    y2 = constant2 / (np.exp(x2 / 18))

    df_cur = df[df["Year"] < year_threshold]
    plt.figure()
    # ax = sns.lineplot(data=df_cur, x="e0", y="birth", color=current_palette[0], errorbar=('ci', 90))
    ax = sns.lineplot(data=df_cur, x="e0", y="birth", color=current_palette[0], errorbar='sd')
    df_cur = df[df["Year"] >= year_threshold]
    # ax = sns.lineplot(data=df_cur, x="e0", y="birth", color=current_palette[1], errorbar=('ci', 90))
    ax = sns.lineplot(data=df_cur, x="e0", y="birth", color=current_palette[1], errorbar='sd')
    ax.plot(x1, y1, color=current_palette[0], linestyle='--')
    ax.plot(x2, y2, color=current_palette[1], linestyle='--')
    ax.set_xlabel(f"life expectancy $e_0$", fontsize=20)
    ax.set_ylabel(f"crude birth rate $\lambda$", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_pathway_isoclines_{year_threshold}.pdf'), bbox_inches='tight')
    plt.close('all')

    plt.figure()
    ax = sns.scatterplot(data=df[df["Year"] >= year_threshold], x="e0", y="birth", color=current_palette[1])
    ax = sns.scatterplot(data=df[df["Year"] < year_threshold], x="e0", y="birth", color=current_palette[0])
    ax.plot(x1, y1, color=current_palette[0], linestyle='--')
    ax.plot(x2, y2, color=current_palette[1], linestyle='--')
    ax.set_xlabel(f"life expectancy $e_0$", fontsize=20)
    ax.set_ylabel(f"crude birth rate $\lambda$", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_scatter_isoclines_{year_threshold}.pdf'), bbox_inches='tight')
    plt.close('all')

    # For each data point, the relative distances from ("e0", "birth") to the master curves are calculated.
    # The distance is calculated as the minimum distance from the master curves.

    for ind in df.index:
        df.loc[ind, "dist1"] = np.min(np.sum((np.array(df.loc[ind, ["e0", "birth"]]) - master_curve1) ** 2, axis=1))
        df.loc[ind, "dist2"] = np.min(np.sum((np.array(df.loc[ind, ["e0", "birth"]]) - master_curve2) ** 2, axis=1))
        df.loc[ind, "dist3"] = np.min(np.sum((np.array(df.loc[ind, ["e0", "birth"]]) - master_curve3) ** 2, axis=1))
    df["relative dist"] = np.log(df["dist1"] / df["dist2"])
    df["type"] = 1 * (df["dist1"] < df["dist2"]) * (df["dist1"] < df["dist3"]) + 2 * (df["dist2"] < df["dist1"]) * (df["dist2"] < df["dist3"]) + 3 * (df["dist3"] < df["dist1"]) * (df["dist3"] < df["dist2"])
    df[["country", "Year", "e0", "birth", "type"]].head(20)
    df[df["Year"] == 2000]["type"].value_counts()
    df[df["country"] == "Japan"][["Year", "e0", "birth", "type"]]

    # Plot temporal change of the type
    # Plot the ratios of each type for each year
    df_cur = df[(df["Year"] >= 1800) & (df["Year"] <= 2015)]
    df_cur = df_cur.groupby(["Year", "type"]).agg({"country": "count"}).reset_index()
    df_cur = df_cur.pivot(index="Year", columns="type", values="country")
    df_cur = df_cur.fillna(0)
    df_cur = df_cur.div(df_cur.sum(axis=1), axis=0)
    df_cur = df_cur.reset_index()
    df_cur = df_cur.melt(id_vars="Year", value_vars=[1, 2, 3])
    df_cur.columns = ["Year", "type", "ratio"]
    plt.figure()
    ax = sns.lineplot(data=df_cur[df_cur["type"] < 3], x="Year", y="ratio", hue="type", palette=current_palette[:2], linewidth = 3)
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel("ratio", fontsize=20)
    # ax.legend(title="Type", loc="upper left", fontsize=16)
    ax.get_legend().remove()
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'transition_type.pdf'), bbox_inches='tight')

def get_line_numbers_concat(line_nums):
    seq = []
    final = []
    last = 0

    for index, val in enumerate(line_nums):
        val = round(val)
        if last + 1 == val or index == 0:
            seq.append(val)
            last = val
        else:
            if len(seq) > 1:
               final.append(str(seq[0]) + '-' + str(seq[len(seq)-1]))
            else:
               final.append(str(seq[0]))
            seq = []
            seq.append(val)
            last = val

        if index == len(line_nums) - 1:
            if len(seq) > 1:
                final.append(str(seq[0]) + '-' + str(seq[len(seq)-1]))
            else:
                final.append(str(seq[0]))

    final_str = ', '.join(map(str, final))
    return final_str

if True:
    # For each country, I identified the first year and durations when the type is 1 or 2.
    # The first year is the first year when the type is 1 or 2.
    # The duration is the number of consecutive years when the type is 1 or 2.
    df_cur = df[(df["e0"] > 40)]
    type1_series = df_cur[df_cur["type"] == 1].groupby("country")["Year"].apply(list)
    type2_series = df_cur[df_cur["type"] == 2].groupby("country")["Year"].apply(list)
    df_country_type = pd.DataFrame(index = ["first year 1", "year type I", "duration type I", "first year 2", "year type II", "duration type II"])
    for country in df["country"].unique():
        if country in type1_series.index:
            year_list1 = type1_series[country]
            first_year1 = year_list1[0]
        else:
            year_list1 = []
            first_year1 = np.nan
        if country in type2_series.index:
            year_list2 = type2_series[country]
            first_year2 = year_list2[0]
        else:
            year_list2 = []
            first_year2 = np.nan
        df_country_type[country] = [first_year1, get_line_numbers_concat(year_list1), len(year_list1), first_year2, get_line_numbers_concat(year_list2), len(year_list2)]
    df_country_type = df_country_type.T
    # df_country_type1 = df_cur[df_cur["type"] == 1].groupby("country").agg({"Year": ["min", "max", "count"]}).astype("int")
    # df_country_type2 = df_cur[df_cur["type"] == 2].groupby("country").agg({"Year": ["min", "max", "count"]}).astype("int")
    # df_country_type = df_country_type1.merge(df_country_type2, left_index=True, right_index=True, suffixes=("_1", "_2"), how = "outer")
    # df_country_type.columns = ["onset year type I", "last year type I", "duration type I", "onset year type II", "last year type II", "duration type II"]
    df_country_type["type"] = 1 * (df_country_type["duration type I"] > df_country_type["duration type II"]) + 1 * (1 - df_country_type["duration type I"].isna()) * (df_country_type["duration type II"].isna()) + 2 * (df_country_type["duration type I"] < df_country_type["duration type II"]) + 2 * (df_country_type["duration type I"].isna()) * (1 - df_country_type["duration type II"].isna())
    # df_country_type.fillna("---", inplace=True)
    # for col in df_country_type.columns:
    #     df_country_type[col] = df_country_type[col].astype("Int64")
    df_country_type["type"].replace({1: "I", 2: "II"}, inplace=True)
    df_country_type.to_csv("country_type.csv")
    df_country_type["type"].replace({"I": 1, "II": 2}, inplace=True)

    # Histogram of the first year when the type is 1 or 2.
    # X-axis is the first year and y-axis is the number of countries.
    # Color indicates the type of countries.
    plt.figure()
    ax = sns.histplot(data=df_country_type[df_country_type["type"] == 1], x="first year 1", color = current_palette[0], bins=30, binrange= (1800, 2020), alpha=0.5)
    sns.histplot(data=df_country_type[df_country_type["type"] == 2], x="first year 2", color = current_palette[1], bins=30, binrange= (1800, 2020), alpha=0.5)
    ax.set_xlabel("first year of transition", fontsize=20)
    ax.set_ylabel("number of countries", fontsize=20)
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir,f'life_birth_transition_first_year.pdf'), bbox_inches='tight')
    plt.close('all')

if True:
    # Draw the scatterplot of "e0" and "birth" for all countries.
    # X-axis is "e0" and y-axis is "birth". Colors indicate the type.
    plt.figure()
    ax = sns.scatterplot(data=df[(df["type"] > 0) & (df["type"] < 3)], x="e0", y="birth", hue="type", palette=current_palette, s=20, marker="$\circ$", ec="face", alpha = .7, lw = .05)
    ax = sns.scatterplot(data=df[(df["type"] == 3)], x = "e0", y = "birth", c = current_palette2[0], s = 20, marker = "$\circ$", ec = "face", alpha = .7, lw = .05)
    ax.set_xlabel(f"life expectancy $e_0$", fontsize=20)
    ax.set_ylabel(f"crude birth rate $\lambda$", fontsize=20)
    ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir,f'life_birth_scatter_type.pdf'), bbox_inches='tight')
    plt.close('all')

if True:
    # Draw the lineplot of "e0" and "birth" for some countries.
    # X-axis is "e0" and y-axis is "birth". Colors indicate the country.
    country_ls = ["Sweden", "Italy",  "Myanmar", "Iran", "Kenya", "South Korea"]
    x1 = np.linspace(30, 90, 100)
    y1 = constant1 / x1
    x2 = np.linspace(55, 90, 100)
    y2 = constant2 / (np.exp(x2 / 18))

    plt.figure()
    ax = plt.gca()
    ax.plot(x1, y1, color=current_palette[0], linestyle='--', linewidth=4)
    ax.plot(x2, y2, color=current_palette[1], linestyle='--', linewidth=4)
    for country in country_ls:
        df_cur = df[df["country"] == country]
        ax.plot(df_cur["e0"], df_cur["birth"], label=country, linewidth=2.8)
    ax.set_xlabel(f"life expectancy $e_0$", fontsize=20)
    ax.set_ylabel(f"crude birth rate $\lambda$", fontsize=20)
    ax.legend()
    # ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir,f'life_birth_lines.pdf'), bbox_inches='tight')
    plt.close('all')

if True:
    col = "education"
    plt.figure()
    ax = sns.scatterplot(data=df[df["type"] == 1], x="e0", y =col, color=current_palette[0], alpha=0.5)
    sns.regplot(data=df[df["type"] == 1], x="e0", y =col, color=current_palette[0], scatter=False, ax = ax)
    sns.scatterplot(data=df[df["type"] == 2], x="e0", y =col, color=current_palette[1], alpha=0.5, ax = ax)
    sns.regplot(data=df[df["type"] == 2], x="e0", y=col, color=current_palette[1], scatter=False, ax=ax)
    ax.set_xlabel(r"$e_0$", fontsize=20)
    ax.set_ylabel("average education investment", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_transition_{col}.pdf'), bbox_inches='tight')
    plt.close('all')

    col = "student"
    plt.figure()
    ax = sns.scatterplot(data=df[df["type"] == 1], x="e0", y=col, color=current_palette[0], alpha=0.5)
    sns.regplot(data=df[df["type"] == 1], x="e0", y=col, color=current_palette[0], scatter=False, ax=ax)
    sns.scatterplot(data=df[df["type"] == 2], x="e0", y=col, color=current_palette[1], alpha=0.5, ax=ax)
    sns.regplot(data=df[df["type"] == 2], x="e0", y=col, color=current_palette[1], scatter=False, ax=ax)
    ax.set_xlabel(r"$e_0$", fontsize=20)
    ax.set_ylabel("average education duration", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_transition_{col}.pdf'), bbox_inches='tight')
    plt.close('all')

if True:
    col = "e0_diff"
    plt.figure()
    df_cur = df[(df["type"] == 1) & (df[col] != 0)]
    ax = sns.histplot(data=df_cur, x=col,binrange=(-3, 3),
                      color=current_palette[0], bins=50, alpha=0.5, weights=1 / len(df_cur.index))
    df_cur = df[(df["type"] == 2) & (df[col] != 0)]
    sns.histplot(data=df_cur, x=col, color=current_palette[1],binrange=(-3, 3),
                 bins=50, alpha=0.5, ax=ax, weights=1 / len(df_cur.index))
    ax.set_xlabel(r"$e_0$ growth", fontsize=20)
    ax.set_ylabel("")
    # ax.set_ylabel(" of countries", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_transition_{col}.pdf'), bbox_inches='tight')
    plt.close('all')

    col = "birth_diff"
    plt.figure()
    df_cur = df[(df["type"] == 1) & (df[col] != 0)]
    ax = sns.histplot(data=df_cur, x=col,
                      color=current_palette[0], bins=50, alpha=0.5, binrange=(-3, 3),weights=1 / len(df_cur.index))
    df_cur = df[(df["type"] == 2) & (df[col] != 0)]
    sns.histplot(data=df_cur, x=col, color=current_palette[1],binrange=(-3, 3),
                 bins=50, alpha=0.5, ax=ax, weights=1 / len(df_cur.index))
    ax.set_xlabel(r"$\lambda$ growth", fontsize=20)
    ax.set_ylabel("")
    # ax.set_ylabel("number of countries", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_transition_{col}.pdf'), bbox_inches='tight')
    plt.close('all')

    col = "child_mortality"
    plt.figure()
    df_cur = df[(df["type"] == 1) & (df[col] != 0)]
    ax = sns.histplot(data=df_cur, x=col,
                      color=current_palette[0], bins=50, alpha=0.5, weights=1 / len(df_cur.index))
    df_cur = df[(df["type"] == 2) & (df[col] != 0)]
    sns.histplot(data=df_cur, x=col, color=current_palette[1],
                 bins=30, alpha=0.5, ax=ax, weights=1 / len(df_cur.index))
    ax.set_xlabel(r"Child mortality", fontsize=20)
    ax.set_ylabel("")
    # ax.set_ylabel("number of countries", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_transition_{col}.pdf'), bbox_inches='tight')
    plt.close('all')

    col = "gdp_growth"
    plt.figure()
    df_cur = df[(df["type"] == 1) & (df[col] != 0)]
    ax = sns.histplot(data=df_cur, x=col,
                      color=current_palette[0], bins=50, alpha=0.5, binrange=(-0.1, 0.15),
                      weights=1 / len(df_cur.index))
    df_cur = df[(df["type"] == 2) & (df[col] != 0)]
    sns.histplot(data=df_cur, x=col, color=current_palette[1],
                 bins=50, alpha=0.5, ax=ax, binrange=(-0.1, 0.15), weights=1 / len(df_cur.index))
    ax.set_xlabel(r"GDP growth", fontsize=20)
    ax.set_ylabel("")
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_transition_{col}.pdf'), bbox_inches='tight')
    plt.close('all')

    col = "pop_growth"
    plt.figure()
    df_cur = df[(df["type"] == 1) & (df[col] != 0)]
    ax = sns.histplot(data=df_cur, x=col,
                      color=current_palette[0], bins=50, alpha=0.5, binrange=(-0.02, 0.05), weights=1 / len(df_cur.index))
    df_cur = df[(df["type"] == 2) & (df[col] != 0)]
    sns.histplot(data=df_cur, x=col, color=current_palette[1], binrange=(-0.02, 0.05),
                 bins=50, alpha=0.5, ax=ax, weights=1 / len(df_cur.index))
    ax.set_xlabel(r"Population growth", fontsize=20)
    ax.set_ylabel("")
    # ax.set_ylabel("number of countries", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_transition_{col}.pdf'), bbox_inches='tight')
    plt.close('all')

    col = "hdi"
    plt.figure()
    ax = sns.histplot(data=df[df["type"] == 1], x=col,
                      color=current_palette[0], bins=100, alpha=0.5, weights=1 / len(df[df["type"] == 1].index))
    sns.histplot(data=df[df["type"] == 2], x=col, color=current_palette[1],
                 bins=100, alpha=0.5, ax=ax, weights=1 / len(df[df["type"] == 2].index))
    # sns.histplot(data=df[df["type"] == 3], x=col, color=current_palette[2],
    #              bins=30, alpha=0.5, ax=ax)
    ax.set_xlabel(f"{col} during transition", fontsize=20)
    ax.set_ylabel("ratio of countries", fontsize=20)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_during_transition_{col}.pdf'), bbox_inches='tight')
    plt.close('all')

if True:
    df_location2 = df_location[["Country", "Longitude (average)", "Latitude (average)"]]
    df_location2.columns = ["country", "longitude", "latitude"]

    geo_df = gpd.GeoDataFrame(index=["type", "name", "marker", "marker-color", "marker-size", "geometry", "year"])

    for key in df_transition["country"]:
        try:
            geo_df[len(geo_df.columns)] = ["Feature", key, "o", df_transition[df_transition["country"] == key]["type"], "small", Point([df_location2[df_location2["country"] == key]["longitude"], df_location2[df_location2["country"] == key]["latitude"]]), df_transition[df_transition["country"] == key]["Year"]]
        except:
            pass
        # geo_df[len(geo_df.columns)] = ["Feature", key, "o", "small", df_transition.at[key, ], Point([tdwg[key]["lon"], tdwg[key]["lat"]])]
    geo_df = geo_df.T

    if True:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        map_df.plot(ax=ax, color="grey")
        geo_df.plot(ax=ax, markersize=5, marker="o", color = df_transition["type"].map({1: current_palette[0], 2: current_palette[1]}))
        # geo_df[geo_df["marker"] == "v"].plot(ax = ax, color = geo_df[geo_df["marker"] == "v"]["marker-color"], markersize = 20, marker = "^")
        # geo_df[geo_df["marker"] == "*"].plot(ax = ax, color = geo_df[geo_df["marker"] == "*"]["marker-color"], markersize = 20, marker = "*")
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig("figs/transition_worldmap.pdf", bbox_inches='tight')
        plt.close('all')

    geo_df = gpd.GeoDataFrame(index=["type", "name", "marker", "marker-color", "marker-size", "geometry", "year"])

    for key in df["country"].unique():
        try:
            if (df[(df["country"] == key) & (df["Year"] == 2015)]["type"] == 1).tolist()[0]:
                geo_df[len(geo_df.columns)] = ["Feature", key, "o", current_palette[0],
                                           "small", Point([df_location2[df_location2["country"] == key]["longitude"],
                                                           df_location2[df_location2["country"] == key]["latitude"]]),
                                           2015]
            elif (df[(df["country"] == key) & (df["Year"] == 2015)]["type"] == 2).tolist()[0]:
                geo_df[len(geo_df.columns)] = ["Feature", key, "o", current_palette[1], "small",
                                               Point([df_location2[df_location2["country"] == key]["longitude"],
                                                      df_location2[df_location2["country"] == key]["latitude"]]),
                                               2015]
        except:
            pass
        # geo_df[len(geo_df.columns)] = ["Feature", key, "o", "small", df_transition.at[key, ], Point([tdwg[key]["lon"], tdwg[key]["lat"]])]
    geo_df = geo_df.T


    if True:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        map_df.plot(ax=ax, color="grey")
        geo_df.plot(ax=ax, markersize=5, marker="o",
                    color=geo_df["marker-color"])
        # geo_df[geo_df["marker"] == "v"].plot(ax = ax, color = geo_df[geo_df["marker"] == "v"]["marker-color"], markersize = 20, marker = "^")
        # geo_df[geo_df["marker"] == "*"].plot(ax = ax, color = geo_df[geo_df["marker"] == "*"]["marker-color"], markersize = 20, marker = "*")
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig("figs/transition_worldmap_2015.pdf", bbox_inches='tight')
        plt.close('all')

    geo_df = gpd.GeoDataFrame(index=["type", "name", "marker", "marker-color", "marker-size", "geometry"])
    df_country_type.fillna(0, inplace=True)
    my_palette = sns.color_palette("vlag", n_colors=101)
    key = "Myanmar"
    for key in df["country"].unique():
        try:
            duration_ratio = df_country_type.at[key, "duration 1"] / (df_country_type.at[key, "duration 1"] + df_country_type.at[key, "duration 2"])
            geo_df[len(geo_df.columns)] = ["Feature", key, "o", my_palette[int(duration_ratio * 100)],
                                           "small",
                                           Point([df_location2[df_location2["country"] == key]["longitude"],
                                                  df_location2[df_location2["country"] == key]["latitude"]])]

        except:
            pass
        # geo_df[len(geo_df.columns)] = ["Feature", key, "o", "small", df_transition.at[key, ], Point([tdwg[key]["lon"], tdwg[key]["lat"]])]
    geo_df = geo_df.T

    if True:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        map_df.plot(ax=ax, color="grey")
        geo_df.plot(ax=ax, markersize=5, marker="o",
                    color=geo_df["marker-color"])
        # geo_df[geo_df["marker"] == "v"].plot(ax = ax, color = geo_df[geo_df["marker"] == "v"]["marker-color"], markersize = 20, marker = "^")
        # geo_df[geo_df["marker"] == "*"].plot(ax = ax, color = geo_df[geo_df["marker"] == "*"]["marker-color"], markersize = 20, marker = "*")
        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig("figs/transition_worldmap_ratio.pdf", bbox_inches='tight')
        plt.close('all')


if True:
    # Draw the pathway of "e0" * "birth" for each country._
    # X-axis is "year" and y-axis is the product of "e0" and "birth". Colors indicate the country.
    df_cur = df[["country", "Year", "e0", "birth", "type"]].dropna()
    df_cur["e0*birth"] = df_cur["e0"] * df_cur["birth"]
    df_cur["e0*birth_work"] = df_cur["e0"] * df_cur["birth"] * np.exp(df_cur["e0"] / 25)
    plt.figure()
    ax = sns.lineplot(data=df_cur, x="Year", y="e0*birth", hue="country", palette="tab20")
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel(r"$\lambda e_0$", fontsize=20)
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
    ax.set_ylabel(r"$e_0$", fontsize=20)
    ax.get_legend().remove()
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_year_l.pdf'), bbox_inches='tight')
    plt.close('all')

    # Calculate the coefficients of variance of "e0" * "birth"  and "e0" * "birth" * "exp(e0)" for each country.
    df_cur[df_cur["Year"] < 1940][["e0*birth", "e0*birth_work"]].std() / df_cur[df_cur["Year"] < 1940][["e0*birth", "e0*birth_work"]].mean()
    df_cur[df_cur["Year"] > 1950][["e0*birth", "e0*birth_work"]].std() / df_cur[df_cur["Year"] > 1950][["e0*birth", "e0*birth_work"]].mean()

    plt.figure()
    ax = sns.lineplot(data=df_cur[df_cur["Year"] > 2000], x="Year", y="birth", hue="country", palette="tab20")
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel(r"$\lambda$", fontsize=20)
    ax.get_legend().remove()
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_year_lambda_2000.pdf'), bbox_inches='tight')
    plt.close('all')

    plt.figure()
    ax = sns.lineplot(data=df[df["Year"] > 2000], x="Year", y="tfr", hue="country", palette="tab20")
    ax.set_xlabel("year", fontsize=20)
    ax.set_ylabel("TFR", fontsize=20)
    ax.get_legend().remove()
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, f'life_birth_year_TFR_2000.pdf'), bbox_inches='tight')
    plt.close('all')


