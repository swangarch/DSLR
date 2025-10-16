import pandas as pd
import sys
from load_csv import load



def find_median(sorted_li: list):
    """Find the median of an array."""

    length = len(sorted_li)
    if length % 2 == 1:
        return sorted_li[int(length / 2)]
    else:
        mi_idx = int(length / 2)
        return (sorted_li[mi_idx - 1] + sorted_li[mi_idx]) / 2.0
    

# def quartile(sorted_li: list, quartile_num: int) -> float:
#     """Given a list, calculate its quartile"""

#     length = len(sorted_li)

#     # One definition of quartile
#     if quartile_num == 1:
#         sub_li_l = sorted_li[0:length // 2 + 1]
#         return find_median(sub_li_l)
#     elif quartile_num == 3:
#         sub_li_r = sorted_li[length // 2:]
#         return find_median(sub_li_r)

#     # if length % 2 == 1:
#     #     if quartile_num == 1:
#     #         sub_li_l = sorted_li[0:length // 2]
#     #         return find_median(sub_li_l)
#     #     elif quartile_num == 3:
#     #         sub_li_r = sorted_li[length // 2 + 1:]
#     #         return find_median(sub_li_r)
#     # else:
#     #     if quartile_num == 1:
#     #         sub_li_l = sorted_li[0:length // 2]
#     #         return find_median(sub_li_l)
#     #     elif quartile_num == 3:
#     #         sub_li_r = sorted_li[length // 2:]
#     #         return find_median(sub_li_r)


def quartile(data: list[float], q: float) -> float:
    
    n = len(data)
    pos = (n - 1) * q
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    fraction = pos - lower
    
    return data[lower] + fraction * (data[upper] - data[lower])


def ft_describe_one(series: pd.Series, title: str, all: dict) -> None:
    """Function to describe the data series."""

    sum,  count, mean, err = 0.0, 0, 0.0, 0.0
    cleanLi = []
    for elem in series: 
        if elem == elem:
            sum += elem
            cleanLi.append(elem)

    cleanLi = sorted(cleanLi)
    count = len(cleanLi)
    mean = sum / count
    for num in cleanLi:
        err += (num - mean) ** 2
    std = (err / count) ** 0.5
    min_num = cleanLi[0]
    max_num = cleanLi[-1]
    percent25 = quartile(cleanLi, 0.24)
    percent50 = quartile(cleanLi, 0.50)
    percent75 = quartile(cleanLi, 0.75)

    all[title] = [count, mean, std, min_num, percent25, percent50, percent75, max_num]


def ft_describe(series: pd.Series)-> None:
    """Describe all features"""

    all = {"Feature": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}

    print("\n--------------------my describe----------------------")
    for col in series:
        ft_describe_one(series[col], col, all)
    df_describ = pd.DataFrame(all)
    df_describ = df_describ.set_index("Feature")
    df_describ.index.name = None
    print(df_describ)


def additional_info(series: pd.Series):
    """Show some additional info."""

    print("\n--------------------additional describe----------------------")
    for col in series:
        s = set()
        
        for elem in series[col]:
            if elem is not None:
                s.add(elem)

        show_set = str(list(s)) if len(list(s)) <= 5 else (str(list(s)[:5]) + "...")

        print(f"\033[33m<FEATURE--{col}>:\033[0m {show_set} \033[33m<feature type => {len(s)}>\033[0m")


def main():
    """Test of reading data and print it."""

    try:
        print("\033[33mUsage: python3 describe.py <path_csv>\033[0m")
        argv = sys.argv
        assert len(argv) == 2, "Wrong argument number."

        pd.set_option('display.float_format', '{:.6f}'.format)
        df = load(argv[1])
        if df is None:
            sys.exit(1)
 
        df_num = df.select_dtypes(include="number")
        ft_describe(df_num)
        # additional_info(df)


    except KeyboardInterrupt:
        print("\033[33mStopped by user.\033[0m")
        sys.exit(1)
    
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()