import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---  Data Loading ---
file_path = 'ai.xlsx'

if not os.path.exists(file_path):
    print(f"HATA: '{file_path}' dosyası bulunamadı.")
    print("Lütfen Excel dosyasının doğru yolda olduğundan emin olun.")
    exit()

print(f"'{file_path}' dosyasından veri okunuyor...")
try:
    df = pd.read_excel(file_path)
    print("Veri başarıyla okundu.")
except ImportError:
    print("HATA: 'openpyxl' kütüphanesi bulunamadı.")
    print("Lütfen 'pip install openpyxl' komutu ile kurun.")
    exit()
except Exception as e:
    print(f"HATA: Excel dosyası okunurken bir hata oluştu: {e}")
    exit()

print("--- Initial Data Sample ---")
print(df.head())
print("\n--- Data Information (before detailed processing) ---")
print(df.info())

# --- Data Preprocessing ---

# Convert date columns to datetime format
if 'SaleDate' in df.columns:
    df['SaleDate'] = pd.to_datetime(df['SaleDate'])
else:
    print("WARNING: 'SaleDate' column not found. Some analyses might be skipped.")
if 'CheckInDate' in df.columns:
    df['CheckInDate'] = pd.to_datetime(df['CheckInDate'])
else:
    print("WARNING: 'CheckInDate' column not found. Some analyses might be skipped.")

# Ensure Price is numeric
if 'Price' in df.columns:
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Price'] = df['Price'] * 40
    print("\n'Price' sütunu 40 ile çarpıldı.")
else:
    print("WARNING: 'Price' column not found. Some analyses might not run.")

# Missing data check and handling for 'Price'
print("\nEksik Veri Kontrolü:")
print(df.isnull().sum())

initial_rows = len(df)
if 'Price' in df.columns:
    df.dropna(subset=['Price'], inplace=True)
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"\n'Price' sütununda eksik değer içeren {removed_rows} satır kaldırıldı.")
        print(f"Kalan satır sayısı: {len(df)}")
else:
    print("Skipping Price NaN removal as 'Price' column was not found.")

print("\n--- Removing Price Outliers (2.5 IQR Method) ---")
if 'Price' in df.columns:
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    

    #lower_bound = Q1 - (1.5 * IQR) 
    lower_bound = 1
    #upper_bound = 450 * 40
    upper_bound = Q3 + (1.5 * IQR)
    
    initial_rows_before_outlier_removal = len(df)
    df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)].copy() # .copy() to avoid SettingWithCopyWarning
    rows_removed_by_outlier = initial_rows_before_outlier_removal - len(df)
    
    print(f"{rows_removed_by_outlier} rows removed as outliers from 'Price' using 2.5 IQR method.")
    print(f"Remaining rows after outlier removal: {len(df)}")
else:
    print("Skipping outlier removal as 'Price' column was not found.")


# Extract SaleYearMonth, SaleMonth, SaleYear
if 'SaleDate' in df.columns:
    df['SaleYearMonth'] = df['SaleDate'].dt.to_period('M')
    df['SaleMonth'] = df['SaleDate'].dt.month
    df['SaleYear'] = df['SaleDate'].dt.year
else:
    print("Skipping SaleYearMonth, SaleMonth, SaleYear extraction as 'SaleDate' column was not found.")

# --- 2.1 Feature Engineering: Holiday Flags ---
if 'CheckInDate' in df.columns:
    national_holidays = [
        # 2015–2023
        "2015-01-01", "2015-04-23", "2015-05-01", "2015-05-19", "2015-08-30", "2015-10-29",
        "2016-01-01", "2016-04-23", "2016-05-01", "2016-05-19", "2016-08-30", "2016-10-29",
        "2017-01-01", "2017-04-23", "2017-05-01", "2017-05-19", "2017-08-30", "2017-10-29",
        "2018-01-01", "2018-04-23", "2018-05-01", "2018-05-19", "2018-08-30", "2018-10-29",
        "2019-01-01", "2019-04-23", "2019-05-01", "2019-05-19", "2019-08-30", "2019-10-29",
        "2020-01-01", "2020-04-23", "2020-05-01", "2020-05-19", "2020-08-30", "2020-10-29",
        "2021-01-01", "2021-04-23", "2021-05-01", "2021-05-19", "2021-08-30", "2021-10-29",
        "2022-01-01", "2022-04-23", "2022-05-01", "2022-05-19", "2022-08-30", "2022-10-29",
        "2023-01-01", "2023-04-21", "2023-04-23", "2023-05-01", "2023-05-19", "2023-08-30", "2023-10-29"
    ]

    religious_holidays = [
        # Ramazan Bayramı ve Kurban Bayramı (2015–2023)
        "2015-07-17", "2015-07-18", "2015-07-19",
        "2015-09-24", "2015-09-25", "2015-09-26",

        "2016-07-05", "2016-07-06", "2016-07-07",
        "2016-09-12", "2016-09-13", "2016-09-14",

        "2017-06-25", "2017-06-26", "2017-06-27",
        "2017-09-01", "2017-09-02", "2017-09-03",

        "2018-06-15", "2018-06-16", "2018-06-17",
        "2018-08-21", "2018-08-22", "2018-08-23",

        "2019-06-04", "2019-06-05", "2019-06-06",
        "2019-08-11", "2019-08-12", "2019-08-13",

        "2020-05-24", "2020-05-25", "2020-05-26",
        "2020-07-31", "2020-08-01", "2020-08-02",

        "2021-05-13", "2021-05-14", "2021-05-15",
        "2021-07-20", "2021-07-21", "2021-07-22",

        "2022-05-02", "2022-05-03", "2022-05-04",
        "2022-07-09", "2022-07-10", "2022-07-11",

        "2023-04-21", "2023-04-22", "2023-04-23",
        "2023-06-28", "2023-06-29", "2023-06-30"
    ]

    national_holidays_dt = pd.to_datetime(national_holidays)
    religious_holidays_dt = pd.to_datetime(religious_holidays)
    all_holiday_dates = pd.to_datetime(list(set(national_holidays + religious_holidays)))

    df['IsHolidayCheckIn'] = df['CheckInDate'].isin(all_holiday_dates).astype(int)
    df['IsNationalHoliday'] = df['CheckInDate'].isin(national_holidays_dt).astype(int)
    df['IsReligiousHoliday'] = df['CheckInDate'].isin(religious_holidays_dt).astype(int)

    holiday_range_dates = set()
    for h_date in all_holiday_dates:
        for i in range(-7, 8): # From -7 days to +7 days, inclusive
            holiday_range_dates.add(h_date + pd.Timedelta(days=i))
    
    # Convert CheckInDate to just date for comparison with the set of holiday_range_dates
    df['IsInHolidayRangeCheckIn'] = df['CheckInDate'].dt.normalize().isin(holiday_range_dates).astype(int)

    print("\n--- Holiday Features Created ---")
    print(df[['CheckInDate', 'IsHolidayCheckIn', 'IsNationalHoliday', 'IsReligiousHoliday', 'IsInHolidayRangeCheckIn']].head())
else:
    print("\nSkipping Holiday Feature Engineering as 'CheckInDate' column was not found.")

# --- 2.2 Additional Feature Engineering ---

# Hafta içi/sonu kategorisi
if 'CInDay' in df.columns:
    df['CheckInDayCategory'] = df['CInDay'].apply(
        lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday'
    )
    print("\n'CheckInDayCategory' created.")
else:
    print("\nSkipping 'CheckInDayCategory' creation as 'CInDay' column was not found.")

# Gün numarası
if 'CInDay' in df.columns:
    day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['CheckInDayNumber'] = df['CInDay'].map(day_order)
    print("'CheckInDayNumber' created.")
else:
    print("Skipping 'CheckInDayNumber' creation as 'CInDay' column was not found.")

# Şehir popülaritesi
if 'SaleCityName' in df.columns:
    city_freq = df['SaleCityName'].value_counts(normalize=True)
    df['CityPopularity'] = df['SaleCityName'].map(city_freq)
    print("'CityPopularity' created.")
else:
    print("Skipping 'CityPopularity' creation as 'SaleCityName' column was not found.")

# Sezon binary
if 'Seasons' in df.columns:
    df['SeasonBinary'] = df['Seasons'].map({'Low': 0, 'High': 1, 'Mid': 0}).fillna(0).astype(int)
    print("'SeasonBinary' created.")
else:
    print("Skipping 'SeasonBinary' creation as 'Seasons' column was not found.")

# Tarih Bilgisi (CheckInMonth, CheckInYear)
if 'CheckInDate' in df.columns and pd.api.types.is_datetime64_any_dtype(df['CheckInDate']):
    df['CheckInMonth'] = df['CheckInDate'].dt.month
    df['CheckInYear'] = df['CheckInDate'].dt.year
    print("'CheckInMonth' and 'CheckInYear' created.")

    # Calculate and print average price by CheckInMonth
    if 'Price' in df.columns:
        avg_price_by_checkin_month = df.groupby('CheckInMonth')['Price'].mean().reset_index()
        print("\n--- Average Price by Check-In Month ---")
        print(avg_price_by_checkin_month)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_price_by_checkin_month, x='CheckInMonth', y='Price', palette='viridis')
        plt.title('Average Price by Check-In Month')
        plt.xlabel('Check-In Month')
        plt.ylabel('Average Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('avg_price_by_checkin_month.png')
        plt.show()

    # Calculate and print average price by CheckInYear
    if 'Price' in df.columns:
        avg_price_by_checkin_year = df.groupby('CheckInYear')['Price'].mean().reset_index()
        print("\n--- Average Price by Check-In Year ---")
        print(avg_price_by_checkin_year)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_price_by_checkin_year, x='CheckInYear', y='Price', palette='plasma')
        plt.title('Average Price by Check-In Year')
        plt.xlabel('Check-In Year')
        plt.ylabel('Average Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('avg_price_by_checkin_year.png')
        plt.show()
else:
    print("UYARI: 'CheckInDate' sütunu bulunamadı veya datetime formatında değil. 'CheckInMonth', 'CheckInYear' türetilemedi.")

# Tarih Bilgisi (SaleMonth, SaleYear)
if 'SaleDate' in df.columns and pd.api.types.is_datetime64_any_dtype(df['SaleDate']):
    if 'SaleMonth' not in df.columns:
        df['SaleMonth'] = df['SaleDate'].dt.month
    if 'SaleYear' not in df.columns:
        df['SaleYear'] = df['SaleDate'].dt.year
    print("'SaleMonth' and 'SaleYear' ensured/created.")

    # Calculate and print average price by SaleMonth
    if 'Price' in df.columns:
        avg_price_by_sale_month = df.groupby('SaleMonth')['Price'].mean().reset_index()
        print("\n--- Average Price by Sale Month ---")
        print(avg_price_by_sale_month)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_price_by_sale_month, x='SaleMonth', y='Price', palette='cividis')
        plt.title('Average Price by Sale Month')
        plt.xlabel('Sale Month')
        plt.ylabel('Average Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('avg_price_by_sale_month.png')
        plt.show()

    # Calculate and print average price by SaleYear
    if 'Price' in df.columns:
        avg_price_by_sale_year = df.groupby('SaleYear')['Price'].mean().reset_index()
        print("\n--- Average Price by Sale Year ---")
        print(avg_price_by_sale_year)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_price_by_sale_year, x='SaleYear', y='Price', palette='magma')
        plt.title('Average Price by Sale Year')
        plt.xlabel('Sale Year')
        plt.ylabel('Average Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('avg_price_by_sale_year.png')
        plt.show()
else:
    print("UYARI: 'SaleDate' sütunu bulunamadı veya datetime formatında değil. 'SaleMonth', 'SaleYear' türetilemedi.")


# Konsept ortalama fiyat
if 'ConceptName' in df.columns and 'Price' in df.columns:
    concept_price_avg = df.groupby('ConceptName')['Price'].mean()
    df['ConceptScore'] = df['ConceptName'].map(concept_price_avg)
    print("'ConceptScore' created.")
else:
    print("Skipping 'ConceptScore' creation as 'ConceptName' or 'Price' column was not found.")

# Şehir ortalama fiyat
if 'SaleCityName' in df.columns and 'Price' in df.columns:
    city_price_avg = df.groupby('SaleCityName')['Price'].mean()
    df['CityAvgPrice'] = df['SaleCityName'].map(city_price_avg)
    print("'CityAvgPrice' created.")
else:
    print("Skipping 'CityAvgPrice' creation as 'SaleCityName' or 'Price' column was not found.")

# Rezervasyon süresi kategorisi
if 'SaleCheckInDayDiff' in df.columns:
    def categorize_lead(x):
        if pd.isna(x): return 'Unknown'
        elif x <= 7: return 'Late'
        elif x <= 30: return 'Normal'
        else: return 'Early'
    df['BookingLeadCategory'] = df['SaleCheckInDayDiff'].apply(categorize_lead)
    print("'BookingLeadCategory' created.")
else:
    print("Skipping 'BookingLeadCategory' creation as 'SaleCheckInDayDiff' column was not found.")

# Sezon + Gün birleşik kategorisi
if 'Seasons' in df.columns and 'CInDay' in df.columns:
    df['Season_Day'] = df['Seasons'] + "_" + df['CInDay']
    print("'Season_Day' created.")
else:
    print("Skipping 'Season_Day' creation as 'Seasons' or 'CInDay' column was not found.")

# Konsept z-score (sapma)
if 'ConceptName' in df.columns and 'Price' in df.columns:
    concept_group_counts = df['ConceptName'].value_counts()
    concepts_with_enough_data = concept_group_counts[concept_group_counts >= 2].index
    
    if not concepts_with_enough_data.empty:
        df_filtered_for_zscore = df[df['ConceptName'].isin(concepts_with_enough_data)].copy()
        concept_group = df_filtered_for_zscore.groupby('ConceptName')['Price']
        df['ConceptZScore'] = (df['Price'] - concept_group.transform('mean')) / concept_group.transform('std')
        df['ConceptZScore'].fillna(0, inplace=True)
        print("'ConceptZScore' created.")
    else:
        print("Skipping 'ConceptZScore' creation: Not enough data points per 'ConceptName' to calculate standard deviation.")
else:
    print("Skipping 'ConceptZScore' creation as 'ConceptName' or 'Price' column was not found.")


print("\n--- Data after Preprocessing & All Feature Engineering ---")
print(df.info())
print("\n--- Processed Data Sample with New Features ---")
print(df.head())


if 'Price' in df.columns:
    print("\n--- Basic Statistics for Price (After Outlier Removal) ---")
    print(df['Price'].describe())

    # Price-wise Histogram and Box Plot (After Outlier Removal)
    print("\n--- Price Distribution Analysis (Histogram and Box Plot - After Outlier Removal) ---")
    plt.figure(figsize=(14, 6))

    # Histogram
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    sns.histplot(df['Price'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Price (Histogram) - After Outlier Removal')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Box Plot
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    sns.boxplot(y=df['Price'], color='lightcoral')
    plt.title('Distribution of Price (Box Plot) - After Outlier Removal')
    plt.ylabel('Price')
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.savefig('price_distribution_after_outlier_removal.png')
    plt.show()

else:
    print("\nSkipping Price statistics and distribution plots as 'Price' column was not found.")

# --- Sales Over Time (Historical Trends) ---
if 'SaleYearMonth' in df.columns and 'Price' in df.columns:
    print("\n--- Sales Over Time Analysis ---")
    monthly_sales = df.groupby('SaleYearMonth')['Price'].sum().reset_index()
    monthly_sales['SaleYearMonth'] = monthly_sales['SaleYearMonth'].astype(str)

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=monthly_sales, x='SaleYearMonth', y='Price', marker='o')
    plt.title('Total Sales Price Over Time (Monthly)')
    plt.xlabel('Sale Year-Month')
    plt.ylabel('Total Sales Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('total_sales_over_time.png')
    plt.show()
else:
    print("\nSkipping 'Sales Over Time' analysis due to missing 'SaleYearMonth' or 'Price' column.")

# --- Sales by Concept Name ---
if 'ConceptName' in df.columns and 'Price' in df.columns:
    print("\n--- Sales by Concept Name ---")
    sales_by_concept = df.groupby('ConceptName')['Price'].sum().sort_values(ascending=False).reset_index()
    avg_price_by_concept = df.groupby('ConceptName')['Price'].mean().sort_values(ascending=False).reset_index()

    print("Total Sales by Concept:")
    print(sales_by_concept)
    print("\nAverage Price by Concept:")
    print(avg_price_by_concept)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=sales_by_concept, x='ConceptName', y='Price', palette='viridis')
    plt.title('Total Sales Price by Concept Name')
    plt.xlabel('Concept Name')
    plt.ylabel('Total Sales Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sales_by_concept.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_price_by_concept, x='ConceptName', y='Price', palette='viridis')
    plt.title('Average Price by Concept Name')
    plt.xlabel('Concept Name')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('avg_price_by_concept.png')
    plt.show()
else:
    print("\nSkipping 'Sales by Concept Name' analysis due to missing 'ConceptName' or 'Price' column.")

# --- Sales by City ---
if 'SaleCityName' in df.columns and 'Price' in df.columns:
    print("\n--- Sales by City ---")
    sales_by_city = df.groupby('SaleCityName')['Price'].sum().sort_values(ascending=False).reset_index()
    avg_price_by_city = df.groupby('SaleCityName')['Price'].mean().sort_values(ascending=False).reset_index()

    print("Total Sales by City:")
    print(sales_by_city)
    print("\nAverage Price by City:")
    print(avg_price_by_city)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=sales_by_city, x='SaleCityName', y='Price', palette='mako')
    plt.title('Total Sales Price by Sale City Name')
    plt.xlabel('Sale City Name')
    plt.ylabel('Total Sales Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sales_by_city.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_price_by_city, x='SaleCityName', y='Price', palette='mako')
    plt.title('Average Price by Sale City Name')
    plt.xlabel('Sale City Name')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('avg_price_by_city.png')
    plt.show()
else:
    print("\nSkipping 'Sales by City' analysis due to missing 'SaleCityName' or 'Price' column.")

# --- Sales by Season ---
if 'Seasons' in df.columns and 'Price' in df.columns:
    print("\n--- Sales by Season ---")
    sales_by_season = df.groupby('Seasons')['Price'].sum().sort_values(ascending=False).reset_index()
    avg_price_by_season = df.groupby('Seasons')['Price'].mean().sort_values(ascending=False).reset_index()

    print("Total Sales by Season:")
    print(sales_by_season)
    print("\nAverage Price by Season:")
    print(avg_price_by_season)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=sales_by_season, x='Seasons', y='Price', palette='rocket')
    plt.title('Total Sales Price by Season')
    plt.xlabel('Season')
    plt.ylabel('Total Sales Price')
    plt.tight_layout()
    plt.savefig('sales_by_season.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_price_by_season, x='Seasons', y='Price', palette='rocket')
    plt.title('Average Price by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('avg_price_by_season.png')
    plt.show()
else:
    print("\nSkipping 'Sales by Season' analysis due to missing 'Seasons' or 'Price' column.")

# --- Sales by Check-In Day of Week (CInDay) ---
if 'CInDay' in df.columns and 'Price' in df.columns:
    print("\n--- Sales by Check-In Day of Week ---")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_filtered_cinday = df[df['CInDay'].isin(day_order)]

    if not df_filtered_cinday.empty:
        sales_by_cinday = df_filtered_cinday.groupby('CInDay')['Price'].sum().reindex(day_order).reset_index()
        avg_price_by_cinday = df_filtered_cinday.groupby('CInDay')['Price'].mean().reindex(day_order).reset_index()

        print("Total Sales by Check-In Day:")
        print(sales_by_cinday)
        print("\nAverage Price by Check-In Day:")
        print(avg_price_by_cinday)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=sales_by_cinday, x='CInDay', y='Price', palette='cubehelix')
        plt.title('Total Sales Price by Check-In Day of Week')
        plt.xlabel('Check-In Day of Week')
        plt.ylabel('Total Sales Price')
        plt.tight_layout()
        plt.savefig('sales_by_cinday.png')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(data=avg_price_by_cinday, x='CInDay', y='Price', palette='cubehelix')
        plt.title('Average Price by Check-In Day of Week')
        plt.xlabel('Check-In Day of Week')
        plt.ylabel('Average Price')
        plt.tight_layout()
        plt.savefig('avg_price_by_cinday_avg.png')
        plt.show()
    else:
        print("No valid 'CInDay' data for analysis.")
else:
    print("\nSkipping 'Sales by Check-In Day of Week' analysis due to missing 'CInDay' or 'Price' column.")


# --- Impact of SaleCheckInDayDiff ---
if 'SaleCheckInDayDiff' in df.columns and 'Price' in df.columns:
    print("\n--- Impact of SaleCheckInDayDiff ---")
    sales_by_diff = df.groupby('SaleCheckInDayDiff')['Price'].mean().reset_index()

    print("Average Price by SaleCheckInDayDiff:")
    print(sales_by_diff)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=sales_by_diff, x='SaleCheckInDayDiff', y='Price', marker='o')
    plt.title('Average Price vs. Sale-CheckIn Day Difference')
    plt.xlabel('Sale-CheckIn Day Difference')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_price_by_day_diff.png')
    plt.show()
else:
    print("\nSkipping 'Impact of SaleCheckInDayDiff' analysis due to missing 'SaleCheckInDayDiff' or 'Price' column.")

if 'IsHolidayCheckIn' in df.columns and 'Price' in df.columns:
    print("\n--- Sales by Holiday Check-In Status ---")
    sales_by_holiday = df.groupby('IsHolidayCheckIn')['Price'].mean().reset_index()
    sales_by_holiday['IsHolidayCheckIn'] = sales_by_holiday['IsHolidayCheckIn'].map({0: 'Non-Holiday', 1: 'Holiday'})
    print("Average Price by Holiday Check-In:")
    print(sales_by_holiday)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sales_by_holiday, x='IsHolidayCheckIn', y='Price', palette='Pastel1')
    plt.title('Average Price: Holiday vs. Non-Holiday Check-In')
    plt.xlabel('Check-In on Holiday')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('avg_price_by_holiday.png')
    plt.show()

    print("\n--- Sales by National Holiday Check-In Status ---")
    sales_by_national_holiday = df.groupby('IsNationalHoliday')['Price'].mean().reset_index()
    sales_by_national_holiday['IsNationalHoliday'] = sales_by_national_holiday['IsNationalHoliday'].map({0: 'Non-National Holiday', 1: 'National Holiday'})
    print("Average Price by National Holiday Check-In:")
    print(sales_by_national_holiday)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sales_by_national_holiday, x='IsNationalHoliday', y='Price', palette='Pastel2')
    plt.title('Average Price: National Holiday vs. Non-National Holiday Check-In')
    plt.xlabel('Check-In on National Holiday')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('avg_price_by_national_holiday.png')
    plt.show()

    print("\n--- Sales by Religious Holiday Check-In Status ---")
    sales_by_religious_holiday = df.groupby('IsReligiousHoliday')['Price'].mean().reset_index()
    sales_by_religious_holiday['IsReligiousHoliday'] = sales_by_religious_holiday['IsReligiousHoliday'].map({0: 'Non-Religious Holiday', 1: 'Religious Holiday'})
    print("Average Price by Religious Holiday Check-In:")
    print(sales_by_religious_holiday)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sales_by_religious_holiday, x='IsReligiousHoliday', y='Price', palette='Pastel2')
    plt.title('Average Price: Religious Holiday vs. Non-Religious Holiday Check-In')
    plt.xlabel('Check-In on Religious Holiday')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('avg_price_by_religious_holiday.png')
    plt.show()

    print("\n--- Sales by Holiday Range Check-In Status (+/- 7 days) ---")
    sales_by_holiday_range = df.groupby('IsInHolidayRangeCheckIn')['Price'].mean().reset_index()
    sales_by_holiday_range['IsInHolidayRangeCheckIn'] = sales_by_holiday_range['IsInHolidayRangeCheckIn'].map({0: 'Outside Holiday Range', 1: 'Within Holiday Range'})
    print("Average Price by Holiday Range Check-In:")
    print(sales_by_holiday_range)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sales_by_holiday_range, x='IsInHolidayRangeCheckIn', y='Price', palette='coolwarm')
    plt.title('Average Price: Within vs. Outside Holiday Range Check-In (+/- 7 days)')
    plt.xlabel('Check-In within Holiday Range')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('avg_price_by_holiday_range.png')
    plt.show()

else:
    print("\nSkipping Holiday features EDA as 'IsHolidayCheckIn' or 'Price' column was not found.")

if 'CheckInDayCategory' in df.columns and 'Price' in df.columns:
    print("\n--- Sales by Check-In Day Category (Weekday/Weekend) ---")
    sales_by_day_cat = df.groupby('CheckInDayCategory')['Price'].mean().reset_index()
    print("Average Price by Check-In Day Category:")
    print(sales_by_day_cat)
    plt.figure(figsize=(7, 5))
    sns.barplot(data=sales_by_day_cat, x='CheckInDayCategory', y='Price', palette='tab10')
    plt.title('Average Price by Check-In Day Category')
    plt.xlabel('Check-In Day Category')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('avg_price_by_day_category.png')
    plt.show()
else:
    print("\nSkipping 'Sales by Check-In Day Category' EDA due to missing 'CheckInDayCategory' or 'Price' column.")

if 'CityPopularity' in df.columns and 'Price' in df.columns:
    print("\n--- Impact of City Popularity on Average Price (Top 10 Cities) ---")
    city_avg_price = df.groupby('SaleCityName')['Price'].mean().sort_values(ascending=False).head(10).reset_index()
    print("Average Price by City (Top 10):")
    print(city_avg_price)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=city_avg_price, x='SaleCityName', y='Price', palette='GnBu_d')
    plt.title('Average Price by City (Top 10)')
    plt.xlabel('Sale City Name')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('avg_price_by_city_popularity.png')
    plt.show()
else:
    print("\nSkipping 'City Popularity' EDA due to missing 'CityPopularity' or 'Price' column.")

if 'BookingLeadCategory' in df.columns and 'Price' in df.columns:
    print("\n--- Sales by Booking Lead Category ---")
    booking_lead_avg_price = df.groupby('BookingLeadCategory')['Price'].mean().reindex(['Late', 'Normal', 'Early', 'Unknown']).reset_index()
    print("Average Price by Booking Lead Category:")
    print(booking_lead_avg_price)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=booking_lead_avg_price, x='BookingLeadCategory', y='Price', palette='cool')
    plt.title('Average Price by Booking Lead Category')
    plt.xlabel('Booking Lead Category')
    plt.ylabel('Average Price')
    plt.tight_layout()
    plt.savefig('avg_price_by_booking_lead_category.png')
    plt.show()
else:
    print("\nSkipping 'Sales by Booking Lead Category' EDA due to missing 'BookingLeadCategory' or 'Price' column.")

if 'Season_Day' in df.columns and 'Price' in df.columns:
    print("\n--- Sales by Season and Day Combination ---")
    season_day_avg_price = df.groupby('Season_Day')['Price'].mean().sort_values(ascending=False).reset_index()
    if len(season_day_avg_price) > 20:
        print("Showing top 20 Season_Day combinations by average price.")
        season_day_avg_price = season_day_avg_price.head(20)

    print("Average Price by Season_Day Combination:")
    print(season_day_avg_price)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=season_day_avg_price, x='Season_Day', y='Price', palette='Spectral')
    plt.title('Average Price by Season_Day Combination')
    plt.xlabel('Season_Day Combination')
    plt.ylabel('Average Price')
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig('avg_price_by_season_day.png')
    plt.show()
else:
    print("\nSkipping 'Sales by Season and Day Combination' EDA due to missing 'Season_Day' or 'Price' column.")


# Updating features list to include all new engineered features
features = [
    'ConceptName', 'SaleCityName', 'CInDay', 'SaleCheckInDayDiff', 'Seasons',
    'IsHolidayCheckIn', 'IsNationalHoliday', 'IsReligiousHoliday', 'IsInHolidayRangeCheckIn',
    'CheckInDayCategory', 'CheckInDayNumber', 'CityPopularity', 'SeasonBinary',
    'CheckInMonth', 'CheckInYear', 'SaleMonth', 'SaleYear',
    'ConceptScore', 'CityAvgPrice', 'BookingLeadCategory', 'Season_Day', 'ConceptZScore'
]
target = 'Price'

