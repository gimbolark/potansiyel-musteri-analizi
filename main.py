import pandas as pd
import numpy as np
import os
import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QMessageBox, QGroupBox, QScrollArea, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

class MLApp(QMainWindow):
    log_signal = pyqtSignal(str)
    training_finished_signal = pyqtSignal()
    prediction_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gezinomi Fiyat Tahmin Uygulaması (PyQt5)")
        self.setGeometry(100, 100, 900, 800)

        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.best_model = None
        self.y_pred = None
        self.y_train_pred = None

        self.plot_window = None
        self.current_figure = None

        self.initUI()
        self.log_message("Uygulama başlatıldı. Lütfen 'Veriyi Yükle ve Ön İşle' butonuna basın.")

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        log_group_box = QGroupBox("Log Çıktısı")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group_box.setLayout(log_layout)
        main_layout.addWidget(log_group_box, 2)

        self.log_signal.connect(self.update_log)
        self.training_finished_signal.connect(self.on_training_finished)
        self.prediction_finished_signal.connect(self.on_prediction_finished)

        button_layout = QHBoxLayout()

        self.load_button = QPushButton("Veriyi Yükle ve Ön İşle")
        self.load_button.clicked.connect(self.load_and_preprocess_data_threaded)
        button_layout.addWidget(self.load_button)

        self.train_button = QPushButton("Modeli Eğit")
        self.train_button.clicked.connect(self.start_training_thread)
        self.train_button.setEnabled(False)
        button_layout.addWidget(self.train_button)

        self.predict_button = QPushButton("Modeli Çalıştır (Test)")
        self.predict_button.clicked.connect(self.start_prediction_thread)
        self.predict_button.setEnabled(False)
        button_layout.addWidget(self.predict_button)

        self.plot_button = QPushButton("Model Grafiğini Göster")
        self.plot_button.clicked.connect(self.show_model_plot)
        self.plot_button.setEnabled(False)
        button_layout.addWidget(self.plot_button)

        main_layout.addLayout(button_layout)

        results_group_box = QGroupBox("Model Performans Sonuçları")
        results_layout = QVBoxLayout()

        self.mae_label = QLabel("MAE: N/A")
        results_layout.addWidget(self.mae_label)
        self.rmse_label = QLabel("RMSE: N/A")
        results_layout.addWidget(self.rmse_label)
        self.r2_label = QLabel("R²: N/A")
        results_layout.addWidget(self.r2_label)

        results_layout.addSpacing(10)

        self.mae_train_label = QLabel("MAE (Eğitim): N/A")
        results_layout.addWidget(self.mae_train_label)
        self.rmse_train_label = QLabel("RMSE (Eğitim): N/A")
        results_layout.addWidget(self.rmse_train_label)
        self.r2_train_label = QLabel("R² (Eğitim): N/A")
        results_layout.addWidget(self.r2_train_label)

        results_group_box.setLayout(results_layout)
        main_layout.addWidget(results_group_box, 1)

    @pyqtSlot(str)
    def update_log(self, message):
        self.log_text.append(message)

    def log_message(self, message):
        self.log_signal.emit(message)

    def load_and_preprocess_data_threaded(self):
        self.log_message("Veri yükleniyor ve ön işleniyor...")
        self.load_button.setEnabled(False)
        threading.Thread(target=self._load_and_preprocess_data, daemon=True).start()

    def _load_and_preprocess_data(self):
        file_path = 'ai.xlsx'
        if not os.path.exists(file_path):
            self.log_message(f"HATA: {file_path} bulunamadı. Lütfen dosyanın uygulamanın çalıştığı klasörde olduğundan emin olun.")
            self.load_button.setEnabled(True)
            return

        try:
            self.df = pd.read_excel(file_path)
            self.log_message("Veri başarıyla yüklendi.")

            if 'SaleDate' in self.df.columns:
                self.df['SaleDate'] = pd.to_datetime(self.df['SaleDate'], errors='coerce')
            if 'CheckInDate' in self.df.columns:
                self.df['CheckInDate'] = pd.to_datetime(self.df['CheckInDate'], errors='coerce')

            self.df.dropna(subset=['Price'], inplace=True)
            self.log_message("Eksik fiyat değerleri temizlendi.")

            self.log_message("Öznitelikler türetiliyor...")

            self.df['CheckInDayCategory'] = self.df['CInDay'].apply(
                lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday'
            )
            day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                         'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            self.df['CheckInDayNumber'] = self.df['CInDay'].map(day_order)
            city_freq = self.df['SaleCityName'].value_counts(normalize=True)
            self.df['CityPopularity'] = self.df['SaleCityName'].map(city_freq)
            self.df['SeasonBinary'] = self.df['Seasons'].map({'Low': 0, 'High': 1})
            self.df['CheckInMonth'] = self.df['CheckInDate'].dt.month
            self.df['CheckInYear'] = self.df['CheckInDate'].dt.year
            self.df['SaleMonth'] = self.df['SaleDate'].dt.month
            self.df['SaleYear'] = self.df['SaleDate'].dt.year

            national_holidays = [
                "2015-01-01", "2015-04-23", "2015-05-01", "2015-05-19", "2015-08-30", "2015-10-29",
                "2016-01-01", "2016-04-23", "2016-05-01", "2016-05-19", "2016-08-30", "2016-10-29",
                "2017-01-01", "2017-04-23", "2017-05-01", "2017-05-19", "2017-08-30", "2017-10-29",
                "2018-01-01", "2018-04-23", "2018-05-01", "2018-05-19", "2018-08-30", "2018-10-29",
                "2019-01-01", "2019-04-23", "2019-05-01", "2019-05-19", "2019-08-30", "2019-10-29",
                "2020-01-01", "2020-04-23", "2020-05-01", "2020-05-19", "2020-08-30", "2020-10-29",
                "2021-01-01", "2021-04-23", "2021-05-01", "2021-05-19", "2021-08-30", "2021-10-29",
                "2022-01-01", "2022-04-23", "2022-05-01", "2022-05-19", "2022-08-30", "2022-10-29",
                "2023-01-01", "2023-04-23", "2023-05-01", "2023-05-19", "2023-08-30", "2023-10-29"
            ]

            religious_holidays = [
                "2015-07-17", "2015-07-18", "2015-07-19", "2015-09-24", "2015-09-25", "2015-09-26",
                "2016-07-05", "2016-07-06", "2016-07-07", "2016-09-12", "2016-09-13", "2016-09-14",
                "2017-06-25", "2017-06-26", "2017-06-27", "2017-09-01", "2017-09-02", "2017-09-03",
                "2018-06-15", "2018-06-16", "2018-06-17", "2018-08-21", "2018-08-22", "2018-08-23",
                "2019-06-04", "2019-06-05", "2019-06-06", "2019-08-11", "2019-08-12", "2019-08-13",
                "2020-05-24", "2020-05-25", "2020-05-26", "2020-07-31", "2020-08-01", "2020-08-02",
                "2021-05-13", "2021-05-14", "2021-05-15", "2021-07-20", "2021-07-21", "2021-07-22",
                "2022-05-02", "2022-05-03", "2022-05-04", "2022-07-09", "2022-07-10", "2022-07-11",
                "2023-04-21", "2023-04-22", "2023-04-23", "2023-06-28", "2023-06-29", "2023-06-30"
            ]

            holiday_dates_combined = pd.to_datetime(national_holidays + religious_holidays)
            self.df['IsHolidayCheckIn'] = self.df['CheckInDate'].isin(holiday_dates_combined).astype(int)

            national_holidays_dt = pd.to_datetime(national_holidays)
            religious_holidays_dt = pd.to_datetime(religious_holidays)

            self.df['IsNationalHoliday'] = self.df['CheckInDate'].isin(national_holidays_dt).astype(int)
            self.df['IsReligiousHoliday'] = self.df['CheckInDate'].isin(religious_holidays_dt).astype(int)

            holiday_range_dates = set()
            for h_date in holiday_dates_combined:
                for i in range(-7, 8):
                    holiday_range_dates.add(h_date + pd.Timedelta(days=i))
            self.df['IsInHolidayRangeCheckIn'] = self.df['CheckInDate'].dt.normalize().isin(holiday_range_dates).astype(int)

            concept_price_avg = self.df.groupby('ConceptName')['Price'].mean()
            self.df['ConceptScore'] = self.df['ConceptName'].map(concept_price_avg)

            city_price_avg = self.df.groupby('SaleCityName')['Price'].mean()
            self.df['CityAvgPrice'] = self.df['SaleCityName'].map(city_price_avg)

            def categorize_lead(x):
                if pd.isna(x): return 'Unknown'
                elif x <= 7: return 'Late'
                elif x <= 30: return 'Normal'
                else: return 'Early'

            self.df['BookingLeadCategory'] = self.df['SaleCheckInDayDiff'].apply(categorize_lead)

            self.df['Season_Day'] = self.df['Seasons'] + "_" + self.df['CInDay']

            concept_group = self.df.groupby('ConceptName')['Price']
            self.df['ConceptZScore'] = (self.df['Price'] - concept_group.transform('mean')) / (concept_group.transform('std') + 1e-6)

            self.log_message("Öznitelikler türetildi.")

            features = [
                'ConceptName', 'SaleCityName', 'CInDay', 'SaleCheckInDayDiff', 'Seasons',
                'CheckInDayNumber', 'SeasonBinary', 'CheckInMonth','CheckInYear', 'SaleMonth', 'SaleYear',
                'IsReligiousHoliday', 'ConceptScore', 'CityAvgPrice', 'BookingLeadCategory',
                'ConceptZScore', 'IsInHolidayRangeCheckIn'
            ]
            target = 'Price'

            X = self.df[features]
            y = self.df[target]

            self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            self.log_message(f"Veri eğitim ({len(self.X_train)} örnek) ve test ({len(self.X_test)} örnek) setlerine ayrıldı.")

            self.train_button.setEnabled(True)

        except Exception as e:
            self.log_message(f"Veri yükleme veya ön işleme sırasında hata oluştu: {e}")
        finally:
            self.load_button.setEnabled(True)

    def start_training_thread(self):
        self.log_message("Model eğitimi başlatılıyor... Bu biraz zaman alabilir.")
        self.train_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.plot_button.setEnabled(False)
        threading.Thread(target=self._train_model, daemon=True).start()

    def _train_model(self):
        try:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
                ],
                remainder='passthrough'
            )

            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', xgb_model)
            ])

            param_grid = {
                'regressor__n_estimators': [100, 300, 500, 700, 1000],
                'regressor__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'regressor__max_depth': [3, 5, 7, 9, 11],
                'regressor__min_child_weight': [1, 3, 5, 7],
                'regressor__gamma': [0, 0.1, 0.2, 0.3, 0.4],
                'regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'regressor__lambda': [8, 16, 160], # L2 regülarizasyon terimi (aşırı öğrenmeyi önler)
                'regressor__alpha': [8, 16, 160],
                'regressor__tree_method': ['hist'],
                'regressor__grow_policy': ['depthwise', 'lossguide']
            }

            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=20,
                cv=5,
                scoring='neg_mean_absolute_error',
                verbose=0,
                n_jobs=-1,
                random_state=42
            )

            random_search.fit(self.X_train, self.y_train)
            self.best_model = random_search.best_estimator_

            self.log_message("Model eğitimi tamamlandı.")
            self.training_finished_signal.emit()

        except Exception as e:
            self.log_message(f"Model eğitimi sırasında hata oluştu: {e}")
            self.training_finished_signal.emit()

    @pyqtSlot()
    def on_training_finished(self):
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)

    def start_prediction_thread(self):
        self.log_message("Model test verisi üzerinde çalıştırılıyor ve performans değerlendiriliyor...")
        self.predict_button.setEnabled(False)
        self.plot_button.setEnabled(False)
        threading.Thread(target=self._predict_on_test_data, daemon=True).start()

    def _predict_on_test_data(self):
        if self.best_model is None:
            self.log_message("Hata: Model eğitilmemiş. Lütfen önce 'Modeli Eğit' butonuna basın.")
            self.prediction_finished_signal.emit()
            return

        try:
            self.y_pred = self.best_model.predict(self.X_test)
            self.y_train_pred = self.best_model.predict(self.X_train)

            # Test Seti Metrikleri
            mae_test = mean_absolute_error(self.y_test, self.y_pred)
            rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
            r2_test = r2_score(self.y_test, self.y_pred)

            self.log_message("\n--- Nihai Model Performansı (Test Seti) ---")
            self.log_message(f"MAE : {mae_test:.2f}")
            self.log_message(f"RMSE: {rmse_test:.2f}")
            self.log_message(f"R²  : {r2_test:.2f}")

            # Eğitim Seti Metrikleri
            mae_train = mean_absolute_error(self.y_train, self.y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
            r2_train = r2_score(self.y_train, self.y_train_pred)

            self.log_message("\n--- Eğitim Seti Performansı ---")
            self.log_message(f"MAE Eğitim: {mae_train:.2f}")
            self.log_message(f"RMSE Eğitim: {rmse_train:.2f}")
            self.log_message(f"R² Eğitim : {r2_train:.2f}")

            self.mae_label.setText(f"MAE: {mae_test:.2f}")
            self.rmse_label.setText(f"RMSE: {rmse_test:.2f}")
            self.r2_label.setText(f"R²: {r2_test:.2f}")
            self.mae_train_label.setText(f"MAE (Eğitim): {mae_train:.2f}")
            self.rmse_train_label.setText(f"RMSE (Eğitim): {rmse_train:.2f}")
            self.r2_train_label.setText(f"R² (Eğitim): {r2_train:.2f}")

            self.log_message("\n--- Gerçek ve Tahmin Edilen Fiyatlardan Örnekler (Test Seti) ---")
            self.log_message("--------------------------------------------------")
            self.log_message("Gerçek Fiyat | Tahmin Edilen Fiyat | Fark")
            self.log_message("--------------------------------------------------")

            # İlk N (örneğin 20) değeri yazdır
            N = 20
            for i in range(min(N, len(self.y_test))):
                actual = self.y_test.iloc[i]
                predicted = self.y_pred[i]
                difference = actual - predicted
                self.log_message(f"{actual:<12.2f} | {predicted:<19.2f} | {difference:<.2f}")
            self.log_message("--------------------------------------------------\n")


            self.log_message("Model çalıştırma ve değerlendirme tamamlandı.")
            self.prediction_finished_signal.emit()

        except Exception as e:
            self.log_message(f"Model çalıştırma veya değerlendirme sırasında hata oluştu: {e}")
            self.prediction_finished_signal.emit()

    @pyqtSlot()
    def on_prediction_finished(self):
        self.predict_button.setEnabled(True)
        if self.best_model is not None and self.y_pred is not None:
            self.plot_button.setEnabled(True)
            self.show_model_plot()

    def show_model_plot(self):
        if self.y_pred is None or self.y_test is None:
            self.log_message("Hata: Grafik için gerçek veya tahmin edilen değerler bulunamadı. Lütfen önce 'Modeli Çalıştır (Test)' butonuna basın.")
            self.plot_button.setEnabled(True)
            return

        self.log_message("Model grafiği oluşturuluyor...")
        self.plot_button.setEnabled(False)

        try:
            if self.plot_window is not None:
                self.plot_window.close()
                self.plot_window = None
            if self.current_figure is not None:
                plt.close(self.current_figure)
                self.current_figure = None

            self.plot_window = QWidget()
            self.plot_window.setWindowTitle("Gerçek vs Tahmin Edilen Fiyatlar")
            self.plot_window.setGeometry(200, 200, 800, 600)
            plot_layout = QVBoxLayout()

            self.current_figure, ax = plt.subplots(figsize=(8, 5))

            Q1 = np.percentile(self.y_test, 25)
            Q3 = np.percentile(self.y_test, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            mask = (self.y_test >= lower_bound) & (self.y_test <= upper_bound)
            y_test_filtered = self.y_test[mask]
            y_pred_filtered = self.y_pred[mask]

            ax.scatter(y_test_filtered, y_pred_filtered, alpha=0.5, label='Gerçek vs Tahmin Edilen')
            min_val = min(y_test_filtered.min(), y_pred_filtered.min())
            max_val = max(y_test_filtered.max(), y_pred_filtered.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='İdeal Tahmin (y=x)')

            ax.set_xlabel("Gerçek Fiyatlar (Outlier'lar Hariç)")
            ax.set_ylabel("Tahmin Edilen Fiyatlar (Outlier'lar Hariç)")
            ax.set_title("Gerçek vs Tahmin Edilen Fiyatlar Dağılımı")
            ax.grid(True)
            ax.legend()

            canvas = FigureCanvas(self.current_figure)
            plot_layout.addWidget(canvas)

            toolbar = NavigationToolbar(canvas, self.plot_window)
            plot_layout.addWidget(toolbar)

            self.plot_window.setLayout(plot_layout)
            self.plot_window.show()

            self.plot_window.destroyed.connect(lambda: plt.close(self.current_figure))

            self.log_message("Model grafiği gösterildi.")

        except Exception as e:
            self.log_message(f"Grafik oluşturulurken hata oluştu: {e}")
        finally:
            self.plot_button.setEnabled(True)
            if self.plot_window is not None:
                self.plot_window.activateWindow()
                self.plot_window.raise_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MLApp()
    ex.show()
    sys.exit(app.exec_())