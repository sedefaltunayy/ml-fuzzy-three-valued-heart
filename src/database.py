"""
database.py
-----------
SQLite bağlantı sınıfı ve tablo yönetimi.
5 ana tablo:
  1. raw_heart_data             – yüklenen ham CSV verisi
  2. processed_original_data    – önişlenmiş, binary hedef
  3. fuzzy_modified_data        – bulanık hedef + other_factors
  4. model_results              – eğitilen modellerin metrikleri
  5. user_predictions           – kullanıcı tahmin geçmişi
"""

import sqlite3
import os
import pandas as pd


# ---------------------------------------------------------------------------
# Varsayılan veritabanı yolu
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = os.path.join("database", "heart_disease.db")


class DatabaseManager:
    """SQLite veritabanı bağlantısı ve CRUD işlemleri."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Bağlantı yardımcıları
    # ------------------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Her çağrıda yeni bir bağlantı döndürür."""
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    # Tablo oluşturma
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """5 ana tabloyu oluşturur; zaten varsa dokunmaz."""
        ddl_statements = [
            # 1 – Ham veri
            """
            CREATE TABLE IF NOT EXISTS raw_heart_data (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                age             REAL,
                gender          INTEGER,
                height          REAL,
                weight          REAL,
                ap_hi           REAL,
                ap_lo           REAL,
                cholesterol     INTEGER,
                gluc            INTEGER,
                smoke           INTEGER,
                alco            INTEGER,
                active          INTEGER,
                cardio          INTEGER,
                loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # 2 – Önişlenmiş, binary hedef
            """
            CREATE TABLE IF NOT EXISTS processed_original_data (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                age_years       REAL,
                gender          INTEGER,
                height          REAL,
                weight          REAL,
                bmi             REAL,
                ap_hi           REAL,
                ap_lo           REAL,
                cholesterol     INTEGER,
                gluc            INTEGER,
                smoke           INTEGER,
                alco            INTEGER,
                active          INTEGER,
                cardio          INTEGER
            )
            """,
            # 3 – Bulanık veri seti
            """
            CREATE TABLE IF NOT EXISTS fuzzy_modified_data (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                age_years       REAL,
                gender          INTEGER,
                height          REAL,
                weight          REAL,
                bmi             REAL,
                ap_hi           REAL,
                ap_lo           REAL,
                cholesterol     INTEGER,
                gluc            INTEGER,
                other_factors   REAL,
                fuzzy_target    INTEGER
            )
            """,
            # 4 – Model metrikleri
            """
            CREATE TABLE IF NOT EXISTS model_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_type    TEXT,
                model_name      TEXT,
                accuracy        REAL,
                f1_score        REAL,
                precision_score REAL,
                recall_score    REAL,
                computation_time REAL,
                trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # 5 – Kullanıcı tahmin geçmişi
            """
            CREATE TABLE IF NOT EXISTS user_predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                age_years       REAL,
                gender          INTEGER,
                height          REAL,
                weight          REAL,
                bmi             REAL,
                ap_hi           REAL,
                ap_lo           REAL,
                cholesterol     INTEGER,
                gluc            INTEGER,
                smoke           INTEGER,
                alco            INTEGER,
                active          INTEGER,
                other_factors   REAL,
                fuzzy_target    INTEGER,
                fuzzy_label     TEXT,
                predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        ]

        with self.get_connection() as conn:
            for stmt in ddl_statements:
                conn.execute(stmt)
            conn.commit()

    # ------------------------------------------------------------------
    # Yükleme / okuma yardımcıları
    # ------------------------------------------------------------------

    def to_sql(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
        """DataFrame'i belirtilen tabloya yazar."""
        with self.get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    def read_sql(self, table_name: str) -> pd.DataFrame:
        """Tabloyu DataFrame olarak döndürür."""
        with self.get_connection() as conn:
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)

    def read_sql_query(self, query: str) -> pd.DataFrame:
        """Özel SQL sorgusu çalıştırır."""
        with self.get_connection() as conn:
            return pd.read_sql(query, conn)

    def insert_prediction(self, record: dict) -> None:
        """Tek bir kullanıcı tahminini user_predictions tablosuna ekler."""
        df = pd.DataFrame([record])
        self.to_sql(df, "user_predictions", if_exists="append")

    def insert_model_result(self, record: dict) -> None:
        """Model metriğini model_results tablosuna ekler."""
        df = pd.DataFrame([record])
        self.to_sql(df, "model_results", if_exists="append")

    def table_exists(self, table_name: str) -> bool:
        """Tablo var mı kontrol eder."""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        with self.get_connection() as conn:
            result = conn.execute(query, (table_name,)).fetchone()
        return result is not None

    def row_count(self, table_name: str) -> int:
        """Tablodaki satır sayısını döndürür."""
        if not self.table_exists(table_name):
            return 0
        with self.get_connection() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] if result else 0

    def clear_table(self, table_name: str) -> None:
        """Tablonun tüm verilerini siler (tabloyu korur)."""
        with self.get_connection() as conn:
            conn.execute(f"DELETE FROM {table_name}")
            conn.commit()
