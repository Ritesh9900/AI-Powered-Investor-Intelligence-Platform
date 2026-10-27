from sqlalchemy import text

from database.postgres_sql import get_engine


def get_metrics():
    engine = get_engine()

    query = """
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY company, year
                   ORDER BY created_at DESC
               ) AS rn
        FROM financial_metrics
    ) t
    WHERE rn = 1
    ORDER BY company
    """

    with engine.connect() as connection:
        result = connection.execute(text(query))
        rows = []
        for row in result:
            d = dict(row._mapping)
            if "created_at" in d and d["created_at"] is not None:
                d["created_at"] = str(d["created_at"])
            rows.append(d)

    return rows