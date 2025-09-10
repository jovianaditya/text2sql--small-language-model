import psycopg2
import yaml
from psycopg2 import sql

# Load your YAML descriptions file
with open("descriptions.yml", "r") as f:
    descriptions_dict = yaml.safe_load(f)

# Map your YAML keys to actual PostgreSQL table names
TABLE_MAP = {
    "frpm": "frpm",
    "satscores": "satscores",
    "schools": "schools",
}

conn = psycopg2.connect(
    host="188.166.245.16",
    database="california_schools",
    user="datahub",
    password="datahub"
)

with conn:
    with conn.cursor() as cursor:
        for dataset_name, descriptions in descriptions_dict.items():
            table_name = TABLE_MAP.get(dataset_name)
            if not table_name:
                print(f"No table mapping found for dataset '{dataset_name}'")
                continue

            for column, description in descriptions.items():
                try:
                    query = sql.SQL("COMMENT ON COLUMN {}.{} IS {}").format(
                        sql.Identifier(table_name),
                        sql.Identifier(column),
                        sql.Literal(description)
                    )
                    cursor.execute(query)
                    print(f"Updated comment on {table_name}.{column}")
                except Exception as e:
                    print(f"Failed to update comment on {table_name}.{column}: {e}")
                    # If an error occurs, rollback that statement and continue with others
                    conn.rollback()
                    cursor.execute("BEGIN")

conn.close()